# -*- coding: utf-8 -*-
"""
基于大模型的在线课程答疑系统 —— Streamlit 前端
=============================================

功能模块：
* 多会话管理：每次新建对话/主页提问都会生成带独立会话 ID 的全新会话，会话之间
  消息完全隔离；历史会话持久化到 sessions_store.json，可在侧边栏列表查看与切换
* 主页：新会话发起模式（横幅、统计指标、推荐问题快捷入口与问题输入框）
* 独立问答页：展示当前会话完整历史，答案实时流式输出并自动滚动跟随
* 「➕ 新建对话」「🏠 回到主页」导航 + 系统状态指示（DeepSeek 连通性一键检测）

运行方式：
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
from streamlit.components.v1 import html as components_html

from llm_client import (
    DEFAULT_MODEL,
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekConfigError,
    DeepSeekNetworkError,
)
import course_catalog as catalog

# --------------------------------------------------------------------------- #
# 页面基础配置
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="智能课程答疑系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 课程助教的系统提示词：限定助手的身份、边界与答疑风格
SYSTEM_PROMPT = (
    "你是一名专业的在线课程的智能答疑助教，服务对象为正在大连交通大学学习大学专业课程的学生。"
    "请你遵循以下原则回答问题：\n"
    "1. 先给出直接、准确的结论,如果学生需要详细解释,再分步骤讲解原理,必要时可配合简单示例；\n"
    "2. 善用类比、图示(Markdown 表格/列表)帮助理解,语言亲切、鼓励学生思考；\n"
    "3. 学生概念混淆时,主动指出易错点并给出辨析;如果学生需要,可在答完后提出 1 个延伸思考题；\n"
    "4. 只回答与课程学习相关的问题,对无关或无法确定的内容请如实说明,不要编造事实；\n"
    "5. 使用简体中文回答,回复Markdown排版整洁,引用专业文献时请注明出处。"
)

# --------------------------------------------------------------------------- #
# 回答偏好预设：把技术参数（temperature / max_tokens）转换为用户友好的选项，
# 选择结果即时映射回 API 参数，并以系统指令约束模型的表达风格与篇幅。
# --------------------------------------------------------------------------- #
REPLY_STYLES: Dict[str, dict] = {
    # 名称: 采样温度 / 一句话说明（界面读数）/ 注入系统提示词的风格指令
    "轻快": {
        "temp": 0.9,
        "desc": "轻松明快、亲切有活力",
        "instruction": "请用轻松明快、亲切活泼的语气回答,节奏轻快,可适度使用口语化表达与少量 emoji。",
    },
    "幽默": {
        "temp": 1.1,
        "desc": "风趣生动、善用比喻",
        "instruction": "请在保证知识准确的前提下用风趣幽默的方式讲解,善用生活化比喻和俏皮话,让内容更有记忆点,但不要喧宾夺主。",
    },
    "严谨": {
        "temp": 0.3,
        "desc": "精确克制、逻辑严密",
        "instruction": "请用严谨精确的学术化表达回答,注重概念边界、定义准确性与逻辑严密性,少用口语和修辞,结论必须有依据。",
    },
    "专业": {
        "temp": 0.5,
        "desc": "规范专业、详略得当",
        "instruction": "请保持专业规范、客观均衡的讲解风格,术语使用准确,深度与可读性兼顾,这是默认风格。",
    },
    "简洁": {
        "temp": 0.4,
        "desc": "直击要点、拒绝冗余",
        "instruction": "请极度精简地回答,先给结论,再用最少的要点说明,不做多余展开和寒暄,能用一句话说清就不用两句。",
    },
}
STYLE_ORDER = ["轻快", "幽默", "严谨", "专业", "简洁"]

REPLY_LENGTHS: Dict[str, dict] = {
    "简短": {
        "tokens": 512,
        "desc": "约 150 字内，只讲核心结论",
        "instruction": "篇幅要求：回答控制在 150 字以内，只保留核心结论与最关键的要点，省略示例与延展。",
    },
    "中等": {
        "tokens": 1024,
        "desc": "约 300 字，结论加简要说明",
        "instruction": "篇幅要求：回答控制在 300 字左右，给出结论与简要原理，必要时配一个短小的示例。",
    },
    "详细": {
        "tokens": 2048,
        "desc": "完整讲解，含步骤与示例",
        "instruction": "篇幅要求：回答可以充分展开，包含结论、分步骤原理、代码或示例，以及易错点提示。",
    },
    "超长": {
        "tokens": 4096,
        "desc": "深度剖析，体系化展开",
        "instruction": "篇幅要求：进行体系化的深度讲解，全面覆盖背景、原理、步骤、多个示例、对比辨析与练习建议，篇幅不限。",
    },
}
LENGTH_ORDER = ["简短", "中等", "详细", "超长"]

SUGGESTED_QUESTIONS = [
    "帮我讲讲高等数学的相关内容",
    "我的课程忘选/漏选了,该怎么办",
    "我告诉你课程名称,帮我出几道专业练习题呗",
    "热烈庆祝我校建校70周年!"
]

# --------------------------------------------------------------------------- #
# 设计系统：学术靛蓝主题（与 .streamlit/config.toml 主题令牌配套）
# --------------------------------------------------------------------------- #
st.markdown(
    r"""
    <style>
        /* ===================== 设计令牌 ===================== */
        :root {
            --brand-50:#EEF0FF; --brand-100:#E0E3FF; --brand-500:#4F46E5;
            --brand-600:#4338CA; --brand-700:#3730A3;
            --ink-900:#1E2433; --ink-600:#4A5168; --ink-400:#7B8299;
            --line:#E7E9F2; --surface:#FFFFFF; --bg:#F5F6FA;
            --amber:#F59E0B; --emerald:#10B981; --rose:#F43F5E;
            --radius:16px; --shadow-sm:0 1px 2px rgba(20,28,55,.05);
            --shadow-md:0 8px 24px rgba(35,44,88,.10);
            --shadow-brand:0 8px 20px rgba(79,70,229,.28);
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        code, pre {
            font-family: "JetBrains Mono","SFMono-Regular",Consolas,
                "Courier New",monospace !important;
        }

        /* ===================== 全局底色与排版 ===================== */
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(900px 320px at 85% -120px, rgba(99,102,241,.10), transparent 70%),
                linear-gradient(180deg,#F7F8FC 0%, var(--bg) 300px) fixed;
            color: var(--ink-900);
        }
        /* padding-top 必须为绝对定位的顶部工具条(stHeader, 约 60px)留出避让空间，
           否则问答页标题行/按钮会被压在工具条下，表现为“页面显示不全” */
        [data-testid="stMainBlockContainer"] { max-width: 1060px; padding-top: 4.5rem; }
        h1,h2,h3,h4 { letter-spacing: -0.01em; color: var(--ink-900); }
        hr, [data-testid="stDivider"] { border-color: var(--line) !important; margin: 1.1rem 0; }
        .stCaptionContainer, [data-testid="stCaptionContainer"] { line-height: 1.55; }

        /* 细滚动条 */
        ::-webkit-scrollbar { width: 9px; height: 9px; }
        ::-webkit-scrollbar-thumb { background: #C9CEDF; border-radius: 99px;
            border: 2px solid transparent; background-clip: content-box; }
        ::-webkit-scrollbar-thumb:hover { background: #AEB5CC; background-clip: content-box; }
        ::-webkit-scrollbar-track { background: transparent; }

        /* 隐藏 Deploy，保留主菜单 */
        [data-testid="stAppDeployButton"], #stDeployButton { display: none !important; }

        /* ===================== 侧边栏 ===================== */
        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        .qa-side-title {
            font-size: 1.12rem; font-weight: 800; color: var(--brand-700);
            display: flex; align-items: center; gap: 8px; margin: 0;
        }
        [data-testid="stSidebar"] h5 {
            font-size: .76rem; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; color: var(--ink-400);
            margin: 1.15rem 0 .55rem;
            padding-left: 10px; border-left: 3px solid var(--brand-500);
        }
        .qa-side-head { display: flex; align-items: center; gap: 11px; margin-bottom: 4px; }
        .qa-side-logo {
            width: 40px; height: 40px; flex: none; display: inline-flex;
            align-items: center; justify-content: center; font-size: 1.22rem;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--brand-500), #818CF8);
            box-shadow: 0 6px 14px rgba(79,70,229,.32);
        }
        .qa-side-sub { margin: 2px 0 0; font-size: .76rem; color: var(--ink-400); letter-spacing: .02em; }
        .qa-cur-pill {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: .78rem; border-radius: 999px; padding: 4px 12px;
            margin: 8px 0 2px; border: 1px solid;
            max-width: 100%; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis;
        }
        .qa-cur-pill.home  { background: #FFFBEB; border-color: #FDE68A; color: #B45309; }
        .qa-cur-pill.cur   { background: var(--brand-50); border-color: #C7D2FE; color: var(--brand-700); }
        .qa-cur-pill b { font-family: "JetBrains Mono", Consolas, monospace; font-size: .76rem; }
        .qa-sess-info {
            margin: -4px 0 10px 10px !important;
            font-size: .72rem !important; color: var(--ink-400) !important;
            letter-spacing: .01em;
        }
        .qa-sess-info code {
            font-size: .7rem; background: var(--brand-50);
            color: var(--brand-600); padding: 1px 6px; border-radius: 5px;
        }
        /* 侧边栏会话条目按钮紧凑化 */
        [data-testid="stSidebar"] [data-testid="stExpander"] .stButton button {
            padding: 6px 10px !important; font-size: .85rem !important; min-height: 0;
        }

        /* ===================== 回答偏好：分段选择器 ===================== */
        /* 侧边栏内没有 st.radio，role=radiogroup 仅属于这两个分段控件，
           因此用 stSidebar + role 稳定锚点收窄作用域（该版本无 stSegmentedControl testid） */
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 4px; flex-wrap: wrap; padding: 4px;
            background: #F1F3F9 !important; border: 1px solid var(--line);
            border-radius: 12px; width: 100%;
        }
        [data-testid="stSidebar"] [role="radio"] {
            flex: 1 1 auto; min-width: 40px; margin: 0 !important;
            padding: 5px 6px !important;
            border-radius: 9px !important; border: none !important;
            transition: background-color .25s ease, color .25s ease,
                        box-shadow .25s ease, transform .15s ease;
        }
        [data-testid="stSidebar"] [role="radio"] p {
            font-size: .8rem !important; font-weight: 600; margin: 0;
            transition: color .25s ease;
        }
        [data-testid="stSidebar"] [role="radio"]:hover {
            background: rgba(79, 70, 229, .10) !important;
        }
        [data-testid="stSidebar"] [role="radio"]:active {
            transform: scale(.96);
        }
        /* 选中态：品牌靛蓝填充 + 轻投影，明确指示当前选择 */
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
            background: var(--brand-500) !important;
            box-shadow: 0 3px 10px rgba(79, 70, 229, .38);
        }
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"]:hover {
            background: var(--brand-500) !important;
        }
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] p {
            color: #fff !important; font-weight: 800 !important;
        }
        /* 偏好读数卡片：切换选项时整块淡入上移，仪表条宽度平滑过渡 */
        .pref-readout {
            display: flex; gap: 10px; align-items: flex-start;
            margin: 4px 0 12px; padding: 10px 12px;
            border: 1px solid var(--line); border-radius: 12px;
            background: #FAFBFE;
        }
        .pref-dot {
            width: 8px; height: 8px; border-radius: 50%; flex: none;
            margin-top: 5px; background: #22C55E;
            animation: pref-dot-pulse 2s ease-in-out infinite;
        }
        .pref-body { flex: 1; min-width: 0; }
        .pref-line { display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap; }
        .pref-line b { font-size: .84rem; color: var(--ink-900); font-weight: 800; }
        .pref-desc { font-size: .72rem; color: var(--ink-400); line-height: 1.5; }
        .pref-meter {
            height: 5px; border-radius: 99px; background: #E7E9F2;
            margin: 7px 0 5px; overflow: hidden;
        }
        .pref-meter i {
            display: block; height: 100%; border-radius: 99px;
            background: linear-gradient(90deg, var(--brand-500), #818CF8);
            transition: width .55s cubic-bezier(.22, .8, .36, 1);
        }
        @keyframes pref-flash-in {
            from { opacity: 0; transform: translateY(5px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        /* 类名随所选选项序号变化，切换时动画自动重启 */
        [class*="pref-flash-"] { animation: pref-flash-in .38s cubic-bezier(.22, .8, .36, 1) both; }
        @keyframes pref-dot-pulse {
            0%, 100% { box-shadow: 0 0 0 3px rgba(34, 197, 94, .16); }
            50%      { box-shadow: 0 0 0 6px rgba(34, 197, 94, .04); }
        }
        @media (max-width: 640px) {
            [data-testid="stSidebar"] [role="radio"] { min-width: 40px; }
            [data-testid="stSidebar"] [role="radio"] p { font-size: .76rem !important; }
        }

        /* ===================== 按钮体系 ===================== */
        .stButton > button, .stDownloadButton > button {
            border-radius: 11px !important;
            font-weight: 600 !important;
            transition: transform .16s ease, box-shadow .16s ease,
                        border-color .16s ease, background .16s ease, color .16s ease;
        }
        .stButton > button p, .stDownloadButton > button p {
            white-space: normal !important; line-height: 1.4;
        }
        /* 主按钮：靛蓝渐变 */
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--brand-500), #6366F1) !important;
            border: none !important; color: #fff !important;
            box-shadow: var(--shadow-brand);
        }
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(79,70,229,.36);
            background: linear-gradient(135deg, var(--brand-600), var(--brand-500)) !important;
        }
        .stButton > button[kind="primary"]:active { transform: translateY(0); }
        /* 次按钮：白底描边 */
        .stButton > button[kind="secondary"],
        .stDownloadButton > button {
            background: var(--surface) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink-600) !important;
            box-shadow: var(--shadow-sm);
        }
        .stButton > button[kind="secondary"]:hover,
        .stDownloadButton > button:hover {
            border-color: var(--brand-500) !important;
            color: var(--brand-600) !important;
            background: var(--brand-50) !important;
            transform: translateY(-1px);
        }
        button:focus-visible { outline: 3px solid rgba(79,70,229,.35); outline-offset: 1px; }

        /* ===================== 顶部横幅 ===================== */
        .qa-banner {
            position: relative; overflow: hidden;
            background: linear-gradient(135deg, var(--brand-700) 0%,
                        var(--brand-500) 52%, #6366F1 100%);
            border-radius: 22px;
            padding: 32px 36px 26px; color: #fff;
            margin-bottom: 20px; box-shadow: var(--shadow-md);
        }
        .qa-banner::before, .qa-banner::after {
            content: ""; position: absolute; border-radius: 50%;
            filter: blur(6px); pointer-events: none;
        }
        .qa-banner::before {
            width: 280px; height: 280px; right: -80px; top: -130px;
            background: radial-gradient(circle, rgba(255,255,255,.22), transparent 70%);
        }
        .qa-banner::after {
            width: 220px; height: 220px; right: 130px; bottom: -150px;
            background: radial-gradient(circle, rgba(245,158,11,.20), transparent 70%);
        }
        .qa-overline {
            display: inline-block; font-size: .74rem; font-weight: 700;
            letter-spacing: .14em; text-transform: uppercase;
            background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.22);
            padding: 4px 12px; border-radius: 999px; margin-bottom: 12px;
        }
        .qa-banner h1 { margin: 0; font-size: 1.92rem; font-weight: 800; position: relative; }
        .qa-banner p  { margin: 10px 0 0; font-size: .98rem; line-height: 1.7;
                        opacity: .92; max-width: 640px; position: relative; }
        .qa-banner-foot {
            display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
            margin-top: 18px; position: relative;
        }
        .qa-banner-chips { display: inline-flex; flex-wrap: wrap; gap: 8px; }
        .qa-banner-chips span {
            font-size: .78rem; padding: 4px 12px; border-radius: 999px;
            background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.18);
        }
        .qa-status {
            display: inline-flex; align-items: center; gap: 7px;
            background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.22);
            border-radius: 999px; padding: 4px 14px; font-size: .82rem;
        }
        .qa-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .dot-ok    { background: #34D399; box-shadow: 0 0 8px rgba(52,211,153,.9); }
        .dot-warn  { background: var(--amber); box-shadow: 0 0 8px rgba(245,158,11,.9); }
        .dot-error { background: #FB7185; box-shadow: 0 0 8px rgba(251,113,133,.9); }
        .dot-off   { background: #C7D2FE; }

        /* ===================== 指标卡 ===================== */
        [data-testid="stMetric"] {
            background: var(--surface); border: 1px solid var(--line);
            border-radius: var(--radius); padding: 16px 20px;
            box-shadow: var(--shadow-sm);
            transition: transform .18s ease, box-shadow .18s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px); box-shadow: var(--shadow-md);
        }
        [data-testid="stMetric"] label {
            font-size: .8rem !important; color: var(--ink-400) !important;
            font-weight: 600; letter-spacing: .02em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.55rem !important; font-weight: 800 !important;
            color: var(--brand-700) !important;
        }

        /* ===================== 提示条 / 卡片 ===================== */
        [data-testid="stAlert"] {
            border-radius: 13px;
            padding: 13px 16px; box-shadow: var(--shadow-sm);
            border: 1px solid var(--line); border-left: 4px solid var(--ink-400);
        }
        [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
            background: #ECFDF5 !important; border-color: #A7F3D0;
            border-left-color: #10B981; color: #065F46;
        }
        [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
            background: var(--brand-50) !important; border-color: #C7D2FE;
            border-left-color: var(--brand-500); color: var(--brand-700);
        }
        [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
            background: #FFFBEB !important; border-color: #FDE68A;
            border-left-color: var(--amber); color: #92400E;
        }
        [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
            background: #FFF1F2 !important; border-color: #FECDD3;
            border-left-color: #FB7185; color: #9F1239;
        }
        /* 内层容器/内容区透明，避免与外层定制底色形成“套色” */
        [data-testid="stAlert"] [data-testid="stAlertContainer"],
        [data-testid="stAlert"] [data-testid^="stAlertContent"] {
            background: transparent !important;
        }
        [data-testid="stAlert"] p { line-height: 1.7; }
        .qa-welcome {
            border: 1px solid var(--line); border-radius: 18px;
            padding: 24px 28px; background: var(--surface);
            margin-bottom: 16px; box-shadow: var(--shadow-sm);
            position: relative;
        }
        .qa-welcome::before {
            content: ""; position: absolute; left: 0; top: 18px; bottom: 18px;
            width: 4px; border-radius: 0 4px 4px 0;
            background: linear-gradient(180deg, var(--brand-500), #818CF8);
        }
        .qa-welcome h3 { margin: 0 0 8px; color: var(--brand-700); font-weight: 800; }
        .qa-welcome p  { margin: 0; color: var(--ink-600); font-size: .94rem; line-height: 1.75; }

        /* 折叠面板（会话列表） */
        [data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            background: #FAFBFE !important;
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stExpander"] details summary {
            font-weight: 700; color: var(--ink-600);
        }
        [data-testid="stExpander"] summary:hover { color: var(--brand-600); }

        /* ===================== 会话状态元信息条 ===================== */
        .qa-session-meta {
            display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
            margin: 4px 0 2px; font-size: .82rem; color: var(--ink-600);
        }
        .qa-meta-pill {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--surface); border: 1px solid var(--line);
            border-radius: 999px; padding: 3px 12px; box-shadow: var(--shadow-sm);
        }
        .qa-meta-pill b { color: var(--brand-600); font-weight: 700;
            font-family: "JetBrains Mono", Consolas, monospace; font-size: .8rem; }
        .qa-meta-status.ok    { background: #ECFDF5; border-color: #A7F3D0; color: #047857; }
        .qa-meta-status.run   { background: var(--brand-50); border-color: #C7D2FE; color: var(--brand-700); }
        .qa-meta-status.empty { background: #FFFBEB; border-color: #FDE68A; color: #B45309; }

        /* ===================== 聊天区域 ===================== */
        [data-testid="stChatMessage"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px 18px 12px 14px;
            margin-bottom: 14px;
            box-shadow: var(--shadow-sm);
            animation: qa-fade-up .34s cubic-bezier(.22,.8,.36,1) both;
        }
        [data-testid="stChatMessage"]:hover { box-shadow: var(--shadow-md); }
        /* 学生气泡：淡靛蓝底 + 靛蓝描边（头像 testid 为 stChatMessageAvatarUser） */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--brand-50);
            border-color: #D7DBFF;
        }
        [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
            border-radius: 10px; box-shadow: inset 0 0 0 1px var(--line);
        }
        /* Streamlit 默认用户头像写死 #FF4B4B，覆盖为品牌靛蓝渐变 */
        [data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(135deg, var(--brand-500), #6366F1) !important;
            box-shadow: 0 3px 8px rgba(79,70,229,.35);
        }
        [data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, #0F172A, #475569) !important;
            box-shadow: 0 3px 8px rgba(15,23,42,.30);
        }
        [data-testid="stChatMessage"] p { line-height: 1.78; }
        [data-testid="stChatMessage"] pre {
            border-radius: 12px; border: 1px solid #1E293B;
        }
        @keyframes qa-fade-up {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* 聊天输入框 */
        [data-testid="stChatInput"] {
            border: 1.5px solid var(--line) !important;
            border-radius: 18px !important;
            background: var(--surface) !important;
            box-shadow: var(--shadow-sm);
            transition: border-color .18s ease, box-shadow .18s ease;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--brand-500) !important;
            box-shadow: 0 0 0 4px rgba(79,70,229,.14);
        }
        /* 内层 textarea 包裹层默认取次级底色（灰），改为白色卡片 */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] textarea { background: #fff !important; }
        [data-testid="stChatInput"] textarea { font-size: .95rem !important; }
        [data-testid="stChatInputSendButton"] button,
        [data-testid="stChatInput"] button[kind="primary"] {
            box-shadow: none !important; border-radius: 12px !important;
        }

        /* 滑杆配色跟主题 */
        [data-baseweb="slider"] [role="slider"] {
            background: var(--brand-500) !important;
            border-color: var(--brand-500) !important;
        }

        /* ===================== 小屏适配 ===================== */
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] { padding-left: .7rem; padding-right: .7rem; }
            .qa-banner { padding: 22px 18px; border-radius: 18px; }
            .qa-banner h1 { font-size: 1.42rem; }
            .qa-banner p { font-size: .88rem; }
            .qa-welcome { padding: 18px; }
            /* 主区域多列布局在手机上纵向堆叠（侧边栏多列保持横向） */
            section.stMain [data-testid="stHorizontalBlock"],
            [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
                flex-direction: column; gap: .6rem;
            }
            section.stMain [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
            [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                width: 100% !important;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation: none !important; transition: none !important;
            }
        }

        /* ===================== 课程体系模块 ===================== */
        /* 专业卡片区 */
        .major-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 14px; margin: 4px 0 20px;
        }
        .major-card {
            position: relative; border: 1px solid var(--line); border-radius: 18px;
            padding: 18px 20px; background: var(--surface);
            box-shadow: var(--shadow-sm); overflow: hidden;
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        }
        .major-card.active {
            border-color: var(--brand-500);
            box-shadow: 0 10px 26px rgba(79,70,229,.18);
            background: linear-gradient(160deg, #FFFFFF 0%, var(--brand-50) 130%);
        }
        .major-card.active::before {
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
            background: linear-gradient(180deg, var(--brand-500), #818CF8);
        }
        .major-card .mc-icon { font-size: 1.7rem; line-height: 1; }
        .major-card .mc-name { font-size: 1.06rem; font-weight: 800; color: var(--ink-900); margin: 8px 0 2px; }
        .major-card .mc-en { font-size: .72rem; color: var(--ink-400); letter-spacing: .04em; text-transform: uppercase; }
        .major-card .mc-tag { font-size: .8rem; color: var(--ink-600); line-height: 1.6; margin-top: 8px; }
        .major-card .mc-badge {
            display: inline-block; margin-top: 10px; font-size: .7rem; font-weight: 700;
            color: var(--brand-700); background: var(--brand-50);
            border: 1px solid #C7D2FE; border-radius: 999px; padding: 2px 10px;
        }
        .major-card.locked { opacity: .68; background: #FAFBFE; }
        .major-card.locked .mc-badge { color: var(--ink-400); background: #F1F3F9; border-color: var(--line); }

        /* 类别分段导航 pill */
        .cat-bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 16px; }
        .cat-pill {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: .84rem; font-weight: 600; border-radius: 999px;
            padding: 6px 14px; border: 1px solid var(--line); background: var(--surface);
            color: var(--ink-600); cursor: pointer; user-select: none;
            transition: all .18s ease;
        }
        .cat-pill:hover { transform: translateY(-1px); }
        .cat-pill .cnt {
            font-size: .7rem; font-weight: 700; border-radius: 999px;
            padding: 0 7px; line-height: 17px; background: #EEF1F8; color: var(--ink-400);
        }

        /* 课程列表卡片 */
        .course-card {
            border: 1px solid var(--line); border-left-width: 4px;
            border-radius: 14px; background: var(--surface);
            padding: 13px 15px; margin-bottom: 10px;
            box-shadow: var(--shadow-sm); cursor: pointer;
            transition: transform .18s ease, box-shadow .18s ease;
            animation: qa-fade-up .3s cubic-bezier(.22,.8,.36,1) both;
        }
        .course-card:hover { transform: translateX(3px); box-shadow: var(--shadow-md); }
        .course-card.active { box-shadow: 0 8px 22px rgba(79,70,229,.16); }
        .course-card .cc-top { display: flex; align-items: center; gap: 8px; }
        .course-card .cc-name { font-size: .97rem; font-weight: 700; color: var(--ink-900); }
        .course-card .cc-nature {
            font-size: .66rem; font-weight: 700; border-radius: 6px; padding: 1px 7px;
            margin-left: auto; white-space: nowrap;
        }
        .nature-required { background: #FEE2E2; color: #B91C1C; }
        .nature-elective { background: #DBEAFE; color: #1D4ED8; }
        .course-card .cc-meta {
            display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 7px;
            font-size: .76rem; color: var(--ink-400);
        }
        .course-card .cc-meta b { color: var(--ink-600); font-weight: 600; }

        /* 课程列表：整卡即按钮（st-key-coursecard_ 前缀由 widget key 自动生成） */
        [class*="st-key-coursecard_"] { margin-bottom: 2px; }
        [class*="st-key-coursecard_"] > button {
            text-align: left !important; justify-content: flex-start !important;
            align-items: flex-start !important; min-height: 0 !important;
            padding: 10px 14px !important; border-left-width: 4px !important;
            animation: qa-fade-up .3s cubic-bezier(.22,.8,.36,1) both;
        }
        [class*="st-key-coursecard_"] > button p {
            white-space: pre-line !important; text-align: left;
            font-size: .75rem !important; font-weight: 500;
            color: var(--ink-400) !important; line-height: 1.55; margin: 0;
        }
        [class*="st-key-coursecard_"] > button p::first-line {
            font-size: .95rem !important; font-weight: 800 !important;
            color: var(--ink-900) !important; letter-spacing: -0.005em;
        }
        [class*="st-key-coursecard_"] > button[kind="primary"] p,
        [class*="st-key-coursecard_"] > button[kind="primary"] p::first-line {
            color: #fff !important;
        }
        [class*="st-key-coursecard_"] > button[kind="primary"] p { opacity: .85; }
        [class*="st-key-coursecard_"] > button[kind="primary"] { border-left-color: #fff !important; }

        /* 类别筛选按钮的圆角化与高度收敛 */
        [class*="st-key-catpill_"] > button {
            padding: 6px 8px !important; font-size: .82rem !important; min-height: 0;
        }

        /* 课程列表面板内的分类小标题（固定高度滚动容器中保持视觉分组） */
        .course-group-label {
            display: flex; align-items: center; gap: 8px;
            font-size: .74rem; font-weight: 800; letter-spacing: .04em;
            color: var(--ink-400); margin: 10px 2px 6px;
        }
        .course-group-label::after {
            content: ""; flex: 1; height: 1px; background: var(--line);
        }
        .course-group-label:first-child { margin-top: 2px; }
        .course-search-hint {
            font-size: .76rem; color: var(--ink-400); margin: 0 0 8px;
        }

        /* 课程详情 */
        .detail-head {
            border: 1px solid var(--line); border-radius: 18px;
            background: var(--surface); padding: 20px 22px;
            box-shadow: var(--shadow-sm); margin-bottom: 14px;
            animation: qa-fade-up .34s cubic-bezier(.22,.8,.36,1) both;
        }
        .detail-cat {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: .74rem; font-weight: 700; border-radius: 999px;
            padding: 3px 12px; margin-bottom: 10px;
        }
        .detail-head h2 { margin: 0; font-size: 1.5rem; font-weight: 800; color: var(--ink-900); }
        .detail-head .dh-en { font-size: .82rem; color: var(--ink-400); margin-top: 3px; letter-spacing: .02em; }
        .detail-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
        .detail-badge {
            display: inline-flex; flex-direction: column; gap: 2px;
            min-width: 78px; border: 1px solid var(--line); border-radius: 12px;
            padding: 8px 12px; background: #FAFBFE;
        }
        .detail-badge .db-k { font-size: .66rem; color: var(--ink-400); font-weight: 600; }
        .detail-badge .db-v { font-size: .88rem; color: var(--ink-900); font-weight: 700; }
        .detail-section {
            border: 1px solid var(--line); border-radius: 16px; background: var(--surface);
            padding: 16px 20px; margin-bottom: 12px; box-shadow: var(--shadow-sm);
            animation: qa-fade-up .34s cubic-bezier(.22,.8,.36,1) both;
        }
        .detail-section h4 {
            margin: 0 0 10px; font-size: .95rem; font-weight: 800;
            color: var(--brand-700); display: flex; align-items: center; gap: 7px;
        }
        .detail-section p { margin: 0; color: var(--ink-600); font-size: .9rem; line-height: 1.8; }
        .tag-cloud { display: flex; flex-wrap: wrap; gap: 7px; }
        .tag-chip {
            font-size: .78rem; border-radius: 999px; padding: 3px 12px;
            background: var(--brand-50); color: var(--brand-700);
            border: 1px solid #DDE1FB;
        }
        .prereq-chain { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
        .prereq-node {
            font-size: .8rem; font-weight: 600; border-radius: 9px;
            padding: 4px 12px; background: #F1F5F9; color: var(--ink-600);
            border: 1px solid var(--line);
        }
        .prereq-arrow { color: var(--ink-400); font-size: .8rem; }
        .prereq-self {
            font-size: .8rem; font-weight: 700; border-radius: 9px;
            padding: 4px 12px; background: var(--brand-500); color: #fff;
        }
        .detail-empty {
            border: 1.5px dashed #C9CEDF; border-radius: 18px; background: rgba(255,255,255,.6);
            padding: 52px 24px; text-align: center; color: var(--ink-400);
        }
        .detail-empty .de-icon { font-size: 2.6rem; }
        .detail-empty p { margin: 10px 0 0; font-size: .92rem; line-height: 1.7; }
        .ai-advice-box {
            border: 1px solid #C7D2FE; border-radius: 16px;
            background: linear-gradient(165deg, #FFFFFF 0%, var(--brand-50) 150%);
            padding: 18px 22px; box-shadow: var(--shadow-sm);
        }
        .ai-advice-box h3 { margin: 0 0 6px; font-size: 1.02rem; color: var(--brand-700); font-weight: 800; }
        .ai-advice-box h3 + div > div h4, .ai-advice-box h4 { font-size: .92rem; }

        /* 右侧详情面板：与左侧课程列表面板同规格（边框 + 600px 固定高 + 内部滚动） */
        [data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-course_detail_panel) { padding: 0; }
        /* 左栏面板上方有搜索框，右栏没有：flex 底对齐使两个等高面板上下沿精确对齐；
           窄屏上下堆叠时列高随内容收缩，auto 外边距不产生空白 */
        [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > div:has(.st-key-course_detail_panel) {
            margin-top: auto;
        }
        .st-key-course_detail_panel { padding: 20px 22px 24px; }
        /* 面板内头部扁平化为“面板标题区”，避免边框卡片嵌套 */
        .st-key-course_detail_panel .detail-head {
            border: none !important; border-radius: 0; box-shadow: none;
            background: transparent; padding: 0 0 14px; margin: 0 0 14px;
            border-bottom: 1px solid var(--line);
        }
        /* 信息小节在面板内用淡底卡片、去除投影，视觉更轻 */
        .st-key-course_detail_panel .detail-section { box-shadow: none; background: #FAFBFE; }
        .st-key-course_detail_panel .ai-advice-box { box-shadow: none; }
        /* 空态在固定高度面板内垂直居中 */
        .st-key-course_detail_panel .detail-empty {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            min-height: 452px; margin: 0; background: transparent; border-color: #D7DBEA;
        }

        @media (max-width: 640px) {
            .major-grid { grid-template-columns: 1fr; }
            .detail-badge { min-width: 68px; flex: 1 1 30%; }
            .detail-head { padding: 16px; }
            .detail-section, .ai-advice-box { padding: 14px 16px; }
            /* 窄屏下面板高度收敛为视口比例，左右两栏上下堆叠时不过分占屏 */
            [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-course_list_panel),
            [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-course_detail_panel) {
                height: 70vh !important; max-height: 560px !important; min-height: 360px !important;
            }
            .st-key-course_detail_panel { padding: 16px; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# 会话存储层：每个会话拥有独立 ID 与独立消息空间，JSON 文件持久化
# --------------------------------------------------------------------------- #
STORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions_store.json")
NEW_SESSION_TITLE = "🆕 新会话"


def load_sessions() -> Dict[str, dict]:
    """从磁盘读取全部历史会话；文件不存在或损坏时安全降级为空集合。"""
    try:
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, ValueError, OSError):
        pass
    return {}


def save_sessions() -> None:
    """原子写入全部会话（先写临时文件再替换），避免落盘中途损坏历史数据。"""
    tmp_file = STORE_FILE + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.sessions, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, STORE_FILE)
    except OSError:
        # 持久化失败不应阻断答疑流程，会话在当前浏览器会话内仍然可用
        pass


# --------------------------------------------------------------------------- #
# 会话状态初始化
# --------------------------------------------------------------------------- #
def init_state() -> None:
    # 启动自愈：空会话没有保留价值，从磁盘恢复时直接过滤（也顺带清理历史脏数据）
    persisted_sessions = {
        sid: sess
        for sid, sess in load_sessions().items()
        if sess.get("messages")
    }
    defaults = {
        # ---- 多会话存储 ----
        "sessions": persisted_sessions,   # {sid: {id,title,created_at,updated_at,messages}}
        "current_session_id": None,    # 当前会话 ID；None 表示处于主页"新会话发起模式"
        "conn_state": "unknown",       # unknown / ok / error
        "conn_message": "",
        "temperature": 0.5,
        "max_tokens": 2048,
        "reply_style": "专业",       # 用户友好的回复风格（映射为 temperature + 系统指令）
        "reply_length": "详细",      # 用户友好的回复长度（映射为 max_tokens + 系统指令）
        "last_latency": None,         # 最近一次回答耗时（秒）
        # ---- 双视图路由相关 ----
        "current_view": "home",       # home=主页 / chat=独立问答页 / courses=课程体系
        "need_answer": False,         # 问答页是否有待流式生成的回答
        "last_error": None,           # 最近一次生成失败的提示（跨 rerun 保留）
        # 本次重绘后主区域的一次性滚动定位："top"=回顶 / "bottom"=定位最新 / None=不干预
        "scroll_action": None,
        # ---- 课程体系模块 ----
        "course_major_id": "SE",   # 当前选中的专业
        "course_category": "全部",  # 当前类别筛选（全部=显示五个类别）
        "course_search": "",       # 课程列表搜索关键词
        "course_selected": {},     # {专业id: 选中的课程code}，跨专业记忆选择
        "course_advice": {},       # {专业id:课程code: 建议文本}，本次会话内缓存
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 内存态自愈：F5 刷新会沿用既有 session_state（不会重新走磁盘加载分支），
    # 这里清理历史版本可能遗留的"非当前空会话"；用户刚创建尚未提问的当前空会话保留。
    current_id = st.session_state.get("current_session_id")
    st.session_state.sessions = {
        sid: sess
        for sid, sess in st.session_state.sessions.items()
        if sess.get("messages") or sid == current_id
    }

init_state()

# --------------------------------------------------------------------------- #
# 会话 CRUD：所有消息读写都必须显式绑定会话 ID，杜绝跨会话串写
# --------------------------------------------------------------------------- #
def current_session() -> Optional[dict]:
    """获取当前会话对象；未绑定任何会话时返回 None。"""
    sid = st.session_state.current_session_id
    if sid is None:
        return None
    return st.session_state.sessions.get(sid)


def sorted_sessions() -> List[dict]:
    """按最近更新时间倒序返回全部会话（最新在前）。"""
    return sorted(
        st.session_state.sessions.values(),
        key=lambda s: s.get("updated_at", 0.0),
        reverse=True,
    )


def create_new_session() -> str:
    """创建一个拥有独立 ID 与独立消息空间的全新会话，并切换为当前会话。"""
    # 新建前清理所有空会话——包括"当前空会话"本身（它即将被新会话取代），
    # 否则连续点击新建对话会残留多个空会话
    cur = current_session()
    if cur is not None and not cur.get("messages"):
        st.session_state.sessions.pop(cur["id"], None)
    prune_empty_sessions()
    now = time.time()
    sid = uuid.uuid4().hex
    st.session_state.sessions[sid] = {
        "id": sid,
        "title": NEW_SESSION_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    st.session_state.current_session_id = sid
    st.session_state.need_answer = False
    st.session_state.last_error = None
    st.session_state.scroll_action = "top"
    save_sessions()
    return sid


def switch_session(sid: str) -> None:
    """切换到指定历史会话：仅更换绑定的会话 ID，绝不改动其他会话的数据。"""
    if sid not in st.session_state.sessions:
        return
    prune_empty_sessions(keep_id=sid)
    st.session_state.current_session_id = sid
    st.session_state.need_answer = False
    st.session_state.last_error = None
    st.session_state.current_view = "chat"
    # bottom 模式自带短内容判定：短会话回顶展示完整标题，长会话落到最新消息
    st.session_state.scroll_action = "bottom"


def delete_session(sid: str) -> None:
    """删除指定会话；若删除的是当前会话，则回到主页的新会话发起模式。"""
    st.session_state.sessions.pop(sid, None)
    if st.session_state.current_session_id == sid:
        st.session_state.current_session_id = None
        st.session_state.need_answer = False
        st.session_state.last_error = None
        st.session_state.current_view = "home"
        st.session_state.scroll_action = "top"
    save_sessions()


def prune_empty_sessions(keep_id: Optional[str] = None) -> None:
    """清理没有任何消息的空会话（保留 keep_id 与当前会话），避免无效占位。"""
    empty_ids = [
        sid
        for sid, sess in st.session_state.sessions.items()
        if not sess.get("messages") and sid not in (keep_id, st.session_state.current_session_id)
    ]
    for sid in empty_ids:
        st.session_state.sessions.pop(sid, None)
    if empty_ids:
        save_sessions()


def format_ts(ts: float) -> str:
    """时间戳格式化为 MM-DD HH:MM。"""
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def short_sid(sid: Optional[str]) -> str:
    """会话短 ID，用于界面展示。"""
    return sid[:8] if sid else "--------"


def get_client() -> DeepSeekClient:
    """构造 DeepSeek 客户端（API Key 从项目根目录 .env / 环境变量读取，不在页面暴露）。"""
    return DeepSeekClient(model=DEFAULT_MODEL)


def build_api_messages() -> List[Dict[str, str]]:
    """组装发给 DeepSeek 的完整上下文：系统提示词 + 当前会话内的多轮历史。"""
    session = current_session()
    history = []
    if session is not None:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in session["messages"]
        ]
    # 把用户在侧边栏选择的“回复风格 / 内容长度”作为偏好指令追加到系统提示词，
    # 与基础角色规则分离，保证预设可独立调整而不影响助教身份设定。
    style_meta = REPLY_STYLES.get(st.session_state.get("reply_style", "专业"), REPLY_STYLES["专业"])
    length_meta = REPLY_LENGTHS.get(st.session_state.get("reply_length", "详细"), REPLY_LENGTHS["详细"])
    preference = (
        "\n\n【本次回答偏好（学生在界面中主动选择，请严格遵守）】\n"
        f"· 表达风格：{style_meta['instruction']}\n"
        f"· {length_meta['instruction']}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT + preference}] + history


def prefill_question(text: str) -> None:
    """把内容回填到聊天输入框（需在 chat_input 组件实例化前写入其 key）。"""
    st.session_state["question_input"] = text


# --------------------------------------------------------------------------- #
# 滚动控制：父窗口单例控制器
# --------------------------------------------------------------------------- #
# 这段 JS 运行在 Streamlit 顶层窗口（通过 <script> 元素注入到父文档执行），
# 不是 components.html 的临时 iframe——iframe 会在每次 rerun 时销毁，
# 在其中创建的 rAF / setInterval / 闭包会一同死亡（锁模式卡死、纠偏失效的根因）。
# 顶层窗口在整个会话期间存活，rerun 只重绘 DOM，控制器与监听器不受影响。
_SCROLL_CONTROLLER_JS = """
(function () {
    if (window.__qaScroll) return;
    var C = window.__qaScroll = {
        mode: 'idle',
        start: 0, lastChange: 0, userUp: false,
        mainEl: function () {
            return document.querySelector('section.stMain')
                || document.querySelector('section[data-testid="stMain"]')
                || document.querySelector('section.main');
        },
        nearBottom: function (el) {
            return el.scrollHeight - el.scrollTop - el.clientHeight < 90;
        },
        arm: function (m) {
            C.mode = m;
            C.start = Date.now();
            C.lastChange = Date.now();
            C.userUp = false;
            C.pendingSig = null;
            C.pendingAt = 0;
            var el = C.mainEl();
            if (!el) return;
            if (m === 'top') el.scrollTo({ top: 0 });
            else if (m === 'bottom') el.scrollTo({ top: el.scrollHeight });
            else if (m === 'follow' && el.scrollHeight > el.clientHeight)
                el.scrollTo({ top: el.scrollHeight });
        }
    };

    /* 用户意图监听：挂在顶层 document 捕获阶段，rerun 重建 DOM 不失效。 */
    document.addEventListener('wheel', function (e) {
        if (C.mode === 'top' || C.mode === 'bottom') C.mode = 'idle';
        else if (C.mode === 'follow' && e.deltaY < 0) C.userUp = true;
    }, { capture: true, passive: true });
    var touchY = null;
    document.addEventListener('touchstart', function (e) {
        touchY = e.touches[0].clientY;
    }, { capture: true, passive: true });
    document.addEventListener('touchmove', function (e) {
        if (touchY === null) return;
        var dy = e.touches[0].clientY - touchY;
        touchY = e.touches[0].clientY;
        if (C.mode === 'top' || C.mode === 'bottom') C.mode = 'idle';
        else if (C.mode === 'follow' && dy > 8) C.userUp = true;
    }, { capture: true, passive: true });
    document.addEventListener('keydown', function (e) {
        if (C.mode === 'top' || C.mode === 'bottom') C.mode = 'idle';
        else if (C.mode === 'follow'
                && (e.key === 'PageUp' || e.key === 'Home' || e.key === 'ArrowUp'))
            C.userUp = true;
    }, true);
    document.addEventListener('scroll', function (e) {
        var t = e.target;
        if (C.mode === 'follow' && t && t === C.mainEl() && C.nearBottom(t))
            C.userUp = false;  /* 滚回底部：恢复跟随 */
    }, { capture: true, passive: true });

    /* 唯一的 DOM 变动观察者：为 top/bottom 提供平息计时 */
    if (window.MutationObserver) {
        new MutationObserver(function () { C.lastChange = Date.now(); })
            .observe(document.documentElement,
                     { childList: true, subtree: true, characterData: true });
    }

    function tickBody() {
        var el = C.mainEl();
        if (!el) return;
        if (C.mode === 'top' || C.mode === 'bottom') {
            var now = Date.now();
            var age = now - C.start;

            function finalize() {
                /* 终判只看页面整体尺寸（不依赖消息元素是否逐个就位）：
                   仅微溢出（≤160px）的短会话停在顶部，长会话落到最新消息。
                   内容不超高时 scrollTo(scrollHeight) 浏览器自然钳制为 0。 */
                var finalWant = 0;
                if (C.mode === 'bottom'
                        && el.scrollHeight - el.clientHeight > 160)
                    finalWant = el.scrollHeight;
                el.scrollTo({ top: finalWant });
                C.mode = 'idle';
            }

            if (age >= 4000) { finalize(); return; }  /* 硬上限兜底 */

            /* 平息需“两次确认”：DOM 连续安静 800ms 形成候选，再观察
               350ms，页面高度与消息数都不变才终判——Streamlit 分块渲染
               存在“假安静窗口”（消息尚未挂全就短暂静止），单次确认会
               误判，造成点开会话后又一次回顶跳动。 */
            if (age >= 800 && now - C.lastChange >= 800) {
                var sig = el.scrollHeight + ':'
                    + el.querySelectorAll('[data-testid="stChatMessage"]').length;
                if (sig !== C.pendingSig) {
                    C.pendingSig = sig;
                    C.pendingAt = now;
                } else if (now - C.pendingAt >= 350) {
                    finalize();
                    return;
                }
            }
            /* 挂载/增长期目标恒定：回顶恒 0；bottom 恒贴底。
               绝不逐帧做“短内容回顶”判定（消息分批挂载、停靠栏高度
               从 0 长起，逐帧判定会让目标反复翻转＝疯狂回顶根因）。 */
            var want = C.mode === 'top' ? 0 : el.scrollHeight;
            if (Math.abs(el.scrollTop - want) > 1)
                el.scrollTo({ top: want });
        } else if (C.mode === 'follow' && !C.userUp
                   && el.scrollHeight > el.clientHeight) {
            if (el.scrollHeight - el.scrollTop - el.clientHeight > 2)
                el.scrollTo({ top: el.scrollHeight });
        }
    }
    (function frame() {
        tickBody();
        requestAnimationFrame(frame);
    })();
    /* 后台标签页 rAF 会暂停，低频兜底（独立 interval，不派生 rAF） */
    setInterval(tickBody, 250);
})();
"""

# “回到主页 / 课程体系”点击瞬间立即进入回顶锁，消除 rerun 等待期错位。
# 监听器幂等，只挂一次（顶层 document，rerun 不影响）。
_BACK_HOOK_JS = """
(function () {
    if (document.__backToTopHooked) return;
    document.__backToTopHooked = true;
    document.addEventListener('click', function (e) {
        if (!e.target || !e.target.closest) return;
        var hit = e.target.closest(
            '[class*="courses_back_home"],[class*="st-key-back_home"],'
            + '[class*="st-key-nav_courses"]');
        if (!hit) return;
        if (window.__qaScroll) window.__qaScroll.arm('top');
    }, true);
})();
"""


def _inject_parent_js(parent_js: str) -> None:
    """把 JS 作为 <script> 元素注入到 Streamlit 顶层文档执行。

    components.html 的 iframe 与应用同源，可以访问父文档；脚本元素追加后
    在父窗口全局作用域同步执行并随即移除，代码本体从此属于顶层窗口。
    json.dumps 保证任意内容都被安全编码为合法的 JS 字符串字面量。
    """
    payload = json.dumps(parent_js, ensure_ascii=False)
    components_html(
        "<script>(function () {"
        "var W=window.parent, doc=W.document;"
        "var s=doc.createElement('script');"
        f"s.textContent={payload};"
        "doc.head.appendChild(s); s.parentNode.removeChild(s);"
        "})();</script>",
        height=0,
    )


def scroll_main(mode: str = "bottom") -> None:
    """
    通过顶层窗口单例滚动控制器（window.__qaScroll）接管主区域滚动。

    历史上每 0.25s 注入一个独立“滚动锁”，多锁并存且各自的目标位置
    在页面增长过程中会在 0 / 底部之间翻转，是提问时界面乱跳、
    回看历史被强行拽回底部的根因。控制器全局唯一：

    - idle   ：完全不干预
    - top    ：视图切换回顶，DOM 平息检测 800ms（硬上限 4s），用户操作即放行
    - bottom ：一次性定位（切换历史会话 / 流式结束重绘后），内容静止时
               目标值恒定，平息后自动转 idle；短会话保持顶部不顶走标题
    - follow ：流式跟随，内容增长时持续贴底；用户一旦上滚即停止跟随，
               滚回底部附近自动恢复，绝不与用户抢滚动条
    """
    _inject_parent_js(_SCROLL_CONTROLLER_JS + f"\nwindow.__qaScroll.arm('{mode}');\n")


def inject_instant_back_hook() -> None:
    """首屏即自举控制器（空闲态），并安装“回到主页/课程体系”点击即时回顶钩子。

    每次渲染注入都安全：控制器与钩子均幂等。
    """
    _inject_parent_js(_SCROLL_CONTROLLER_JS + _BACK_HOOK_JS)


def consume_scroll_action() -> None:
    """按场景标记执行一次性滚动定位后清除标记；无标记的普通渲染绝不干预滚动条。

    bottom 为一次性定位（切换历史会话、流式结束重绘后落到最新消息）；
    流式生成过程中的持续跟随由 run_answer_stream 的 follow 模式负责。
    """
    action = st.session_state.get("scroll_action")
    if action in ("top", "bottom"):
        scroll_main(action)
        st.session_state.scroll_action = None


def submit_question(prompt: str, *, new_session: bool = False) -> None:
    """
    记录学生提问并标记待生成回答，随后自动跳转到独立问答页。

    :param new_session: True 时先创建全新会话（主页发起），保证消息绝不写入旧会话；
                        False 时写入当前会话（问答页内追问）。
    """
    prompt = prompt.strip()
    if new_session or current_session() is None:
        create_new_session()
    session = current_session()
    # 首条提问自动作为会话标题，便于在历史列表中区分
    if not session["messages"]:
        session["title"] = prompt[:20] + ("…" if len(prompt) > 20 else "")
    session["messages"].append({"role": "user", "content": prompt})
    session["updated_at"] = time.time()
    st.session_state.last_error = None
    st.session_state.need_answer = True
    st.session_state.current_view = "chat"
    # 新提交后进入问答页：一次性定位到最新问答（流式生成期间另有平滑跟随）
    st.session_state.scroll_action = "bottom"
    save_sessions()


def run_answer_stream() -> None:
    """在问答页实时流式渲染大模型回答，完成后写入会话历史并重绘页面。"""
    client = get_client()
    full_answer = ""
    started_at = time.time()

    with st.chat_message("assistant"):
        answer_slot = st.empty()
        answer_slot.markdown("⏳ 正在思考中…")
        # 立即武装“流式跟随”：后续 answer_slot 内容增长由单例控制器
        # 逐帧贴底，无需（也不得）在 chunk 循环里反复注入滚动脚本——
        # 多锁并存正是提问时界面乱跳、回看历史被拽回的根因。
        scroll_main("follow")
        try:
            stream = client.chat_stream(
                build_api_messages(),
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
            for chunk in stream:
                full_answer += chunk
                # 尾部光标模拟逐字输出效果
                answer_slot.markdown(full_answer + " ▌")

            if full_answer:
                answer_slot.markdown(full_answer)
            else:
                answer_slot.markdown("（模型未返回内容，请换一种问法再试一次。）")

            st.session_state.last_latency = time.time() - started_at
            st.caption(f"✅ 回答完成，耗时 {st.session_state.last_latency:.1f} 秒")
            # 连通正常时自动把系统状态置为在线
            st.session_state.conn_state = "ok"
            st.session_state.conn_message = "流式接口正常"

        except DeepSeekConfigError as exc:
            answer_slot.empty()
            st.error(f"🔑 配置错误：{exc}")
            st.session_state.last_error = f"🔑 配置错误：{exc}"
        except DeepSeekAPIError as exc:
            answer_slot.empty()
            st.error(f"☁️ 接口调用失败：{exc}")
            st.session_state.last_error = f"☁️ 接口调用失败：{exc}"
            st.session_state.conn_state = "error"
            st.session_state.conn_message = f"HTTP {exc.status_code}"
        except DeepSeekNetworkError as exc:
            answer_slot.empty()
            st.error(f"🌐 网络异常：{exc}")
            st.session_state.last_error = f"🌐 网络异常：{exc}"
            st.session_state.conn_state = "error"
            st.session_state.conn_message = "网络错误"
        except Exception as exc:  # noqa: BLE001 - 兜底，避免页面直接崩溃
            answer_slot.empty()
            st.error(f"❌ 发生未预期的错误：{exc.__class__.__name__}: {exc}")
            st.session_state.last_error = (
                f"❌ 发生未预期的错误：{exc.__class__.__name__}: {exc}"
            )

    # 保存回答（含中断时的部分内容）到当前会话的独立存储空间，重绘为正式消息
    session = current_session()
    if full_answer and session is not None:
        session["messages"].append({"role": "assistant", "content": full_answer})
        session["updated_at"] = time.time()
        save_sessions()
    st.session_state.need_answer = False
    # 流式结束：解除跟随，随后的重绘由 consume_scroll_action 做一次性落位
    scroll_main("idle")
    st.rerun()


# --------------------------------------------------------------------------- #
# 侧边栏：状态指示 / 参数配置 / 历史记录
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        """
        <div class="qa-side-head">
            <span class="qa-side-logo">🎓</span>
            <span>
                <p class="qa-side-title">智能课程答疑系统v1.1</p>
                <p class="qa-side-sub">Powered by zhh</p>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 0. 新建对话：立即创建独立会话并进入空白问答页 ---- #
    if st.button("➕ 新建对话", key="new_session_top", use_container_width=True, type="primary"):
        create_new_session()
        st.session_state.current_view = "chat"
        st.rerun()

    # 课程体系导航：进入专业→课程→详情的三级展示模块
    if st.button(
        "📚 课程体系(开发版)",
        key="nav_courses",
        use_container_width=True,
        type="primary" if st.session_state.current_view == "courses" else "secondary",
    ):
        st.session_state.current_view = "courses"
        st.session_state.scroll_action = "top"
        st.rerun()

    # 当前会话状态提示（成熟化界面：隐藏，当前会话已在历史列表中高亮标识）
    cur = current_session()
    # if cur is None:
    #     st.markdown(
    #         '<span class="qa-cur-pill home">🆕 新会话发起模式：主页提问将自动创建独立会话</span>',
    #         unsafe_allow_html=True,
    #     )
    # else:
    #     rounds = len(cur["messages"]) // 2
    #     cur_desc = "空会话" if not cur["messages"] else f"{rounds} 轮对话"
    #     st.markdown(
    #         f'<span class="qa-cur-pill cur" title="{cur["title"][:40]}">'
    #         f'📍 当前会话 <b>{short_sid(cur["id"])}</b> · {cur_desc}</span>',
    #         unsafe_allow_html=True,
    #     )

    # ---- 1. 系统状态（API Key 仅通过项目根目录 .env 配置，不在页面展示） ---- #
    # 成熟化界面：隐藏整个系统状态检测区（状态灯/检测按钮），仅保留 client 初始化
    # st.markdown("##### 1. 系统状态")

    client = get_client()

    # 根据配置与检测结果渲染状态灯
    # if not client.configured:
    #     dot_class, state_text = "dot-off", "未读取到 API Key（请检查 .env）"
    # elif st.session_state.conn_state == "ok":
    #     dot_class = "dot-ok"
    #     state_text = f"服务正常 · {st.session_state.conn_message}"
    # elif st.session_state.conn_state == "error":
    #     dot_class = "dot-error"
    #     state_text = f"连接异常 · {st.session_state.conn_message}"
    # else:
    #     dot_class, state_text = "dot-warn", "配置已加载，待检测"

    # st.markdown(
    #     f'<p style="margin:6px 0 10px;">'
    #     f'<span class="qa-dot {dot_class}"></span> '
    #     f'<span style="font-size:0.86rem;color:#334155;">系统状态：{state_text}</span></p>',
    #     unsafe_allow_html=True,
    # )

    # if st.button("🔌 测试 DeepSeek 连接", use_container_width=True):
    #     with st.spinner("正在检测服务连通性…"):
    #         ok, message, latency_ms = client.check_connection()
    #     st.session_state.conn_state = "ok" if ok else "error"
    #     st.session_state.conn_message = (
    #         f"{latency_ms} ms" if ok else message[:60]
    #     )
    #     if ok:
    #         st.toast(f"连接成功，耗时 {latency_ms} ms ✅", icon="✅")
    #     else:
    #         st.toast(f"连接失败：{message}", icon="⚠️")
    #     st.rerun()

    # ---- 2. 回答偏好：技术参数（temperature / max_tokens）已封装为友好选项 ---- #
    st.markdown("##### ✨ 回答偏好")
    style_choice = st.segmented_control(
        "回复风格",
        options=STYLE_ORDER,
        key="reply_style",
        selection_mode="single",
        help="选择 AI 的表达语气，切换后下一次回答立即生效。",
        label_visibility="visible",
    ) or "专业"
    length_choice = st.segmented_control(
        "回复内容长度",
        options=LENGTH_ORDER,
        key="reply_length",
        selection_mode="single",
        help="选择期望的回答篇幅，切换后下一次回答立即生效。",
        label_visibility="visible",
    ) or "详细"

    # 选择结果即时映射回 API 数值参数（run_answer_stream 直接读取这两个 key）
    style_meta = REPLY_STYLES[style_choice]
    length_meta = REPLY_LENGTHS[length_choice]
    st.session_state.temperature = style_meta["temp"]
    st.session_state.max_tokens = length_meta["tokens"]

    # 当前选择读数：选项变化时整块淡入（class 随值变化以重启动画），
    # 仪表条宽度/颜色平滑过渡，给出明确的“当前状态 + 已生效”视觉反馈。
    from html import escape as _esc_style
    creativity_pct = round(style_meta["temp"] / 1.5 * 100)
    st.markdown(
        f"""
        <div class="pref-readout pref-flash-{STYLE_ORDER.index(style_choice)}">
            <span class="pref-dot"></span>
            <div class="pref-body">
                <div class="pref-line">
                    <b>{_esc_style(style_choice)}</b>
                    <span class="pref-desc">{_esc_style(style_meta['desc'])}</span>
                </div>
                <div class="pref-meter"><i style="width:{creativity_pct}%"></i></div>
                <div class="pref-line"><span class="pref-desc">创意度 {creativity_pct}% · 已即时生效</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="pref-readout pref-flash-{LENGTH_ORDER.index(length_choice)}">
            <span class="pref-dot"></span>
            <div class="pref-body">
                <div class="pref-line">
                    <b>{_esc_style(length_choice)}</b>
                    <span class="pref-desc">{_esc_style(length_meta['desc'])}</span>
                </div>
                <div class="pref-meter"><i style="width:{length_meta['tokens'] / 4096 * 100:.0f}%"></i></div>
                <div class="pref-line"><span class="pref-desc">生成上限 {length_meta['tokens']} tokens · 已即时生效</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 3. 历史会话列表（各会话消息独立存储，可查看/切换/删除） ---- #
    st.markdown("##### 历史会话")
    all_sessions = sorted_sessions()
    non_empty_sessions = [s for s in all_sessions if s.get("messages")]
    # st.caption(f"已保存 {len(non_empty_sessions)} 个会话（独立存储、互不影响）")

    if all_sessions:
        with st.expander(f"🗂️ 会话列表（{len(all_sessions)}）", expanded=True):
            for sess in all_sessions:
                sid = sess["id"]
                is_current = sid == st.session_state.current_session_id
                ask_count = sum(1 for m in sess["messages"] if m["role"] == "user")
                mark = "🔵" if is_current else "⚪"
                label = (
                    f"{mark} {sess['title'][:18]}"
                    f"{'…' if len(sess['title']) > 18 else ''}"
                )
                # 成熟化界面：隐藏每条会话的「N 问 · 时间 · ID」详情行
                # info = (
                #     f"{ask_count} 问 · {format_ts(sess['updated_at'])} · "
                #     f"<code>{short_sid(sid)}</code>"
                # )

                btn_col, del_col = st.columns([0.8, 0.2])
                if btn_col.button(
                    label, key=f"sess_{sid}", use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    switch_session(sid)
                    st.rerun()
                if del_col.button("🗑️", key=f"del_{sid}", help="删除该会话"):
                    delete_session(sid)
                    st.rerun()
                # st.caption(
                #     f"<div class='qa-sess-info'>{info}</div>",
                #     unsafe_allow_html=True,
                # )

        # 当前会话的操作：删除 + 导出 Markdown
        if cur is not None and cur.get("messages"):
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ 删除当前会话", use_container_width=True):
                    delete_session(cur["id"])
                    st.rerun()

            transcript_lines = [
                f"# 智能课程答疑记录（会话 {short_sid(cur['id'])}）",
                f"会话标题：{cur['title']}",
                f"创建时间：{datetime.fromtimestamp(cur['created_at']).strftime('%Y-%m-%d %H:%M:%S')}",
                f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
            for m in cur["messages"]:
                speaker = "🧑‍🎓 学生" if m["role"] == "user" else "🤖 答疑助教"
                transcript_lines += [f"### {speaker}", m["content"], ""]
            with col_b:
                st.download_button(
                    "⬇️ 导出当前会话",
                    data="\n".join(transcript_lines),
                    file_name=f"答疑记录_{short_sid(cur['id'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
    else:
        st.info("还没有历史会话，点击上方「➕ 新建对话」或直接在主页提问吧～")

    st.caption(f"""
                本项目由 [Trae CN] 提供支持\n
                基于DeepSeek-V3通用对话模型搭建
                """)


# --------------------------------------------------------------------------- #
# 主区域：视图渲染（主页 home / 独立问答页 chat）
# --------------------------------------------------------------------------- #
def render_banner() -> None:
    """主页顶部横幅 + 实时状态胶囊。"""
    if client.configured and st.session_state.conn_state == "ok":
        banner_dot = '<span class="qa-dot dot-ok"></span> 服务在线'
    elif not client.configured:
        banner_dot = f'<span class="qa-dot dot-off"></span> 未读取到 API Key(请检查 .env 文件)'
    elif st.session_state.conn_state == "error":
        banner_dot = '<span class="qa-dot dot-error"></span> 服务连接异常'
    else:
        banner_dot = '<span class="qa-dot dot-warn"></span> 我已就绪，随时等待提问'

    st.markdown(
        """
        <div class="qa-banner">
            <!-- 成熟化界面：隐藏眉标、产品描述与功能徽章
            <span class="qa-overline">在线课程 · AI 答疑助教</span>
            <p>随时提出课程疑问，AI 课程答疑助手将分步骤讲解原理、给出示例与延伸思考，
               陪你把每个知识点学懂、练会。</p>
            <div class="qa-banner-foot">
                <span class="qa-status">服务状态</span>
                <span class="qa-banner-chips">
                    <span>🧩 多会话隔离</span>
                    <span>⚡ 实时流式答疑</span>
                    <span>💾 历史自动保存</span>
                </span>
            </div>
            -->
            <h1>🎓 基于大模型的智能课程答疑系统</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    """主页：新会话发起模式——此处提问必定创建一个全新隔离会话。"""
    render_banner()

    # 跨全部历史会话的统计（成熟化界面：隐藏指标卡）
    # non_empty = [s for s in st.session_state.sessions.values() if s.get("messages")]
    # total_asks = sum(
    #     1 for s in non_empty for m in s["messages"] if m["role"] == "user"
    # )
    # m1, m2, m3 = st.columns(3)
    # m1.metric("历史会话数", f"{len(non_empty)} 个")
    # m2.metric("历史提问总数", f"{total_asks} 次")
    # m3.metric(
    #     "上次回答耗时",
    #     f"{st.session_state.last_latency:.1f} 秒" if st.session_state.last_latency else "—",
    # )

    # 最近一次有内容的会话：提供继续入口（位于页面上部，避开底部停靠输入栏遮挡）
    recent = next(
        (s for s in sorted_sessions() if s.get("messages")), None
    )
    if recent is not None:
        entry_cols = st.columns([0.68, 0.32])
        # 成熟化界面：隐藏最近会话说明文本，仅保留继续按钮
        entry_cols[0].info(
            f"💬 最近会话「{recent['title'][:6]}...」--- "
            # f"{sum(1 for m in recent['messages'] if m['role'] == 'user')} 个提问 --- "
            f"{format_ts(recent['updated_at'])}，可返回继续追问。"
        )
        if entry_cols[1].button(
            "💬 继续最近会话", key="goto_chat", use_container_width=True, type="primary"
        ):
            switch_session(recent["id"])
            st.rerun()

    # st.divider()

    # 成熟化界面：隐藏「新会话发起模式」说明条
    # st.success(
    #     "🆕 **新会话发起模式**：在下方提交任意问题，系统都会自动创建一个拥有独立 "
    #     "会话 ID 和独立存储空间的**全新会话**，不会延续任何历史会话的问答上下文。"
    #     "历史会话可在左侧列表中随时查看与切换。"
    # )

    # 欢迎卡片 + 推荐问题快捷入口
    st.markdown(
        """
        <div class="qa-welcome">
            <h3>👋 你好，同学！</h3>
            <p>我是你的智能课程答疑助教。无论你对课程有哪方面的问题，都可以直接向我提问哦！
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**⚡ 没有想法的话,试试这些常见问题：**")
    chip_cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        with chip_cols[i % 2]:
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                prefill_question(question)
                st.rerun()

    # 主页底部问题输入框：提交即创建全新会话并跳转问答页
    # 成熟化界面：占位符精简为中性提示（原长说明已注释）
    # "请输入你的课程问题,提交后自动创建全新会话,例如:Python 中 __init__ 和 __new__ 有什么区别？"
    prompt = st.chat_input(
        "输入你的课程问题…",
        key="question_input",
    )
    if prompt and prompt.strip():
        if not client.configured:
            st.error("⚠️ 未读取到有效的 DeepSeek API Key,请在项目根目录的 .env 文件中配置后再提问。")
            st.stop()
        submit_question(prompt, new_session=True)
        st.rerun()


def render_chat() -> None:
    """独立问答页：仅展示并操作当前会话，含会话状态头、流式答疑与导航。"""
    sess = current_session()

    # 顶部导航栏
    title_col, new_col, back_col = st.columns([0.6, 0.2, 0.2])
    if sess is not None:
        ask_count = sum(1 for m in sess["messages"] if m["role"] == "user")
        title_col.markdown(f"#### 💬 {sess['title']}")
    else:
        title_col.markdown("#### 💬 课程问答 · 实时答疑")
    if new_col.button("➕ 新建对话", key="new_session_chat", use_container_width=True):
        create_new_session()
        st.rerun()
    if back_col.button("🏠 回到主页", key="back_home", use_container_width=True, type="primary"):
        st.session_state.current_view = "home"
        st.session_state.need_answer = False
        st.session_state.scroll_action = "top"
        st.rerun()

    # 会话状态信息条：ID / 轮数 / 时间 / 状态，清晰标识当前所处会话
    if sess is None:
        st.warning("未绑定任何会话，请回到主页重新发起提问。")
        if st.button("🏠 回到主页", use_container_width=True):
            st.session_state.current_view = "home"
            st.session_state.scroll_action = "top"
            st.rerun()
        return

    rounds = len(sess["messages"]) // 2
    # 成熟化界面：隐藏会话状态徽章计算与整条元信息（ID/轮数/时间/状态）药丸
    # if not sess["messages"]:
    #     badge_text, badge_cls = "🟡 空会话 · 等待第一个问题", "empty"
    # elif st.session_state.need_answer:
    #     badge_text, badge_cls = "🔵 回答生成中…", "run"
    # else:
    #     badge_text, badge_cls = "🟢 会话进行中", "ok"
    # st.markdown(
    #     f"""
    #     <div class="qa-session-meta">
    #         <span class="qa-meta-pill">会话 ID <b>{short_sid(sess['id'])}</b></span>
    #         <span class="qa-meta-pill">💬 {ask_count} 个提问 · {rounds} 轮对话</span>
    #         <span class="qa-meta-pill">🕐 创建 {format_ts(sess['created_at'])}</span>
    #         <span class="qa-meta-pill">🔄 更新 {format_ts(sess['updated_at'])}</span>
    #         <span class="qa-meta-pill qa-meta-status {badge_cls}">{badge_text}</span>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )
    # st.divider()

    # 空会话引导（成熟化界面：隐藏说明，留白等待首问即可）
    # if not sess["messages"]:
    #     st.info(
    #         "这是一个**刚刚创建的全新会话**，拥有独立的会话 ID 与存储空间，"
    #         "与其他历史会话完全隔离。请在下方输入第一个问题开始答疑～"
    #     )

    # 上次生成失败的错误提示（重绘后仍可见）
    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    # 当前会话的完整历史
    for message in sess["messages"]:
        # 注意：Streamlit 的 chat_message 角色只接受 user / assistant
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 追问输入框：先于流式块渲染，保证生成期间输入框始终可见可用。
    # 在当前会话内追问（new_session=False），消息只写入本会话。
    # 成熟化界面：占位符精简（原"在当前会话（ID）继续追问…"已注释）
    # f"在当前会话（{short_sid(sess['id'])}）继续追问，或输入新问题"
    prompt = st.chat_input(
        f"在当前会话继续追问，或输入新问题",
        key="question_input",
    )
    if prompt and prompt.strip():
        if not client.configured:
            st.error("⚠️ 未读取到有效的 DeepSeek API Key，请在项目根目录的 .env 文件中配置后再提问。")
            st.stop()
        submit_question(prompt)
        st.rerun()

    # 有待生成的回答：实时流式输出，视角平滑跟随
    if st.session_state.need_answer:
        run_answer_stream()


# --------------------------------------------------------------------------- #
# 课程体系视图：专业 → 类别筛选 → 课程列表 → 课程详情（含 AI 学习建议）
# --------------------------------------------------------------------------- #
def _inject_catalog_dynamic_styles() -> None:
    """根据类别元数据生成数据驱动的 pill/卡片配色（避免在静态 CSS 中写死）。"""
    rules = []
    for cat, meta in catalog.CATEGORY_META.items():
        code = meta["code"]
        # 类别筛选按钮（未选中态）：淡色描边 + 类别强调色文字
        rules.append(
            f'[class*="st-key-catpill_{code}"] > button {{'
            f'border-color:{meta["tint_strong"]} !important;'
            f'color:{meta["accent"]} !important;'
            f'background:{meta["tint"]} !important; }}'
        )
        # 课程卡片左侧类别色条
        rules.append(
            f'[class*="st-key-coursecard_"][class*="-{code}-"] > button {{'
            f'border-left:4px solid {meta["accent"]} !important; }}'
        )
    st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)


def render_courses() -> None:
    """三级课程展示：专业选择 → 类别/课程列表 → 课程详情 + AI 学习建议。"""
    from html import escape as _esc

    _inject_catalog_dynamic_styles()

    # ---- 顶部导航 ---- #
    title_col, back_col = st.columns([0.82, 0.18])
    title_col.markdown("#### 📚 专业课介绍")
    if back_col.button("🏠 回到主页", key="courses_back_home", use_container_width=True, type="primary"):
        st.session_state.current_view = "home"
        st.session_state.scroll_action = "top"
        st.rerun()

    # ---- 第一级：专业卡片（可扩展注册表） ---- #
    major_cols = st.columns(len(catalog.MAJORS))
    for col, major in zip(major_cols, catalog.MAJORS):
        is_active = major.available and st.session_state.course_major_id == major.id
        card_cls = "major-card"
        if is_active:
            card_cls += " active"
        if not major.available:
            card_cls += " locked"
        if major.available:
            loaded = catalog.load_major_courses(major)
            badge = f"{len(loaded.courses)} 门课程 · {loaded.total_credits:g} 总学分"
        else:
            badge = "🚧 敬请期待"
        col.markdown(
            f"""
            <div class="{card_cls}">
                <div class="mc-icon">{major.icon}</div>
                <div class="mc-name">{_esc(major.name)}</div>
                <div class="mc-en">{_esc(major.name_en)}</div>
                <div class="mc-tag">{_esc(major.tagline)}</div>
                <span class="mc-badge">{badge}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if major.available:
            if col.button(
                "进入该专业" if not is_active else "✅ 当前专业",
                key=f"major_enter_{major.id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.course_major_id = major.id
                st.session_state.course_category = "全部"
                st.session_state.course_search = ""
                st.session_state.scroll_action = "top"
                st.rerun()
        else:
            col.button("暂未开放", key=f"major_locked_{major.id}", use_container_width=True, disabled=True)

    # ---- 当前专业数据加载（带 Excel 解析缓存） ---- #
    major = catalog.get_major(st.session_state.course_major_id)
    if major is None or not major.available:
        st.info("该专业课介绍正在建设中，敬请期待。")
        return
    major = catalog.load_major_courses(major)

    st.markdown(
        f"<p style='color:var(--ink-400);font-size:.85rem;margin:-6px 0 14px;'>"
        f"{major.icon} <b style='color:var(--ink-600);'>{major.name}</b> · 共 {len(major.courses)} 门课程 · "
        f"建议培养路径：公共基础 → 专业基础 → 专业核心 → 拓展选修 → 实践环节</p>",
        unsafe_allow_html=True,
    )

    # ---- 第二级：类别分段筛选 ---- #
    filters = ["全部"] + catalog.CATEGORY_ORDER
    filter_cols = st.columns(len(filters))
    current_filter = st.session_state.course_category
    for col, cat in zip(filter_cols, filters):
        if cat == "全部":
            icon, count = "🗂️", len(major.courses)
            key = "catpill_ALL"
        else:
            meta = catalog.CATEGORY_META[cat]
            icon, count, key = meta["icon"], len(major.courses_by_category(cat)), f"catpill_{meta['code']}"
        if col.button(
            f"{icon} {cat} {count}",
            key=key,
            use_container_width=True,
            type="primary" if current_filter == cat else "secondary",
        ):
            st.session_state.course_category = cat
            st.rerun()

    filtered = major.courses if current_filter == "全部" else major.courses_by_category(current_filter)
    if current_filter != "全部":
        cat_meta = catalog.CATEGORY_META[current_filter]
        st.markdown(
            f"<p style='font-size:.82rem;color:{cat_meta['accent']};margin:2px 0 12px;font-weight:600;'>"
            f"{cat_meta['icon']} {_esc(cat_meta['desc'])}</p>",
            unsafe_allow_html=True,
        )

    # 选中课程：在「类别筛选结果」中定位；搜索仅收窄左侧列表，不改变右侧详情
    selected_map = st.session_state.course_selected
    selected_code = selected_map.get(major.id)
    course = next((c for c in filtered if c.code == selected_code), None)
    if course is None and filtered:
        course = filtered[0]
        selected_map[major.id] = course.code

    # 关键词搜索（课程名/英文名/编号/知识点/职业方向），仅作用于左侧列表展示
    keyword = st.session_state.course_search.strip().lower()
    if keyword:
        def _match(c) -> bool:
            haystack = " ".join(
                [c.name, c.name_en, c.code, c.career, c.core_content]
            ).lower()
            return keyword in haystack
        displayed = [c for c in filtered if _match(c)]
    else:
        displayed = filtered

    # ---- 第三级：左列表 + 右详情（响应式：窄屏自动上下堆叠） ---- #
    list_col, detail_col = st.columns([0.4, 0.6], gap="medium")

    with list_col:
        st.text_input(
            "搜索课程",
            key="course_search",
            placeholder="🔍 搜索课程名 / 编号 / 关键词，如：数据结构、SE-PC",
            label_visibility="collapsed",
        )
        scope_name = "全部课程" if current_filter == "全部" else current_filter
        st.markdown(
            f"<p class='course-search-hint'>{_esc(scope_name)} · 共 {len(displayed)} 门"
            + ("（搜索结果）" if keyword else "")
            + "</p>",
            unsafe_allow_html=True,
        )
        # 固定高度滚动面板：课程再多也只在面板内滚动，不撑长整个页面
        with st.container(border=True, height=600, key="course_list_panel"):
            if not displayed:
                st.markdown(
                    "<p style='color:var(--ink-400);font-size:.85rem;text-align:center;"
                    "padding:40px 8px;'>😶 没有匹配的课程<br>换个关键词试试</p>",
                    unsafe_allow_html=True,
                )
            # “全部”视图且未搜索时，在滚动面板内按类别插入分组小标题
            show_groups = current_filter == "全部" and not keyword
            last_group = None
            for c in displayed:
                if show_groups and c.category != last_group:
                    last_group = c.category
                    gm = catalog.CATEGORY_META[c.category]
                    group_n = len(major.courses_by_category(c.category))
                    st.markdown(
                        f"<div class='course-group-label'>{gm['icon']} {_esc(c.category)}"
                        f" <span style='font-weight:600;'>{group_n}</span></div>",
                        unsafe_allow_html=True,
                    )
                label = (
                    f"{catalog.CATEGORY_META[c.category]['icon']} {c.name}　【{c.nature}】\n"
                    f"{c.code} · {c.credits}学分 · {c.hours}学时 · {c.difficulty}"
                )
                if st.button(
                    label,
                    key=f"coursecard_{c.code}",
                    use_container_width=True,
                    type="primary" if course is not None and c.code == course.code else "secondary",
                ):
                    selected_map[major.id] = c.code
                    st.rerun()

    with detail_col:
        # 与左侧课程列表同规格：边框 + 600px 固定高度，内容在面板内滚动
        with st.container(border=True, height=600, key="course_detail_panel"):
            if course is None:
                st.markdown(
                    """
                    <div class="detail-empty">
                        <div class="de-icon">📖</div>
                        <p>请在左侧选择一门课程<br>查看课程详情与 AI 学习建议</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                _render_course_detail(major, course)


def _render_course_detail(major, course) -> None:
    """渲染单门课程的完整详情与 AI 学习建议区域。"""
    from html import escape as _esc

    meta = catalog.CATEGORY_META.get(course.category, {})
    accent = meta.get("accent", "#4F46E5")
    tint = meta.get("tint", "#EEF0FF")

    # 详情头部：类别徽章 / 中英文名
    st.markdown(
        f"""
        <div class="detail-head" style="border-top:4px solid {accent};">
            <span class="detail-cat" style="background:{tint};color:{accent};">
                {meta.get('icon', '📘')} {_esc(course.category)}
            </span>
            <h2>{_esc(course.name)}</h2>
            <div class="dh-en">{_esc(course.name_en)} · {_esc(course.code)}</div>
            <div class="detail-badges">
                <span class="detail-badge"><span class="db-k">课程性质</span><span class="db-v">{_esc(course.nature) or '—'}</span></span>
                <span class="detail-badge"><span class="db-k">学分</span><span class="db-v">{_esc(course.credits) or '—'}</span></span>
                <span class="detail-badge"><span class="db-k">学时</span><span class="db-v">{_esc(course.hours) or '—'}</span></span>
                <span class="detail-badge"><span class="db-k">建议学期</span><span class="db-v">{_esc(course.semester) or '—'}</span></span>
                <span class="detail-badge"><span class="db-k">难度</span><span class="db-v">{_esc(course.difficulty) or '—'}</span></span>
                <span class="detail-badge"><span class="db-k">实践占比</span><span class="db-v">{_esc(course.practice_ratio) or '—'}</span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def section(icon: str, title: str, body: str) -> None:
        st.markdown(
            f'<div class="detail-section"><h4>{icon} {_esc(title)}</h4>'
            f'<p>{_esc(body) or "—"}</p></div>',
            unsafe_allow_html=True,
        )

    section("📝", "课程简介", course.intro)
    section("🎓", "能力培养目标", course.objectives)
    section("🧪", "考核方式", course.assessment)
    section("💼", "对应职业方向", course.career)
    if course.remark:
        section("📌", "备注", course.remark)

    # 核心知识点：标签云
    if course.content_points:
        chips = "".join(f'<span class="tag-chip">{_esc(p)}</span>' for p in course.content_points)
        st.markdown(
            f'<div class="detail-section"><h4>🧩 核心学习内容</h4>'
            f'<div class="tag-cloud">{chips}</div></div>',
            unsafe_allow_html=True,
        )

    # 先修路径：先修课 → 本课程（整块一次输出，避免 HTML 标签被 markdown 解析器截断）
    chain_nodes = "".join(
        f'<span class="prereq-node">{_esc(p)}</span><span class="prereq-arrow">→</span>'
        for p in course.prerequisite_list
    )
    if not chain_nodes:
        chain_nodes = '<span class="prereq-node">无硬性先修</span><span class="prereq-arrow">→</span>'
    st.markdown(
        f'<div class="detail-section"><h4>🔗 先修课程路径</h4>'
        f'<div class="prereq-chain">{chain_nodes}'
        f'<span class="prereq-self">{meta.get("icon", "📘")} {_esc(course.name)}</span></div></div>',
        unsafe_allow_html=True,
    )

    # ---- AI 学习建议 ---- #
    advice_key = f"{major.id}:{course.code}"
    advice = st.session_state.course_advice.get(advice_key)
    if advice is None:
        advice = catalog.get_cached_advice(major.id, course.code)

    st.markdown("#### ✨ AI 学习建议")
    if advice:
        with st.container(border=True):
            st.markdown(advice)
            if st.button("🔄 重新生成建议", key=f"advice_refresh_{course.code}"):
                # 清除磁盘缓存与本次会话缓存
                cache = catalog.load_advice_cache()
                cache.pop(advice_key, None)
                catalog.save_advice_cache(cache)
                st.session_state.course_advice.pop(advice_key, None)
                st.rerun()
    else:
        c1, c2 = st.columns(2)
        if c1.button("✨ 获取 AI 个性化建议", key=f"advice_get_{course.code}", type="primary", use_container_width=True):
            if not client.configured:
                st.session_state.course_advice[advice_key] = catalog.fallback_advice(major, course)
                st.info("未配置 DeepSeek API Key，已为你生成基于课程数据的通用建议。")
            else:
                try:
                    with st.spinner("AI 导师正在分析课程定位与学习路径…"):
                        text = catalog.generate_ai_advice(major, course, client)
                    st.session_state.course_advice[advice_key] = text
                except (DeepSeekConfigError, DeepSeekAPIError, DeepSeekNetworkError) as exc:
                    st.session_state.course_advice[advice_key] = catalog.fallback_advice(major, course)
                    st.warning(f"AI 建议生成失败（{exc}），已展示通用建议，可稍后重试。")
            st.rerun()
        if c2.button("💬 就这门课提问", key=f"ask_course_{course.code}", use_container_width=True):
            create_new_session()
            st.session_state.current_view = "chat"
            prefill_question(f"我正在学习《{course.name}》，请帮我讲解：")
            st.rerun()


# --------------------------------------------------------------------------- #
# 视图路由
# --------------------------------------------------------------------------- #
if st.session_state.current_view == "chat":
    render_chat()
elif st.session_state.current_view == "courses":
    render_courses()
else:
    render_home()

# 按本次交互场景（新建/切换回顶、提交后定位最新）做一次性滚动定位；
# 普通渲染不携带标记，绝不强制滚动，避免短会话标题与顶部按钮被顶出可视区。
consume_scroll_action()

# “回到主页”点击瞬间立即回顶的全局钩子（监听器内部去重，每次渲染都注入也安全）
inject_instant_back_hook()
