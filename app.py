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
    "你是一名在线课程的智能答疑助教，服务对象为正在学习编程与计算机相关课程的学生。"
    "请遵循以下原则回答问题：\n"
    "1. 先给出直接、准确的结论，再分步骤讲解原理，必要时配合可运行的代码示例；\n"
    "2. 善用类比、图示（Markdown 表格/列表）帮助理解，语言亲切、鼓励学生思考；\n"
    "3. 学生概念混淆时，主动指出易错点并给出辨析；答完后可提出 1 个延伸思考题；\n"
    "4. 只回答与课程学习相关的问题，对无关或无法确定的内容如实说明，不要编造事实；\n"
    "5. 全部使用简体中文回答，代码注释清晰，Markdown 排版整洁。"
)

SUGGESTED_QUESTIONS = [
    "什么是 Python 中的装饰器？请举个例子",
    "用生活中的例子解释一下什么是递归",
    "帮我梳理面向对象的三大特性",
    "出一道关于类继承的练习题并给出解析",
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
        "temperature": 0.7,
        "max_tokens": 2048,
        "last_latency": None,         # 最近一次回答耗时（秒）
        # ---- 双视图路由相关 ----
        "current_view": "home",       # home=主页 / chat=独立问答页
        "need_answer": False,         # 问答页是否有待流式生成的回答
        "last_error": None,           # 最近一次生成失败的提示（跨 rerun 保留）
        # 本次重绘后主区域的一次性滚动定位："top"=回顶 / "bottom"=定位最新 / None=不干预
        "scroll_action": None,
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
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def prefill_question(text: str) -> None:
    """把内容回填到聊天输入框（需在 chat_input 组件实例化前写入其 key）。"""
    st.session_state["question_input"] = text


def scroll_main(position: str = "bottom", smooth: bool = False) -> None:
    """
    把主区域滚动条一次性定位到顶部或底部。

    只有内容真正超出可视区时才滚动，避免短会话被强制位移、
    把标题与顶部按钮顶出可视区（页面显示不全的根因）。
    """
    components_html(
        f"""
        <script>
        (function () {{
            var doc = window.parent.document;
            // Streamlit 1.63 主容器自带“滚到底”行为（stAppScrollToBottomContainer），
            // 按钮触发重绘后会在不确定的时机强制滚动。这里启动一个 1.5s 的
            // 事件驱动“滚动锁”：期间任何非用户发起的滚动都会被立即纠回；
            // 用户一旦主动滚轮/触摸/按键翻页则立刻放行，绝不和用户抢滚动条。
            var POSITION = '{position}';
            var ENFORCE_MS = 1500;
            var start = Date.now();
            var userScrolled = false;
            function onUser() {{ userScrolled = true; }}

            function desiredTop(el) {{
                if (POSITION === 'top') return 0;
                // 底部停靠输入栏会预留滚动空间：最后一条消息没超出
                // “停靠栏上方可视区”时回到顶部，避免标题与按钮被顶出。
                var dock = doc.querySelector('[data-testid="stBottom"]');
                var dockH = dock ? dock.offsetHeight : 0;
                var msgs = el.querySelectorAll('[data-testid="stChatMessage"]');
                if (msgs.length) {{
                    var elTop = el.getBoundingClientRect().top;
                    var lastBottom = msgs[msgs.length - 1].getBoundingClientRect().bottom - elTop;
                    if (lastBottom <= el.clientHeight - dockH + 8) return 0;
                }}
                return el.scrollHeight;
            }}
            function enforce() {{
                if (userScrolled || Date.now() - start > ENFORCE_MS) return false;
                var el = doc.querySelector('section.stMain')
                      || doc.querySelector('section[data-testid="stMain"]')
                      || doc.querySelector('section.main');
                if (!el) return false;
                var want = desiredTop(el);
                if (Math.abs(el.scrollTop - want) > 1) {{
                    el.scrollTo({{ top: want, behavior: 'auto' }});
                }}
                return true;
            }}
            var el0 = doc.querySelector('section.stMain');
            if (el0) {{
                el0.addEventListener('wheel', onUser, {{ passive: true }});
                el0.addEventListener('touchstart', onUser, {{ passive: true }});
                doc.addEventListener('keydown', onUser, true);
                el0.addEventListener('scroll', function () {{
                    if (!userScrolled && Date.now() - start <= ENFORCE_MS) enforce();
                }}, {{ passive: true }});
            }}
            enforce();
            var timer = setInterval(function () {{
                if (!enforce()) clearInterval(timer);
            }}, 100);
        }})();
        </script>
        """,
        height=0,
    )


def consume_scroll_action() -> None:
    """按场景标记执行一次性滚动定位后清除标记；无标记的普通渲染绝不干预滚动条。"""
    action = st.session_state.get("scroll_action")
    if action in ("top", "bottom"):
        scroll_main(action, smooth=False)
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
        scroll_main("bottom", smooth=True)
        try:
            stream = client.chat_stream(
                build_api_messages(),
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
            # 节流：约每 0.25 秒滚动一次，避免频繁注入脚本
            last_scroll = 0.0
            for chunk in stream:
                full_answer += chunk
                # 尾部光标模拟逐字输出效果
                answer_slot.markdown(full_answer + " ▌")
                now = time.time()
                if now - last_scroll >= 0.25:
                    scroll_main("bottom", smooth=True)
                    last_scroll = now

            if full_answer:
                answer_slot.markdown(full_answer)
                scroll_main("bottom", smooth=True)
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
                <p class="qa-side-title">智能课程答疑系统</p>
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

    # ---- 2. 模型与回答参数 ---- #
    st.markdown("##### 模型与回答参数")
    # st.caption(f"当前模型：`{DEFAULT_MODEL}`（DeepSeek-V3 通用对话模型）")
    st.slider(
        "回答随机性 temperature",
        min_value=0.0, max_value=1.5,
        step=0.1,
        help="数值越小回答越严谨稳定，值越大越发散有创意。课程答疑建议 0.3~0.8。",
        key="temperature",
    )
    st.slider(
        "最大回答长度 max_tokens",
        min_value=256, max_value=4096,
        step=128,
        help="单次回答允许生成的最大 token 数，期望回答较长时可适当调大。",
        key="max_tokens",
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
                基于DeepSeek-V3 通用对话模型搭建
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
            <h1>🎓 基于大模型的在线课程答疑系统</h1>
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
            <p>我是你的智能课程答疑助教。无论是课程概念辨析、时间安排还是练习题讲解，
               都可以直接在下方输入框提问。提交后将自动创建新会话并进入独立问答页，
               答案实时流式呈现。</p>
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
# 视图路由
# --------------------------------------------------------------------------- #
if st.session_state.current_view == "chat":
    render_chat()
else:
    render_home()

# 按本次交互场景（新建/切换回顶、提交后定位最新）做一次性滚动定位；
# 普通渲染不携带标记，绝不强制滚动，避免短会话标题与顶部按钮被顶出可视区。
consume_scroll_action()
