# -*- coding: utf-8 -*-
"""
DeepSeek API 客户端
===================

封装在线课程答疑系统与 DeepSeek 大模型之间的完整调用流程：

1. 请求参数配置：模型、温度、最大生成长度、流式开关等；
2. 身份验证：通过 ``Authorization: Bearer <API_KEY>`` 请求头完成；
3. 响应处理：解析 DeepSeek 返回的 SSE（Server-Sent Events）流式数据；
4. 错误捕获：覆盖网络异常、超时、HTTP 状态码错误、JSON 解析失败等情况。

接口文档：https://api-docs.deepseek.com/
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Generator, List, Optional, Tuple

import requests

# 可选：从项目根目录的 .env 文件中读取环境变量（未安装 python-dotenv 时静默降级）
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - 仅为缺少可选依赖时的降级处理
    pass

# --------------------------------------------------------------------------- #
# 常量配置
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"          # DeepSeek-V3 通用对话模型
DEFAULT_TIMEOUT = 60                     # 流式读取超时时间（秒）
CONNECT_TIMEOUT = 10                     # 建连超时时间（秒）

# 常见 HTTP 状态码 -> 面向学生/教师的中文提示
_FRIENDLY_MESSAGES: Dict[int, str] = {
    400: "请求参数有误，请检查问题内容或参数配置后重试。",
    401: "API Key 无效或已过期，请在左侧边栏重新填写正确的密钥。",
    402: "DeepSeek 账户余额不足，请前往平台充值后再试。",
    403: "访问被拒绝，请确认当前 API Key 是否具备调用权限。",
    404: "接口地址不存在，请检查 BASE_URL 配置是否正确。",
    422: "请求参数无法被服务器识别，请检查消息格式。",
    429: "请求过于频繁或账户额度已用尽，请稍后再试。",
    500: "DeepSeek 服务内部异常，请稍后重试。",
    502: "网关错误，DeepSeek 服务暂时不可用，请稍后重试。",
    503: "服务暂不可用（可能正在维护），请稍后重试。",
    504: "网关超时，服务器未能及时返回结果，请稍后重试。",
}


# --------------------------------------------------------------------------- #
# 自定义异常
# --------------------------------------------------------------------------- #
class DeepSeekError(Exception):
    """所有 DeepSeek 调用相关异常的基类。"""


class DeepSeekConfigError(DeepSeekError):
    """API Key 缺失等配置类错误。"""


class DeepSeekAPIError(DeepSeekError):
    """DeepSeek 服务返回了非 200 的 HTTP 状态码。"""

    def __init__(self, status_code: int, message: str, raw: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.raw = raw
        friendly = _FRIENDLY_MESSAGES.get(status_code)
        full = f"[{status_code}] {message}"
        if friendly:
            full = f"{friendly}（原始错误：{message}）"
        super().__init__(full)


class DeepSeekNetworkError(DeepSeekError):
    """网络连接、超时等传输层错误。"""


# --------------------------------------------------------------------------- #
# 客户端实现
# --------------------------------------------------------------------------- #
class DeepSeekClient:
    """DeepSeek 对话补全接口客户端（支持流式输出）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        # API Key 优先使用显式传入值，其次读取环境变量 DEEPSEEK_API_KEY
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        self.base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.model = model
        self.timeout = timeout

        self._session = requests.Session()
        # 本机若开启了系统代理（Clash / VPN 等），直连国内接口可能出现 SSL 错误，
        # 默认绕过系统代理；如需走代理可设置环境变量 DEEPSEEK_USE_PROXY=1
        if os.getenv("DEEPSEEK_USE_PROXY", "0") != "1":
            self._session.trust_env = False

    # ----------------------------- 基础属性 ----------------------------- #
    @property
    def configured(self) -> bool:
        """是否已配置可用的 API Key。"""
        return bool(self.api_key) and self.api_key.startswith("sk-")

    def _headers(self) -> Dict[str, str]:
        """构造带身份验证信息的请求头。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> Dict:
        """构造请求体（请求参数配置）。"""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,   # 0~2，值越小回答越稳定严谨
            "max_tokens": max_tokens,     # 单次回答最大 token 数
            "stream": stream,             # 是否流式返回
        }

    # ----------------------------- 流式问答 ----------------------------- #
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        以流式方式调用 DeepSeek 对话接口，逐段 yield 文本增量。

        :param messages: OpenAI 风格消息列表，如 [{"role": "user", "content": "..."}]
        :param temperature: 采样温度
        :param max_tokens: 最大生成 token 数
        :raises DeepSeekConfigError: 未配置 API Key
        :raises DeepSeekAPIError: 服务返回错误状态码
        :raises DeepSeekNetworkError: 网络层异常
        """
        if not self.configured:
            raise DeepSeekConfigError(
                "尚未配置 DeepSeek API Key，请在项目根目录的 .env 文件中设置 "
                "DEEPSEEK_API_KEY（或通过同名环境变量提供）。"
            )

        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)

        try:
            response = self._session.post(
                url,
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=(CONNECT_TIMEOUT, self.timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise DeepSeekNetworkError(
                "连接 DeepSeek 服务超时，请检查网络后重试。"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise DeepSeekNetworkError(
                "无法连接 DeepSeek 服务，请检查网络连接或接口地址配置。"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise DeepSeekNetworkError(f"发送请求时发生网络错误：{exc}") from exc

        with response:
            if response.status_code != 200:
                self._raise_api_error(response)

            # 解析 SSE 流：每行形如 ``data: {"choices": [...]}``
            try:
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()
                    if data == "[DONE]":  # DeepSeek 流式结束标记
                        break
                    if not data:
                        continue

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # 单个分片解析失败不应中断整段回答
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
            except requests.exceptions.Timeout as exc:
                raise DeepSeekNetworkError(
                    "等待 DeepSeek 响应超时，请缩短问题或稍后重试。"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise DeepSeekNetworkError(f"接收流式数据中断：{exc}") from exc

    # ----------------------------- 连通性检测 ----------------------------- #
    def check_connection(self) -> Tuple[bool, str, int]:
        """
        发起一次极轻量（max_tokens=1）的非流式请求，用于在界面上展示系统状态。

        :return: (是否正常, 状态描述, 耗时毫秒)
        """
        if not self.configured:
            return False, "未配置 API Key", 0

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
        started = time.time()
        try:
            response = self._session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=(CONNECT_TIMEOUT, 30),
            )
        except requests.exceptions.RequestException as exc:
            return False, f"无法连接服务：{exc.__class__.__name__}", 0

        elapsed_ms = int((time.time() - started) * 1000)
        if response.status_code == 200:
            return True, "连接正常", elapsed_ms

        # 提取错误信息
        try:
            err_data = response.json()
            message = (err_data.get("error") or {}).get("message", response.text[:120])
        except ValueError:
            message = response.text[:120]
        friendly = _FRIENDLY_MESSAGES.get(response.status_code, message)
        return False, f"{response.status_code} · {friendly}", elapsed_ms

    # ----------------------------- 内部工具 ----------------------------- #
    @staticmethod
    def _raise_api_error(response: requests.Response) -> None:
        """从错误响应中提取信息并抛出 DeepSeekAPIError。"""
        try:
            err_data = response.json()
            message = (err_data.get("error") or {}).get("message", "")
            if not message:
                message = json.dumps(err_data, ensure_ascii=False)[:200]
        except ValueError:
            message = response.text[:200] or "未知服务端错误"
        raise DeepSeekAPIError(response.status_code, message, raw=response.text)
