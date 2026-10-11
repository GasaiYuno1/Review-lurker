"""Клиент OpenAI-совместимого chat API без вендорских SDK (urllib): работает с OpenAI, HF router,
Ollama, vLLM, LM Studio и корпоративным шлюзом — меняется только base_url/model в конфиге.

- chat(messages) -> текст ответа; chat_json(...) -> распарсенный JSON (снимает ```json-ограждения).
- Кэш на диске по хэшу (модель + сообщения + параметры): повторные запуски не тратят кредиты.
- Повторы с экспоненциальной паузой на 429/5xx/сетевых ошибках.
- FakeLLM — та же сигнатура для тестов без сети.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .config import LLMConfig

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_dir: Optional[str | Path] = None):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) / "llm" if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0  # реальные обращения к API (без кэша)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_usage: dict = {}
        self._json_ok = cfg.json_mode
        self._last_call = 0.0

    # ---- публичный интерфейс ----
    def chat(self, messages: Sequence[Message], *, json_mode: bool = False,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        messages = [dict(m) for m in messages]
        if self.cfg.message_suffix and messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = str(messages[-1]["content"]) + self.cfg.message_suffix
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if max_tokens or self.cfg.max_tokens:
            body["max_tokens"] = max_tokens or self.cfg.max_tokens
        if self.cfg.extra_body:
            body.update(self.cfg.extra_body)
        if json_mode and self._json_ok:
            body["response_format"] = {"type": "json_object"}
        key = self._cache_key(body)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        try:
            text = self._request(body)
        except LLMError as e:
            # endpoint не знает response_format → один раз повторяем без него и запоминаем
            if json_mode and self._json_ok and "HTTP 400" in str(e):
                self._json_ok = False
                body.pop("response_format", None)
                text = self._request(body)
            else:
                raise
        self._cache_put(key, text)
        return text

    def ping(self) -> dict:
        """GET /models: доступность endpoint и ключа; возвращает {"ok", "status", "models", "error"}."""
        url = self.cfg.base_url.rstrip("/") + "/models"
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(self.cfg.timeout_s, 30)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
            return {"ok": True, "status": resp.status, "models": models, "error": ""}
        except urllib.error.HTTPError as e:
            return {"ok": False, "status": e.code, "models": [], "error": e.read().decode("utf-8", "replace")[:300]}
        except Exception as e:
            return {"ok": False, "status": 0, "models": [], "error": str(e)}

    def _throttle(self) -> None:
        if self.cfg.rate_limit_rpm and self.cfg.rate_limit_rpm > 0:
            gap = 60.0 / self.cfg.rate_limit_rpm
            wait = self._last_call + gap - time.time()
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.time()

    def chat_json(self, messages: Sequence[Message], **kw) -> Any:
        text = self.chat(messages, json_mode=True, **kw)
        return parse_json(text)

    # ---- внутреннее ----
    def _request(self, body: dict) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        delay = 2.0
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                self._throttle()
                self.calls += 1
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                u = payload.get("usage") or {}
                self.last_usage = u
                for k in ("prompt_tokens", "completion_tokens"):
                    self.usage[k] += int(u.get(k) or 0)
                text = payload["choices"][0]["message"]["content"] or ""
                return strip_thinking(text) if self.cfg.strip_thinking else text
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last = LLMError(f"HTTP {e.code} от {url}: {detail}")
                if e.code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise last
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after and str(retry_after).isdigit():
                    delay = max(delay, float(retry_after))
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = LLMError(f"сеть: {e}")
            except (KeyError, IndexError, ValueError) as e:
                raise LLMError(f"неожиданный ответ API: {e}")
            if attempt < self.cfg.max_retries:
                time.sleep(delay)
                delay *= 2
        raise last or LLMError("неизвестная ошибка")

    def _cache_key(self, body: dict) -> str:
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))["content"]
        return None

    def _cache_put(self, key: str, text: str) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps({"model": self.cfg.model, "content": text}, ensure_ascii=False), encoding="utf-8")


class FakeLLM:
    """Заглушка для тестов: отвечает фиксированной строкой или результатом функции от сообщений."""

    def __init__(self, reply: str | Callable[[Sequence[Message]], str] = "{}"):
        self.reply = reply
        self.calls = 0
        self.history: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], **kw) -> str:
        self.calls += 1
        self.history.append(list(messages))
        return self.reply(messages) if callable(self.reply) else self.reply

    def chat_json(self, messages: Sequence[Message], **kw) -> Any:
        return parse_json(self.chat(messages, **kw))


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text: str) -> str:
    """Убирает рассуждения <think>…</think> (Qwen3, DeepSeek-R1); незакрытый <think> — всё после него считается ответом."""
    text = _THINK.sub("", text)
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0] or text
    return text.strip()


def parse_json(text: str) -> Any:
    """JSON из ответа модели: снимает ```-ограждения, при мусоре вокруг берёт первый {...} или [...]."""
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = text.find(open_), text.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"ответ модели не JSON: {text[:200]!r}")
