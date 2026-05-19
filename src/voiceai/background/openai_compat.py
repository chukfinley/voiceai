"""Single OpenAI-compatible bridge.

Works with any /v1/chat/completions endpoint:
  - OpenAI            base_url = "https://api.openai.com/v1"
  - Anthropic         base_url = "https://api.anthropic.com/v1"     (OpenAI-compat)
  - DashScope (Qwen)  base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  - Google Gemini     base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
  - Local vLLM        base_url = "http://localhost:8000/v1"
  - Ollama            base_url = "http://localhost:11434/v1"
  - LM Studio         base_url = "http://localhost:1234/v1"

Streaming returns an async iterator of token chunks.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass


PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "gpt-4.1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "claude-opus-4-7", "ANTHROPIC_API_KEY"),
    "dashscope": (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "qwen3-max",
        "DASHSCOPE_API_KEY",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    ),
    "vllm": ("http://localhost:8000/v1", "Qwen/Qwen3-32B", "VOICEAI_LOCAL_KEY"),
    "ollama": ("http://localhost:11434/v1", "qwen3.5:0.8b", "VOICEAI_OLLAMA_KEY"),
    "lmstudio": ("http://localhost:1234/v1", "local", "VOICEAI_LMSTUDIO_KEY"),
}


DEFAULT_SYSTEM = (
    "You are a background reasoning model. Reply precisely, 1-3 sentences. No filler."
)


@dataclass
class BridgeConfig:
    provider: str = "openai"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = 60.0


class OpenAICompatBridge:
    """Single client for any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, cfg: BridgeConfig | None = None, **kwargs) -> None:
        cfg = cfg or BridgeConfig(**kwargs)
        provider_defaults = PROVIDERS.get(cfg.provider)
        if provider_defaults is None and (cfg.base_url is None or cfg.model is None):
            raise ValueError(f"unknown provider {cfg.provider}; set base_url+model explicitly")

        if provider_defaults:
            default_url, default_model, default_env = provider_defaults
            self.base_url = cfg.base_url or default_url
            self.model = cfg.model or default_model
            self.api_key = cfg.api_key or os.getenv(default_env) or "sk-local"
        else:
            self.base_url = cfg.base_url
            self.model = cfg.model
            self.api_key = cfg.api_key or "sk-local"

        self.timeout = cfg.timeout
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def query(
        self,
        text: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        client = self._ensure_client()
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content or ""

    async def stream(
        self,
        text: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        client = self._ensure_client()
        stream = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Full multi-turn variant."""
        client = self._ensure_client()
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


class StubBridge:
    """Offline placeholder for tests."""

    def __init__(self, response: str = "Stub answer.") -> None:
        self.response = response

    async def query(self, text: str, **_) -> str:
        return f"{self.response} (query: {text[:60]})"

    async def stream(self, text: str, **_):
        for word in self.response.split():
            yield word + " "

    async def chat(self, messages, **_) -> str:
        return self.response
