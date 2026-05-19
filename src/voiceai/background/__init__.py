"""Pluggable background-LLM bridge — single OpenAI-compatible client.

Foreground emits <background_query>X</background_query>. Orchestrator
dispatches to the bridge async. Result returns as <bg_result id=N>...</bg_result>
at next frame boundary.

Quick usage:
    from voiceai.background import get_bridge
    bridge = get_bridge("openai")           # or "anthropic", "dashscope", "vllm", etc.
    text = await bridge.query("what is paraguay's capital?")

Streaming:
    async for tok in bridge.stream("..."):
        ...

Custom endpoint:
    bridge = get_bridge("custom", base_url="https://my.api/v1", model="my-model")
"""
from __future__ import annotations

from .openai_compat import BridgeConfig, OpenAICompatBridge, PROVIDERS, StubBridge


def get_bridge(provider: str = "openai", **kwargs):
    """Construct a bridge. Provider names: see PROVIDERS dict."""
    if provider == "stub":
        return StubBridge(**kwargs)
    return OpenAICompatBridge(cfg=BridgeConfig(provider=provider, **kwargs))


__all__ = ["get_bridge", "OpenAICompatBridge", "BridgeConfig", "StubBridge", "PROVIDERS"]
