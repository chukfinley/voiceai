"""Phase 0 demo: run cascade stack live with mic + speaker.

Usage:
    uv run python scripts/bootstrap.py [--option a|b] [--bg]

Options:
    a  Cascade: Qwen3-ASR + Qwen3-0.6B + Qwen3-TTS  (default, dev)
    b  Native: Qwen3-Omni-30B-A3B                   (needs H100)
    --bg  Enable background reasoning bridge (DashScope API)
    --vis Enable visual proactivity watcher
    --aec Enable AEC (requires webrtc-audio-processing)
"""
from __future__ import annotations

import argparse
import asyncio
import os

from voiceai.orchestrator.audio_io import MicLoop, SpeakerLoop
from voiceai.orchestrator.core import Orchestrator, TickLoop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--option", choices=["a", "b"], default="a")
    p.add_argument("--bg", action="store_true")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--aec", choices=["passthrough", "webrtc"], default="passthrough")
    return p


async def amain() -> None:
    args = build_parser().parse_args()
    orch = Orchestrator()

    orch.add(MicLoop())
    orch.add(SpeakerLoop())
    orch.add(TickLoop(period=0.5))

    if args.aec != "passthrough":
        from voiceai.orchestrator.aec import AECLoop

        orch.add(AECLoop(backend=args.aec))

    if args.option == "a":
        from voiceai.foreground.cascade import CascadeForeground

        orch.add(CascadeForeground())
    else:
        from voiceai.foreground.qwen_omni import OmniForeground

        orch.add(OmniForeground())

    if args.bg:
        from voiceai.background.qwen_max_api import BackgroundBridge

        if not os.getenv("DASHSCOPE_API_KEY"):
            print("warning: DASHSCOPE_API_KEY not set — bg bridge will no-op")
        orch.add(BackgroundBridge())

    if args.vis:
        from voiceai.visual.watcher import VisualWatcher

        orch.add(VisualWatcher())

    print(f"voiceai bootstrap: option={args.option} bg={args.bg} vis={args.vis}")
    print("speak. Ctrl-C to quit.")
    await orch.run()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
