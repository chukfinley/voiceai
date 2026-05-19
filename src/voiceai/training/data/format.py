"""200ms micro-turn frame format for training.

Goal: teach a Qwen3-Omni-style model to produce token sequences that interleave
USER + ASSISTANT streams aligned to 200ms ticks, matching TML's micro-turn
design and Moshi's dual-stream layout.

A "frame" is a 200ms slice. Each frame has:
  - tick token       : <t:N>            (frame index, integer)
  - user stream      : user audio tokens for that 200ms (or <silent>)
  - assistant stream : assistant audio tokens for that 200ms (or <silent>)
  - optional visual  : <visual:event=...> (sparse, only when fired)
  - optional bg      : <bg_result>...</bg_result> (when async result arrives)
  - optional control : <wait:Ns>, <barge>, <ack>, <thinking>

The text monologue (Moshi's "Inner Monologue") is interleaved within
assistant turns so text always leads audio by ~80ms.

Format example (text representation; actual tokens are codec IDs):

    <t:0>  <u:silent>            <a:silent>
    <t:1>  <u:silent>            <a:silent>
    <t:2>  <u:audio:a3,b7,c1...> <a:silent>
    <t:3>  <u:audio:...>         <a:silent>
    <t:4>  <u:audio:...>         <a:text:"Ja">  <a:audio:...>
    <t:5>  <u:silent>            <a:text:", was">  <a:audio:...>
    ...

This module emits:
  - flatten(frames) → list[int] token sequence
  - decode(ids)     → list[Frame]
  - synth_pairs()   → synthetic dialog (used by data/synth.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Frame rate constants
FRAME_MS = 200
FRAMES_PER_SEC = 1000 // FRAME_MS  # 5


# Special control tokens (string form for tokenizer training; real model uses IDs)
TOK_SILENT = "<silent>"
TOK_BARGE = "<barge>"
TOK_THINKING = "<thinking>"
TOK_ACK = "<ack>"
TOK_BG_QUERY_OPEN = "<background_query>"
TOK_BG_QUERY_CLOSE = "</background_query>"
TOK_BG_RESULT_OPEN = "<bg_result>"
TOK_BG_RESULT_CLOSE = "</bg_result>"


@dataclass
class Frame:
    idx: int                                # 200ms frame index since session start
    user_audio: list[int] = field(default_factory=list)    # codec ids, or [] = silent
    asst_audio: list[int] = field(default_factory=list)
    asst_text: str = ""                                    # text monologue piece
    visual: dict | None = None                             # {"event": ..., ...}
    bg_result: str | None = None
    control: list[str] = field(default_factory=list)       # extra control tokens


def flatten(frames: list[Frame]) -> list[str]:
    """Tokenize frames into the flat training sequence.

    Order within a frame (mirrors Moshi's RQ-Transformer layout):
        <t:N>
        <u:audio:...> OR <u:silent>
        <a:text:"...">  (if any — Inner Monologue lead)
        <a:audio:...> OR <a:silent>
        <visual:...>    (if any)
        <bg_result>...</bg_result>  (if any)
        <ctrl:...>      (zero or more)
    """
    out: list[str] = []
    for f in frames:
        out.append(f"<t:{f.idx}>")
        if f.user_audio:
            out.append("<u:audio:" + ",".join(str(c) for c in f.user_audio) + ">")
        else:
            out.append("<u:silent>")
        if f.asst_text:
            out.append(f'<a:text:"{f.asst_text}">')
        if f.asst_audio:
            out.append("<a:audio:" + ",".join(str(c) for c in f.asst_audio) + ">")
        else:
            out.append("<a:silent>")
        if f.visual:
            ev = f.visual.get("event", "?")
            c = f.visual.get("confidence", 0)
            out.append(f"<visual:{ev},conf={c:.2f}>")
        if f.bg_result:
            out.append(TOK_BG_RESULT_OPEN + f.bg_result + TOK_BG_RESULT_CLOSE)
        for ctl in f.control:
            out.append(ctl)
    return out


def decode(tokens: list[str]) -> list[Frame]:
    """Best-effort inverse of flatten() — useful for eval scripts."""
    frames: list[Frame] = []
    cur: Frame | None = None
    for tok in tokens:
        if tok.startswith("<t:"):
            if cur is not None:
                frames.append(cur)
            cur = Frame(idx=int(tok[3:-1]))
        elif cur is None:
            continue
        elif tok.startswith("<u:audio:"):
            cur.user_audio = [int(x) for x in tok[len("<u:audio:"):-1].split(",") if x]
        elif tok == "<u:silent>":
            cur.user_audio = []
        elif tok.startswith('<a:text:"'):
            cur.asst_text = tok[len('<a:text:"'):-2]
        elif tok.startswith("<a:audio:"):
            cur.asst_audio = [int(x) for x in tok[len("<a:audio:"):-1].split(",") if x]
        elif tok == "<a:silent>":
            cur.asst_audio = []
        elif tok.startswith("<visual:"):
            body = tok[len("<visual:"):-1]
            parts = body.split(",")
            ev = parts[0]
            conf = 0.0
            for p in parts[1:]:
                if p.startswith("conf="):
                    conf = float(p[5:])
            cur.visual = {"event": ev, "confidence": conf}
        elif tok.startswith(TOK_BG_RESULT_OPEN):
            cur.bg_result = tok[len(TOK_BG_RESULT_OPEN):-len(TOK_BG_RESULT_CLOSE)]
        else:
            cur.control.append(tok)
    if cur is not None:
        frames.append(cur)
    return frames
