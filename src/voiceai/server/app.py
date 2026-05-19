"""FastAPI + WebSocket inference server.

Protocol (binary frames over /ws):
  client → server  : raw 24kHz mono PCM int16 frames (80ms = 1920 samples = 3840 bytes)
  server → client  : raw 24kHz mono PCM int16 frames (assistant audio)

Server pipeline per incoming frame:
  1. Push PCM into Mimi encoder streaming state.
  2. When 8 codebook tokens emitted (every 80ms), append to model context.
  3. Run model forward, sample next assistant audio tokens.
  4. Decode assistant tokens with Mimi decoder.
  5. Send PCM back to client.

Run:
    uv run python -m voiceai.server.app --model runs/stage3/final
"""
from __future__ import annotations

import argparse
import asyncio
import struct
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse


CHUNK_MS = 80
CHUNK_SAMPLES = 24000 * CHUNK_MS // 1000  # 1920


HTML_PAGE = """<!doctype html>
<html><head><title>voiceai</title></head>
<body style="font-family: sans-serif; max-width: 720px; margin: 2em auto">
<h1>voiceai live</h1>
<button id=btn>start</button>
<p id=status>idle</p>
<script>
const btn = document.getElementById('btn');
const status = document.getElementById('status');
let ws, audioCtx, micStream, mediaStreamSource, scriptNode;
btn.onclick = async () => {
  if (ws) { ws.close(); audioCtx?.close(); return btn.innerText='start'; }
  status.innerText = 'connecting';
  ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = async () => {
    status.innerText = 'connected';
    btn.innerText = 'stop';
    audioCtx = new AudioContext({sampleRate: 24000});
    micStream = await navigator.mediaDevices.getUserMedia({audio: true});
    mediaStreamSource = audioCtx.createMediaStreamSource(micStream);
    scriptNode = audioCtx.createScriptProcessor(1920, 1, 1);
    mediaStreamSource.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);
    scriptNode.onaudioprocess = (e) => {
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) i16[i] = Math.max(-32768, Math.min(32767, f32[i]*32767));
      if (ws.readyState === 1) ws.send(i16.buffer);
    };
  };
  ws.onmessage = (ev) => {
    const i16 = new Int16Array(ev.data);
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i]/32768;
    const buf = audioCtx.createBuffer(1, f32.length, 24000);
    buf.copyToChannel(f32, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf; src.connect(audioCtx.destination); src.start();
  };
  ws.onclose = () => { status.innerText='closed'; btn.innerText='start'; };
};
</script>
</body></html>
"""


app = FastAPI()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = app.state.session_factory()
    try:
        await session.run(ws)
    except Exception as e:
        print(f"ws error: {e}")
    finally:
        await ws.close()


class InferenceSession:
    """One open browser connection → one model session."""

    def __init__(self, model, mimi, device: str) -> None:
        self.model = model
        self.mimi = mimi
        self.device = device
        self.user_token_buffer: list[torch.Tensor] = []
        self.context_codes: list[torch.Tensor] = []
        self.elapsed_frames = 0

    async def run(self, ws: WebSocket) -> None:
        outgoing: asyncio.Queue[bytes] = asyncio.Queue()

        async def receiver():
            async for msg in _ws_iter(ws):
                if not isinstance(msg, bytes):
                    continue
                pcm_int16 = np.frombuffer(msg, dtype=np.int16)
                if len(pcm_int16) == 0:
                    continue
                pcm = pcm_int16.astype(np.float32) / 32768.0
                pcm = pcm[: CHUNK_SAMPLES]
                if len(pcm) < CHUNK_SAMPLES:
                    pcm = np.pad(pcm, (0, CHUNK_SAMPLES - len(pcm)))
                await self._step(pcm, outgoing)

        async def sender():
            while True:
                pcm = await outgoing.get()
                await ws.send_bytes(pcm)

        send_task = asyncio.create_task(sender())
        try:
            await receiver()
        finally:
            send_task.cancel()

    async def _step(self, pcm: np.ndarray, outgoing: asyncio.Queue) -> None:
        t = torch.from_numpy(pcm).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.bfloat16)
        with torch.no_grad():
            codes = self.mimi.encode(t)  # [1, K, T_frame]
        self.user_token_buffer.append(codes)
        all_codes = torch.cat(self.user_token_buffer, dim=2)[:, :, -200:]

        attn = torch.ones(1, all_codes.shape[2], device=self.device, dtype=torch.long)
        text_ids = torch.full(
            (1, all_codes.shape[2]),
            self.model.tokenizer.pad_token_id or 0,
            device=self.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            out = self.model(text_ids=text_ids, user_audio_codes=all_codes, attention_mask=attn)
            asst_logits = out["asst_audio_logits"][:, :, -codes.shape[2] :, :]
            asst_codes = asst_logits.argmax(dim=-1)

        silent_id = self.model.cfg.codebook_size
        if (asst_codes == silent_id).all():
            await outgoing.put(np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes())
            return

        with torch.no_grad():
            asst_pcm = self.mimi.decode(asst_codes)[0, 0].float().cpu().numpy()
        out_int16 = np.clip(asst_pcm * 32767, -32768, 32767).astype(np.int16)[:CHUNK_SAMPLES]
        if len(out_int16) < CHUNK_SAMPLES:
            out_int16 = np.pad(out_int16, (0, CHUNK_SAMPLES - len(out_int16)))
        await outgoing.put(out_int16.tobytes())
        self.elapsed_frames += codes.shape[2]


async def _ws_iter(ws: WebSocket):
    while True:
        msg = await ws.receive()
        if msg.get("type") == "websocket.disconnect":
            return
        if "bytes" in msg and msg["bytes"] is not None:
            yield msg["bytes"]
        elif "text" in msg and msg["text"] is not None:
            yield msg["text"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import uvicorn

    from ..model.mimi_utils import load_mimi
    from ..model.voiceai_lm import VoiceAILM

    print(f"loading model {args.model}…")
    model = VoiceAILM.from_pretrained(args.model).to(args.device).eval()
    mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    app.state.session_factory = lambda: InferenceSession(model, mimi, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
