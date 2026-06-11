#!/usr/bin/env python
"""Real (small) Stage-1 ASR on a free Colab T4 — generalises to NEW clips.

Unlike the overfit demo (memorises a few clips), this streams a few thousand
real LibriSpeech clips, trains the Stage-1 audio adapter on them, and then
transcribes a HELD-OUT set the model never saw. Rough WER (it's tiny), but
real audio->text on unseen speech.

~1h on a free T4:
    uv run --extra train python scripts/colab_asr_mini.py

Save the trained model to HuggingFace too:
    uv run --extra train python scripts/colab_asr_mini.py --push-to-hub
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

DATA = REPO / "data" / "asr_mini"
WAVS = DATA / "wav"
TRAIN_MANIFEST = DATA / "train.jsonl"
TEST_MANIFEST = DATA / "test.jsonl"
OUT = REPO / "runs" / "asr_mini"


def build_data(n_train: int, n_test: int) -> None:
    if TRAIN_MANIFEST.exists() and TEST_MANIFEST.exists():
        ntr = sum(1 for _ in TRAIN_MANIFEST.open())
        if ntr >= n_train * 0.9:
            print(f"[data] reuse {ntr} train clips")
            return
    import soundfile as sf
    from datasets import Audio, load_dataset

    print(f"[data] streaming {n_train + n_test} real LibriSpeech clips…")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    WAVS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= n_train + n_test:
            break
        au = ex["audio"]
        arr, sr = (
            sf.read(io.BytesIO(au["bytes"])) if au.get("bytes") else sf.read(au["path"])
        )
        txt = (ex.get("text") or "").strip()
        if not txt:
            continue
        wav = WAVS / f"clip_{i:05d}.wav"
        sf.write(str(wav), arr, sr)
        rows.append({"audio": str(wav), "text": txt, "duration": round(len(arr) / sr, 2)})
    # last n_test are the held-out (never trained) clips
    train, test = rows[:-n_test], rows[-n_test:]
    TRAIN_MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in train))
    TEST_MANIFEST.write_text("".join(json.dumps(r) + "\n" for r in test))
    print(f"[data] {len(train)} train  +  {len(test)} held-out test clips")


def train(backbone: str, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "voiceai.training.stage1_adapter",
        "--manifest", str(TRAIN_MANIFEST),
        "--output", str(OUT),
        "--backbone", backbone,
        "--steps", str(steps),
        "--batch-size", "4",
        "--grad-accum", "2",
        "--lr", "3e-4",
        "--warmup", "100",
        "--log-every", "50",
        "--ckpt-every", "100000",
        "--max-audio-s", "14",
        "--mix-asr", "0.8",   # favour ASR (audio->text) for this demo
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--wandb-disable",
    ]
    print("[train] real Stage-1 ASR mini:\n  " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("[train] FAILED")


def evaluate_heldout(max_text_len: int = 120) -> None:
    """Transcribe the held-out clips the model never saw — real generalisation."""
    import torch

    from voiceai.model.mimi_utils import load_mimi, mimi_encode, resample_to_mimi
    from voiceai.model.voiceai_lm import VoiceAILM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    import soundfile as sf

    model = VoiceAILM.from_pretrained(OUT / "final").to(device).eval()
    mimi = load_mimi(device=device, dtype=dtype)
    tok = model.tokenizer
    emb = model.backbone.get_input_embeddings()
    bb_dtype = emb.weight.dtype
    eos = tok.eos_token_id
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(line) for line in TEST_MANIFEST.open() if line.strip()]
    print(f"\n[eval] transcribe {len(rows)} HELD-OUT clips (never trained):\n")
    refs, hyps = [], []
    for r in rows:
        arr, sr = sf.read(r["audio"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        t = torch.from_numpy(arr)[None, None].to(device)
        t = resample_to_mimi(t, sr).to(dtype)
        with torch.no_grad():
            codes = mimi_encode(mimi, t)
            embeds = model.audio_in(codes).to(bb_dtype)
            gen: list[int] = []
            for _ in range(max_text_len):
                out = model.backbone(inputs_embeds=embeds, output_hidden_states=True, return_dict=True)
                nid = int(model.backbone.lm_head(out.hidden_states[-1][:, -1:, :]).argmax(-1).item())
                if nid == eos:
                    break
                gen.append(nid)
                embeds = torch.cat([embeds, emb(torch.tensor([[nid]], device=device)).to(bb_dtype)], dim=1)
        hyp = tok.decode(gen, skip_special_tokens=True).strip()
        refs.append(r["text"].lower())
        hyps.append(hyp.lower())
        print(f"  REF: {r['text'][:90]}")
        print(f"  HYP: {hyp[:90]}\n")
    try:
        from jiwer import wer
        print(f"[eval] WER on held-out: {wer(refs, hyps):.1%}  (lower = better; rough at this scale)")
    except Exception:
        pass


def push_to_hub(repo: str | None) -> None:
    from huggingface_hub import HfApi

    final = OUT / "final"
    if not (final / "adapters.pt").exists():
        sys.exit(f"[hub] no checkpoint at {final}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if not repo:
        repo = f"{api.whoami()['name']}/voiceai-asr-mini"
    api.create_repo(repo, exist_ok=True, private=True, repo_type="model")
    print(f"[hub] uploading {final} -> {repo} …")
    api.upload_folder(folder_path=str(final), repo_id=repo, repo_type="model")
    print(f"[hub] saved: https://huggingface.co/{repo}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    p.add_argument("--clips", type=int, default=3000)
    p.add_argument("--test-clips", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo", default=None)
    a = p.parse_args()
    build_data(a.clips, a.test_clips)
    train(a.backbone, a.steps)
    evaluate_heldout()
    if a.push_to_hub:
        push_to_hub(a.hub_repo)
