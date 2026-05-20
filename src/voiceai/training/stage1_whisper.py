"""Stage 1 (alternative): train Whisper-encoder → Qwen bridge.

Trains only the small bridge MLP (~5M params). Whisper-encoder and Qwen
LLM are frozen. Converges in ~20-60 minutes on RTX 3090.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--whisper-id", default="openai/whisper-small")
    p.add_argument("--llm-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--max-text-len", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="voiceai")
    p.add_argument("--wandb-disable", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None, help="Path to a ckpt dir; loads bridge.pt to warm-start.")
    return p


class WhisperASRDataset(IterableDataset):
    def __init__(self, manifest: str, tokenizer, feat_ext, max_text_len: int = 120, seed: int = 0):
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.feat_ext = feat_ext
        self.max_text_len = max_text_len
        self.seed = seed
        with open(manifest) as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):
        rng = random.Random(self.seed)
        order = list(range(len(self.lines)))
        rng.shuffle(order)
        for idx in order:
            try:
                meta = json.loads(self.lines[idx])
                audio, sr = sf.read(meta["audio"], dtype="float32")
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            feats = self.feat_ext(audio, sampling_rate=16000, return_tensors="pt")
            text_ids = self.tokenizer.encode(meta["text"], add_special_tokens=False)
            text_ids = text_ids[: self.max_text_len]
            # Append EOS so model learns to stop
            if self.tokenizer.eos_token_id is not None:
                text_ids = text_ids + [self.tokenizer.eos_token_id]
            yield {
                "features": feats.input_features[0],  # [80, 3000]
                "text_ids": np.array(text_ids, dtype=np.int64),
            }


def collate(batch, pad_id: int):
    B = len(batch)
    Tmax_t = max(len(s["text_ids"]) for s in batch)
    feats = torch.stack([s["features"] for s in batch], dim=0)
    text_ids = np.full((B, Tmax_t), pad_id, dtype=np.int64)
    attn = np.zeros((B, Tmax_t), dtype=np.int64)
    for i, s in enumerate(batch):
        L = len(s["text_ids"])
        text_ids[i, :L] = s["text_ids"]
        attn[i, :L] = 1
    return {
        "features": feats,
        "text_ids": torch.from_numpy(text_ids),
        "text_attn": torch.from_numpy(attn),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    torch.manual_seed(args.seed)

    from ..model.whisper_lm import WhisperLM, WhisperLMConfig

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = WhisperLMConfig(whisper_id=args.whisper_id, llm_id=args.llm_id, dtype=args.dtype)
    model = WhisperLM(cfg).to(device)

    if args.resume_from is not None and (args.resume_from / "bridge.pt").exists():
        data = torch.load(args.resume_from / "bridge.pt", map_location=device)
        model.bridge.load_state_dict(data["bridge"])
        print(f"resumed bridge from {args.resume_from}")

    print(f"trainable params: {model.trainable_param_count() / 1e6:.2f}M")

    pad_id = model.tokenizer.pad_token_id
    ds = WhisperASRDataset(
        manifest=str(args.manifest),
        tokenizer=model.tokenizer,
        feat_ext=model.feature_extractor,
        max_text_len=args.max_text_len,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate(b, pad_id=pad_id),
        num_workers=2,
        pin_memory=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / max(1, args.warmup))
    )

    wandb_run = None
    if not args.wandb_disable and os.getenv("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=f"whisper-bridge-{int(time.time())}", config=vars(args))

    step = 0
    pbar = tqdm(total=args.steps)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            features = batch["features"].to(device).to(getattr(torch, args.dtype))
            text_ids = batch["text_ids"].to(device)
            text_attn = batch["text_attn"].to(device)

            out = model(audio_features=features, text_ids=text_ids, text_attn=text_attn)
            loss = out["loss"] / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % args.log_every == 0:
                log = {"step": step, "loss": loss.item() * args.grad_accum, "lr": scheduler.get_last_lr()[0]}
                pbar.set_postfix({"loss": f"{log['loss']:.3f}"})
                if wandb_run is not None:
                    wandb_run.log(log)

            if step > 0 and step % args.ckpt_every == 0:
                model.save_pretrained(args.output / f"step_{step}")

            step += 1
            pbar.update(1)

    pbar.close()
    model.save_pretrained(args.output / "final")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"done. output: {args.output / 'final'}")


if __name__ == "__main__":
    main()
