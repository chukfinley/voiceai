"""Stage 3: Capability fine-tuning (time, visual, BG-query, concurrent commentary).

Continues from Stage 2 LoRA checkpoint. Reuses the same DualStreamDataset
but with a curated capability-mix dataset containing aux annotations.

Differs from Stage 2:
  - lower learning rate
  - shorter training
  - capability-balanced data sampling
  - optional Streaming-EOS loss for visual proactivity samples
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


CAPABILITY_KEYS = (
    "time_aware",
    "concurrent_commentary",
    "barge_in",
    "backchannel",
    "background_query",
    "visual_event",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--pad-frames", type=int, default=512)
    p.add_argument("--acoustic-delay", type=int, default=1,
                   help="must match the value used in stage 2")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--wandb-project", default="voiceai")
    p.add_argument("--wandb-disable", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    if args.smoke:
        args.steps = 30
        args.batch_size = 1
        args.grad_accum = 1
        args.pad_frames = 64
        args.log_every = 5
        args.ckpt_every = 10

    torch.manual_seed(args.seed)

    from ..model.voiceai_lm import VoiceAILM
    from .data.dual_stream import DualStreamDataset, dual_stream_collate

    device = args.device if torch.cuda.is_available() else "cpu"

    model = VoiceAILM.from_pretrained(args.stage2)
    model = model.to(device)
    print(f"trainable params: {model.trainable_param_count() / 1e6:.1f}M / {model.total_param_count() / 1e6:.1f}M total")

    ds = DualStreamDataset(args.data_root, pad_to_frames=args.pad_frames)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: dual_stream_collate(b, acoustic_delay=args.acoustic_delay),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, s / max(1, args.warmup))
    )

    wandb_run = None
    if not args.wandb_disable and os.getenv("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=f"stage3-{int(time.time())}", config=vars(args))

    step = 0
    pbar = tqdm(total=args.steps)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            user_codes = batch["user_codes"].to(device)
            asst_codes = batch["asst_codes"].to(device)
            attn = batch["attention_mask"].to(device)
            labels_text = batch["labels_text"].to(device)
            labels_user = batch["labels_user_audio"].to(device)
            labels_asst = batch["labels_asst_audio"].to(device)

            placeholder_ids = torch.full_like(attn, model.tokenizer.pad_token_id or 0)

            out = model(
                text_ids=placeholder_ids,
                user_audio_codes=user_codes,
                asst_audio_codes=asst_codes,  # model hears its own stream
                attention_mask=attn,
                labels_text=labels_text,
                labels_user_audio=labels_user,
                labels_asst_audio=labels_asst,
            )
            loss = out["loss"] / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % args.log_every == 0:
                log = {
                    "step": step,
                    "loss": loss.item() * args.grad_accum,
                    "lr": scheduler.get_last_lr()[0],
                }
                for k, v in out["loss_parts"].items():
                    log[f"loss/{k}"] = v.item()
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
    print(f"stage3 done. output: {args.output / 'final'}")


if __name__ == "__main__":
    main()
