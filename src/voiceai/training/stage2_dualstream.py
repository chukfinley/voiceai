"""Stage 2: Dual-stream conversational SFT.

Builds on Stage 1 weights. Adds:
  - LoRA on backbone for new dual-stream behavior
  - user_audio_out head (we also predict the user stream — Moshi-style)
  - Trains on paired-dialog (Fisher, CANDOR, synth) Mimi-encoded samples

Data format: see voiceai.training.data.dual_stream.DualStreamDataset

Runtime: ~5 days on RTX 3090 24GB, ~70h on Colab Pro+ A100 40GB.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", required=True, type=Path, help="path to stage1 final checkpoint")
    p.add_argument("--data-root", required=True, type=Path, help="dir of .npz dual-stream samples")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--pad-frames", type=int, default=512)
    p.add_argument("--acoustic-delay", type=int, default=1,
                   help="Moshi acoustic-delay pattern: shift Mimi codebooks 1..7 "
                        "by N frames vs the semantic codebook (0 = off)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action="store_true")
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

    model = VoiceAILM.from_pretrained(args.stage1)
    model.unfreeze_backbone()
    model = model.to(device)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.backbone = get_peft_model(model.backbone, lora_cfg)
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

        wandb_run = wandb.init(project=args.wandb_project, name=f"stage2-{int(time.time())}", config=vars(args))

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
                ckpt = args.output / f"step_{step}"
                model.save_pretrained(ckpt)

            step += 1
            pbar.update(1)

    pbar.close()
    model.save_pretrained(args.output / "final")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"stage2 done. output: {args.output / 'final'}")


if __name__ == "__main__":
    main()
