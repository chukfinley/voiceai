"""Stage 1: Audio adapter pretraining.

What we train:
  - audio_in (AudioAdapter)
  - asst_audio_out (MimiOutputHeads)
  - user_audio_out (MimiOutputHeads) if cfg.train_user_audio

What we freeze:
  - Qwen3.5-0.8B backbone (all of it)
  - Mimi codec encoder/decoder

What we teach:
  - ASR: audio -> text  (uses backbone's existing lm_head, no LoRA needed)
  - TTS: text -> audio  (uses our new MimiOutputHeads)

Runtime: ~3 days on RTX 3090 24GB, ~40h on Colab Pro+ A100 40GB.

Usage:
    uv run python -m voiceai.training.stage1_adapter \\
        --manifest data/manifests/librispeech_cv_en.jsonl \\
        --output runs/stage1 \\
        --steps 30000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backbone", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--max-audio-s", type=float, default=20.0)
    p.add_argument("--mix-asr", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="voiceai")
    p.add_argument("--wandb-disable", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run for testing")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (only safe when manifest has pre-encoded `codes`)")
    p.add_argument("--resume-from", type=Path, default=None, help="Path to a previous ckpt dir; loads adapters.pt to warm-start.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    if args.smoke:
        args.steps = 20
        args.batch_size = 2
        args.grad_accum = 1
        args.log_every = 5
        args.ckpt_every = 10

    torch.manual_seed(args.seed)

    from ..model.mimi_utils import load_mimi
    from ..model.voiceai_lm import VoiceAIConfig, VoiceAILM
    from .data.asr_tts import ASRTTSDataset, asr_tts_collate

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    cfg = VoiceAIConfig(
        backbone=args.backbone,
        freeze_backbone=True,
        train_text=True,
        train_asst_audio=True,
        train_user_audio=False,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    model = VoiceAILM(cfg).to(device)

    if args.resume_from is not None and (args.resume_from / "adapters.pt").exists():
        adapters = torch.load(args.resume_from / "adapters.pt", map_location=device)
        model.audio_in.load_state_dict(adapters["audio_in"])
        model.asst_audio_out.load_state_dict(adapters["asst_audio_out"])
        if adapters.get("user_audio_out") and model.user_audio_out is not None:
            model.user_audio_out.load_state_dict(adapters["user_audio_out"])
        print(f"resumed adapters from {args.resume_from}")

    mimi = None if args.num_workers > 0 else load_mimi(device=device, dtype=dtype)

    print(f"trainable params: {model.trainable_param_count() / 1e6:.1f}M / {model.total_param_count() / 1e6:.1f}M total")

    ds = ASRTTSDataset(
        manifest_path=str(args.manifest),
        tokenizer=model.tokenizer,
        mimi=mimi,
        mix=(args.mix_asr, 1 - args.mix_asr),
        max_audio_s=args.max_audio_s,
        seed=args.seed,
    )
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    pad_id = model.tokenizer.pad_token_id
    # If manifest has `codes` field (pre-encoded), can use multiple workers safely.
    use_workers = bool(args.num_workers)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: asr_tts_collate(b, pad_token_id=pad_id),
        num_workers=args.num_workers if use_workers else 0,
        pin_memory=True,
        persistent_workers=use_workers,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / max(1, args.warmup))
    )

    wandb_run = None
    if not args.wandb_disable and os.getenv("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=f"stage1-{int(time.time())}", config=vars(args))

    step = 0
    pbar = tqdm(total=args.steps)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            audio_codes = batch["audio_codes"].to(device)
            text_ids = batch["text_ids"].to(device)
            tasks = batch["tasks"]

            asr_idx = [i for i, t in enumerate(tasks) if t == "asr"]
            tts_idx = [i for i, t in enumerate(tasks) if t == "tts"]

            losses: list[torch.Tensor] = []

            if asr_idx:
                ac = audio_codes[asr_idx]  # [B, K, T_audio]
                tt = text_ids[asr_idx]      # [B, T_text]
                B = ac.shape[0]
                T_audio = ac.shape[2]
                T_text = tt.shape[1]

                # Prefix-LM ASR: input = [audio_embeds, text_embeds[:-1]]
                # logits at audio-end predict text[0]; at text[i-1] predict text[i].
                text_embed_layer = model.backbone.get_input_embeddings()
                bb_dtype = text_embed_layer.weight.dtype
                audio_e = model.audio_in(ac).to(dtype=bb_dtype)  # [B, T_audio, D]
                if T_text > 1:
                    text_in = text_embed_layer(tt[:, :-1])  # [B, T_text-1, D]
                    embeds = torch.cat([audio_e, text_in], dim=1)
                    text_mask = (tt[:, :-1] != pad_id).long()
                else:
                    embeds = audio_e
                    text_mask = torch.zeros(B, 0, device=device, dtype=torch.long)
                audio_mask = torch.ones(B, T_audio, device=device, dtype=torch.long)
                attn = torch.cat([audio_mask, text_mask], dim=1)

                bb_out = model.backbone(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_all = bb_out.hidden_states[-1]
                # Slice positions [T_audio-1 .. T_audio-1+T_text) to predict tt[0..T_text)
                text_hidden = hidden_all[:, T_audio - 1:T_audio - 1 + T_text, :]
                text_logits = model.backbone.lm_head(text_hidden)  # [B, T_text, V]

                target = tt.clone()
                target[tt == pad_id] = -100
                asr_loss = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.size(-1)).float(),
                    target.reshape(-1),
                    ignore_index=-100,
                )
                losses.append(asr_loss)

            if tts_idx:
                tt = text_ids[tts_idx]      # [B, T_text]
                ac = audio_codes[tts_idx]   # [B, K, T_audio]
                B = tt.shape[0]
                T_text = tt.shape[1]
                T_audio = ac.shape[2]
                K = ac.shape[1]

                # Prefix-LM TTS: input = [text_embeds, audio_in(audio[:-1])]
                # logits at text-end predict audio[0]; at audio[i-1] predict audio[i].
                text_embed_layer = model.backbone.get_input_embeddings()
                bb_dtype = text_embed_layer.weight.dtype
                text_e = text_embed_layer(tt)  # [B, T_text, D]
                if T_audio > 1:
                    audio_in = model.audio_in(ac[:, :, :-1]).to(dtype=bb_dtype)  # [B, T_audio-1, D]
                    embeds = torch.cat([text_e, audio_in], dim=1)
                    audio_mask = torch.ones(B, T_audio - 1, device=device, dtype=torch.long)
                else:
                    embeds = text_e
                    audio_mask = torch.zeros(B, 0, device=device, dtype=torch.long)
                text_mask = (tt != pad_id).long()
                attn = torch.cat([text_mask, audio_mask], dim=1)

                bb_out = model.backbone(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_all = bb_out.hidden_states[-1]
                # Slice positions [T_text-1 .. T_text-1+T_audio) to predict ac[:,:,0..T_audio)
                audio_hidden = hidden_all[:, T_text - 1:T_text - 1 + T_audio, :]
                tts_loss = model.asst_audio_out.loss(audio_hidden, ac)
                losses.append(tts_loss)

            if not losses:
                continue

            loss = sum(losses) / len(losses) / args.grad_accum
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
                ckpt = args.output / f"step_{step}"
                # Skip backbone for intermediate ckpts — it's frozen, identical
                # across saves, and otherwise blows the pod disk quota.
                model.save_pretrained(ckpt, save_backbone=False)

            step += 1
            pbar.update(1)

    pbar.close()
    model.save_pretrained(args.output / "final")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"stage1 done. output: {args.output / 'final'}")


if __name__ == "__main__":
    main()
