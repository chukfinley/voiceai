"""LoRA SFT script.

Phase 1 training: tune Qwen3-Omni (or any backbone) on time-tokens + visual
events + barge-in via LoRA only. Cheap, fast iteration.

Run:
    uv run python -m voiceai.training.lora_sft \\
        --backbone Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --data data/interaction_sft.jsonl \\
        --output runs/lora_v1

Hardware: 1× H100 80GB or 2× A100 40GB. ~24h for 10k samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.loaders import TimeAwareSFTDataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Lazy: torch/transformers/peft only at runtime so the module can be
    # imported on a machine without them (for tests / introspection).
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    # Register special tokens used by our 200ms frame format so the model can
    # learn them as atomic units. Token strings come from data/format.py.
    new_tokens = _interaction_special_tokens()
    n_added = tokenizer.add_tokens(new_tokens, special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if n_added:
        model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset → token IDs
    ds = list(TimeAwareSFTDataset(args.data))
    encoded = [_encode_sample(toks, tokenizer, args.max_seq_len) for toks in ds]

    train_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=args.save_every,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=encoded,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output / "final"))


def _interaction_special_tokens() -> list[str]:
    """Atomic tokens for the 200ms frame format.

    Note: <t:N>, <u:audio:...>, <a:audio:...> are NOT added as single tokens
    because their bodies vary. We rely on the tokenizer to BPE-split those.
    Only stable control tokens get vocab slots.
    """
    return [
        "<silent>",
        "<u:silent>",
        "<a:silent>",
        "<barge>",
        "<thinking>",
        "<ack>",
        "<background_query>",
        "</background_query>",
        "<bg_result>",
        "</bg_result>",
        "<visual:",
        "<wait:",
    ]


def _encode_sample(token_strs: list[str], tokenizer, max_len: int) -> dict:
    text = " ".join(token_strs)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v[0] for k, v in enc.items()}
    enc["labels"] = enc["input_ids"].clone()
    # mask pad in loss
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        enc["labels"][enc["labels"] == pad_id] = -100
    return enc


if __name__ == "__main__":
    main()
