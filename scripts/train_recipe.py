"""One-command training recipes: budget PoC vs full model.

Two presets:

  poc   — proof of concept for UNDER €500 on one rented RTX 4090/3090.
          Qwen3-1.7B, ~300 h English audio + synth dialogs.
          ~8-10 GPU-days ≈ €150-250 at 2026 rental prices (~€0.40/h 4090,
          incl. retries comfortably under €500).
          Result: full-duplex English demo — ASR, TTS, emotion tags,
          barge-in/backchannel/time basics. NOT smart, NOT multilingual.

  full  — the real model. Qwen3-8B, multilingual (10 langs), all datasets.
          Designed for a rented multi-GPU node (8×H100 class, ~2-4 weeks,
          see TRAINING_IDEAS.md §6 for cost math). The commands are the
          same — only data volume, backbone and step counts grow.

Each phase is resumable: rerun with --phase to redo a single step; download
phases skip existing output on their own.

Usage:
    uv run python scripts/train_recipe.py --recipe poc --dry-run   # show plan
    uv run python scripts/train_recipe.py --recipe poc             # run all
    uv run python scripts/train_recipe.py --recipe poc --phase stage1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = [sys.executable]

GEN_SCRIPTS = [
    "gen_general_dialog",
    "gen_barge_in",
    "gen_backchannel",
    "gen_rapid_qa",
    "gen_time_aware_audio",
    "gen_time_limited",
    "gen_constraints",
    "gen_sound_recognition",
    "gen_concurrent_commentary",
]


def recipe_poc(root: Path) -> dict[str, list[list[str]]]:
    data = root / "data_poc"
    runs = root / "runs_poc"
    synth = data / "synth"
    backbone = "Qwen/Qwen3-1.7B"
    return {
        "download": [
            # 960 h LibriSpeech, not 100: the original Mimi-only attempt
            # underperformed at 100-300 h — a from-scratch audio adapter needs
            # SLAM-ASR-scale data. Still free and ungated.
            PY + ["scripts/download_hf_datasets.py", "--out", str(data / "hf"),
                  "--datasets", "librispeech", "librispeech_dev",
                  "voice_assistant", "crema_d", "vocalsound", "esc50",
                  "--max-hours", "960", "--max-samples", "20000",
                  "--combine-adapter-manifest"],
        ],
        "encode": [
            PY + ["scripts/preencode_stage1.py",
                  "--manifest", str(data / "hf" / "adapter_manifest.jsonl"),
                  "--out-dir", str(data / "codes"),
                  "--out-manifest", str(data / "manifest_encoded.jsonl")],
        ],
        "stage1": [
            PY + ["-m", "voiceai.training.stage1_adapter",
                  "--manifest", str(data / "manifest_encoded.jsonl"),
                  "--output", str(runs / "stage1"),
                  "--backbone", backbone,
                  "--steps", "60000", "--batch-size", "8", "--grad-accum", "4",
                  "--num-workers", "4"],
        ],
        # GO/NO-GO: WER on held-out dev-clean. Rough guide after stage 1:
        #   <15%  great — continue
        #   15-30% usable — continue, expect rough edges
        #   >30%  STOP. More stage-1 data/steps before spending stage-2 money.
        "gate1": [
            PY + ["-m", "voiceai.eval.asr_quality",
                  "--model", str(runs / "stage1" / "final"),
                  "--manifest", str(data / "hf" / "librispeech_dev" / "manifest.jsonl"),
                  "--n", "200"],
        ],
        "synth": [
            PY + [f"scripts/{name}.py", "--out", str(synth / name.removeprefix("gen_")),
                  "--n", "800", "--encode-mimi"]
            for name in GEN_SCRIPTS
        ] + [
            # residual-echo robustness on the synth dialogs happens at encode
            # time inside the gen scripts' mixing path; volume dynamics extra:
            PY + ["scripts/gen_volume_dynamics.py",
                  "--manifest", str(data / "hf" / "adapter_manifest.jsonl"),
                  "--out", str(data / "volume_dynamics"), "--max-samples", "3000"],
        ],
        "stage2": [
            PY + ["-m", "voiceai.training.stage2_dualstream",
                  "--stage1", str(runs / "stage1" / "final"),
                  "--data-root", str(synth),
                  "--output", str(runs / "stage2"),
                  "--steps", "25000", "--batch-size", "4", "--grad-accum", "8"],
        ],
        "stage3": [
            PY + ["-m", "voiceai.training.stage3_capabilities",
                  "--stage2", str(runs / "stage2" / "final"),
                  "--data-root", str(synth),
                  "--output", str(runs / "stage3"),
                  "--steps", "8000"],
        ],
        "bench": [
            PY + ["scripts/bench_streaming.py", "--model", str(runs / "stage3" / "final"),
                  "--frames", "300"],
            PY + ["-m", "voiceai.eval.emotion_recognition",
                  "--model", str(runs / "stage3" / "final"),
                  "--manifest", str(data / "hf" / "crema_d" / "manifest.jsonl"),
                  "--max-samples", "300"],
        ],
    }


def recipe_full(root: Path) -> dict[str, list[list[str]]]:
    data = root / "data_full"
    runs = root / "runs_full"
    synth = data / "synth"
    backbone = "Qwen/Qwen3-8B"
    langs = ["en", "de", "es", "fr", "it", "pt", "ja", "ko", "zh", "ru"]
    return {
        "download": [
            PY + ["scripts/download_hf_datasets.py", "--out", str(data / "hf"),
                  "--datasets", "librispeech", "librispeech_dev", "gigaspeech",
                  "mls", "voxpopuli", "common_voice", "emilia", "fleurs_ast",
                  "voice_assistant", "crema_d", "meld", "vocalsound", "esc50",
                  "--languages", *langs,
                  "--max-hours", "20000", "--max-samples", "400000",
                  "--combine-adapter-manifest"],
            PY + ["scripts/download_hf_datasets.py", "--out", str(data / "hf"),
                  "--datasets", "spokenwoz", "intrinsicvoice",
                  "--max-samples", "50000", "--encode-mimi"],
        ],
        "encode": [
            PY + ["scripts/preencode_stage1.py",
                  "--manifest", str(data / "hf" / "adapter_manifest.jsonl"),
                  "--out-dir", str(data / "codes"),
                  "--out-manifest", str(data / "manifest_encoded.jsonl"),
                  "--max-audio-s", "30"],
        ],
        "stage1": [
            PY + ["-m", "voiceai.training.stage1_adapter",
                  "--manifest", str(data / "manifest_encoded.jsonl"),
                  "--output", str(runs / "stage1"),
                  "--backbone", backbone,
                  "--steps", "200000", "--batch-size", "16", "--grad-accum", "4",
                  "--num-workers", "8"],
        ],
        "gate1": [
            PY + ["-m", "voiceai.eval.asr_quality",
                  "--model", str(runs / "stage1" / "final"),
                  "--manifest", str(data / "hf" / "librispeech_dev" / "manifest.jsonl"),
                  "--n", "500"],
        ],
        "synth": [
            PY + [f"scripts/{name}.py", "--out", str(synth / name.removeprefix("gen_")),
                  "--n", "20000", "--encode-mimi"]
            for name in GEN_SCRIPTS
        ] + [
            PY + ["scripts/gen_volume_dynamics.py",
                  "--manifest", str(data / "hf" / "adapter_manifest.jsonl"),
                  "--out", str(data / "volume_dynamics"), "--max-samples", "50000"],
        ],
        "stage2": [
            PY + ["-m", "voiceai.training.stage2_dualstream",
                  "--stage1", str(runs / "stage1" / "final"),
                  "--data-root", str(synth),
                  "--output", str(runs / "stage2"),
                  "--steps", "150000", "--batch-size", "8", "--grad-accum", "8"],
        ],
        "stage3": [
            PY + ["-m", "voiceai.training.stage3_capabilities",
                  "--stage2", str(runs / "stage2" / "final"),
                  "--data-root", str(synth),
                  "--output", str(runs / "stage3"),
                  "--steps", "30000"],
        ],
        # stage4 (GRPO interactivity post-training, Kyutai recipe) is the
        # planned next phase — infra not built yet, see TRAINING_IDEAS.md §6.
        "bench": [
            PY + ["scripts/bench_streaming.py", "--model", str(runs / "stage3" / "final"),
                  "--frames", "500"],
            PY + ["-m", "voiceai.eval.emotion_recognition",
                  "--model", str(runs / "stage3" / "final"),
                  "--manifest", str(data / "hf" / "crema_d" / "manifest.jsonl"),
                  "--max-samples", "1000"],
        ],
    }


RECIPES = {"poc": recipe_poc, "full": recipe_full}
PHASE_ORDER = ["download", "encode", "stage1", "gate1", "synth", "stage2", "stage3", "bench"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", choices=list(RECIPES), required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--phase", choices=PHASE_ORDER, default=None,
                   help="run only this phase (default: all, in order)")
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = p.parse_args()

    plan = RECIPES[args.recipe](args.root)
    phases = [args.phase] if args.phase else PHASE_ORDER

    for phase in phases:
        cmds = plan[phase]
        print(f"\n=== [{args.recipe}] phase: {phase} ({len(cmds)} command(s)) ===")
        for cmd in cmds:
            print("  $ " + " ".join(cmd))
            if args.dry_run:
                continue
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"\nphase '{phase}' failed (exit {r.returncode}). "
                      f"Fix and resume with: --recipe {args.recipe} --phase {phase}")
                sys.exit(r.returncode)
    if args.dry_run:
        print("\n(dry run — nothing executed)")
    else:
        print(f"\nrecipe '{args.recipe}' complete.")


if __name__ == "__main__":
    main()
