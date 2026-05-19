"""End-to-end CPU data preparation pipeline.

Designed for a server with weak CPU + lots of RAM that can run for days.
Generates the FULL Stage-2/3 training dataset without touching a GPU.

What it does:
  1. Download HF datasets (LibriSpeech, Common Voice, SpokenWOZ, ...)
  2. Generate diverse dialogs via LLM API (uses any OpenAI-compat endpoint)
  3. Render all dialogs with Kokoro TTS on CPU
  4. Generate capability data (concurrent, backchannel, time-aware, etc.)
  5. Encode everything with Mimi codec on CPU
  6. Build combined manifest ready for GPU training

Resumable: each step skips if its output already exists.
Parallel: uses multiprocessing across all CPU cores.
Memory-friendly: streams datasets, doesn't load everything at once.

Usage:
    nohup python scripts/prep_data_cpu.py \\
        --out data/full \\
        --diverse-n 3000 \\
        --provider openai \\
        > prep.log 2>&1 &

Check progress: tail -f prep.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


STEPS = [
    {
        "name": "01_hf_audio_adapter",
        "what": "download HF datasets for Stage 1 audio adapter pretrain",
        "always_run": False,
    },
    {
        "name": "02_general_dialog",
        "what": "generate general turn-taking dialog (templates, no API)",
        "always_run": False,
    },
    {
        "name": "03_concurrent_commentary",
        "what": "generate concurrent commentary (panda-counter)",
        "always_run": False,
    },
    {
        "name": "04_backchannel",
        "what": "generate backchannel samples (mhm/yeah during user speech)",
        "always_run": False,
    },
    {
        "name": "05_time_aware",
        "what": "generate time-aware (wait N / self-init) samples",
        "always_run": False,
    },
    {
        "name": "06_barge_in",
        "what": "generate barge-in samples",
        "always_run": False,
    },
    {
        "name": "07_rapid_qa",
        "what": "generate rapid-fire QA samples",
        "always_run": False,
    },
    {
        "name": "08_constraints",
        "what": "generate constraint-following samples",
        "always_run": False,
    },
    {
        "name": "09_time_limited",
        "what": "generate time-limited session samples (30s cap)",
        "always_run": False,
    },
    {
        "name": "10_sound_recognition",
        "what": "generate sound-recognition samples (ESC-50)",
        "always_run": False,
    },
    {
        "name": "11_diverse_dialogs",
        "what": "generate diverse dialogs via LLM API (33 scenario types)",
        "always_run": False,
    },
    {
        "name": "12_combine",
        "what": "build unified manifest for Stage 2 training",
        "always_run": True,
    },
]


def step_done(out: Path, step_name: str) -> bool:
    marker = out / f".{step_name}.done"
    return marker.exists()


def mark_done(out: Path, step_name: str) -> None:
    marker = out / f".{step_name}.done"
    marker.touch()


def run(cmd: list[str], log_prefix: str) -> None:
    print(f"\n=== {log_prefix} ===")
    print(f"$ {' '.join(cmd)}")
    sys.stdout.flush()
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"!! step failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)


def step_hf_adapter(args, out: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/download_hf_datasets.py",
            "--out",
            str(out / "hf"),
            "--datasets",
            "librispeech",
            "common_voice",
            "--max-hours",
            str(args.adapter_hours),
            "--combine-adapter-manifest",
        ],
        "Stage-1 adapter datasets",
    )


def gen_step(args, out: Path, script: str, sub: str, n: int) -> None:
    cmd = [
        sys.executable,
        f"scripts/{script}",
        "--out",
        str(out / sub),
        "--n",
        str(n),
    ]
    if args.encode_mimi:
        cmd.extend(["--encode-mimi", "--device", "cpu"])
    cmd.extend(["--backend", args.tts_backend])
    run(cmd, sub)


def gen_diverse(args, out: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/gen_diverse_dialogs.py",
        "--out",
        str(out / "diverse_dialogs"),
        "--n",
        str(args.diverse_n),
        "--provider",
        args.provider,
        "--tts-backend",
        args.tts_backend,
        "--concurrency",
        str(args.api_concurrency),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.encode_mimi:
        cmd.extend(["--encode-mimi", "--device", "cpu"])
    run(cmd, "diverse_dialogs")


def step_combine(args, out: Path) -> None:
    combined = out / "stage2_combined"
    combined.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in out.iterdir():
        if not sub.is_dir():
            continue
        enc = sub / "encoded"
        if not enc.is_dir():
            continue
        for f in list(enc.glob("*.npz")) + list(enc.glob("*.json")):
            target = combined / f.name
            if not target.exists():
                target.symlink_to(f.resolve())
                n += 1
    print(f"combined {n} files into {combined}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/full"))
    p.add_argument("--provider", default="openai", help="LLM provider for diverse dialogs")
    p.add_argument("--model", default=None, help="LLM model override")
    p.add_argument("--tts-backend", default="kokoro", choices=["kokoro", "melotts", "gtts"])
    p.add_argument("--diverse-n", type=int, default=3000)
    p.add_argument("--general-n", type=int, default=800)
    p.add_argument("--concurrent-n", type=int, default=600)
    p.add_argument("--backchannel-n", type=int, default=500)
    p.add_argument("--time-aware-n", type=int, default=500)
    p.add_argument("--barge-n", type=int, default=300)
    p.add_argument("--rapid-n", type=int, default=300)
    p.add_argument("--constraints-n", type=int, default=300)
    p.add_argument("--time-limited-n", type=int, default=300)
    p.add_argument("--sound-n", type=int, default=200)
    p.add_argument("--adapter-hours", type=float, default=100.0)
    p.add_argument("--api-concurrency", type=int, default=8)
    p.add_argument("--encode-mimi", action="store_true", default=True)
    p.add_argument("--skip", nargs="*", default=[], help="step names to skip")
    p.add_argument("--only", nargs="*", default=[], help="run only these steps")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    def maybe_run(step_name: str, fn) -> None:
        if step_name in args.skip:
            print(f"skip {step_name}: explicit --skip")
            return
        if args.only and step_name not in args.only:
            return
        if step_done(args.out, step_name):
            print(f"skip {step_name}: already done")
            return
        ts = time.monotonic()
        fn()
        elapsed = time.monotonic() - ts
        print(f"=> {step_name} done in {elapsed:.0f}s")
        mark_done(args.out, step_name)

    maybe_run("01_hf_audio_adapter", lambda: step_hf_adapter(args, args.out))
    maybe_run(
        "02_general_dialog",
        lambda: gen_step(args, args.out, "gen_general_dialog.py", "general_dialog", args.general_n),
    )
    maybe_run(
        "03_concurrent_commentary",
        lambda: gen_step(args, args.out, "gen_concurrent_commentary.py", "concurrent", args.concurrent_n),
    )
    maybe_run(
        "04_backchannel",
        lambda: gen_step(args, args.out, "gen_backchannel.py", "backchannel", args.backchannel_n),
    )
    maybe_run(
        "05_time_aware",
        lambda: gen_step(args, args.out, "gen_time_aware_audio.py", "time_aware", args.time_aware_n),
    )
    maybe_run(
        "06_barge_in",
        lambda: gen_step(args, args.out, "gen_barge_in.py", "barge_in", args.barge_n),
    )
    maybe_run(
        "07_rapid_qa",
        lambda: gen_step(args, args.out, "gen_rapid_qa.py", "rapid_qa", args.rapid_n),
    )
    maybe_run(
        "08_constraints",
        lambda: gen_step(args, args.out, "gen_constraints.py", "constraints", args.constraints_n),
    )
    maybe_run(
        "09_time_limited",
        lambda: gen_step(args, args.out, "gen_time_limited.py", "time_limited", args.time_limited_n),
    )
    maybe_run(
        "10_sound_recognition",
        lambda: gen_step(args, args.out, "gen_sound_recognition.py", "sound_recognition", args.sound_n),
    )
    maybe_run("11_diverse_dialogs", lambda: gen_diverse(args, args.out))
    maybe_run("12_combine", lambda: step_combine(args, args.out))

    total = time.monotonic() - t0
    print(f"\nALL DONE in {total / 3600:.1f}h")
    print(f"Output: {args.out}/stage2_combined")
    print(f"Stage 1 adapter manifest: {args.out}/hf/adapter_manifest.jsonl")


if __name__ == "__main__":
    main()
