"""Stage 2 launcher: generates synth dual-stream data + launches training."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


GENERATORS = [
    ("scripts/gen_diverse_dialogs.py", "diverse_dialogs", 3000),
    ("scripts/gen_concurrent_commentary.py", "concurrent", 600),
    ("scripts/gen_backchannel.py", "backchannel", 500),
    ("scripts/gen_time_aware_audio.py", "time_aware", 500),
    ("scripts/gen_barge_in.py", "barge_in", 300),
    ("scripts/gen_rapid_qa.py", "rapid_qa", 300),
    ("scripts/gen_general_dialog.py", "general_dialog", 200),
]


def generate_data(out_root: Path, mult: float, tts_backend: str, device: str, llm_provider: str) -> None:
    for script, name, base_n in GENERATORS:
        n = max(10, int(base_n * mult))
        out = out_root / name
        if out.exists() and (out / "samples.jsonl").exists():
            print(f"[data] {name}: already exists, skipping")
            continue
        print(f"[data] {name}: generating {n} samples…")
        cmd = [
            sys.executable,
            script,
            "--out",
            str(out),
            "--n",
            str(n),
            "--encode-mimi",
            "--device",
            device,
        ]
        if "diverse_dialogs" in script:
            cmd.extend(["--tts-backend", tts_backend, "--provider", llm_provider])
        else:
            cmd.extend(["--backend", tts_backend])
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})


def combine_encoded(out_root: Path, combined: Path) -> None:
    combined.mkdir(parents=True, exist_ok=True)
    for script, name, _ in GENERATORS:
        src = out_root / name / "encoded"
        if not src.exists():
            continue
        for f in src.glob("*.npz"):
            target_npz = combined / f.name
            if not target_npz.exists():
                target_npz.symlink_to(f.resolve())
        for f in src.glob("*.json"):
            target_json = combined / f.name
            if not target_json.exists():
                target_json.symlink_to(f.resolve())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/stage2"))
    p.add_argument("--data-root", type=Path, default=Path("data/stage2"))
    p.add_argument("--combined", type=Path, default=Path("data/stage2/combined"))
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--mult", type=float, default=1.0, help="multiplier on data quantity")
    p.add_argument("--tts-backend", default="kokoro")
    p.add_argument("--llm-provider", default="openai", help="provider for diverse_dialogs generator")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-data", action="store_true")
    args = p.parse_args()

    if not args.skip_data:
        generate_data(args.data_root, args.mult, args.tts_backend, args.device, args.llm_provider)
        combine_encoded(args.data_root, args.combined)

    print(f"[train] starting Stage 2: stage1={args.stage1} steps={args.steps}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "voiceai.training.stage2_dualstream",
            "--stage1",
            str(args.stage1),
            "--data-root",
            str(args.combined),
            "--output",
            str(args.output),
            "--steps",
            str(args.steps),
        ],
        check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    print(f"\n✓ Stage 2 done. Output: {args.output}/final")


if __name__ == "__main__":
    main()
