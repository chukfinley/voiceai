"""End-to-end smoke test on CPU + tiny stand-in model.

Builds 3 synthetic dual-stream samples, runs each training stage for 5 steps,
verifies no NaN / no crash. Total runtime ~5 minutes on a laptop.

Usage:
    uv run python scripts/smoke_test.py

Should print "ALL SMOKE TESTS PASSED" on success.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _make_dummy_samples(out: Path, n: int = 3, T: int = 24, K: int = 8) -> None:
    from voiceai.training.data.dual_stream import DualStreamSample, save_sample

    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        s = DualStreamSample(
            user_codes=np.random.randint(0, 2048, size=(K, T)),
            asst_codes=np.random.randint(0, 2048, size=(K, T)),
            text_ids=np.array([1, 2, 3], dtype=np.int32),
            text_align=np.array([2, 5, 8], dtype=np.int32),
            aux={"category": "smoke"},
            sample_id=f"smoke_{i}",
            duration_s=float(T) / 12.5,
        )
        save_sample(s, out)


def _smoke_unit_tests() -> None:
    print("[1/4] running unit tests…")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-x", "-m", "not slow"],
        capture_output=False,
    )
    if r.returncode != 0:
        raise SystemExit("unit tests failed")


def _smoke_dual_stream_dataset(tmp: Path) -> None:
    print("[2/4] dual-stream dataset roundtrip…")
    data_dir = tmp / "data"
    _make_dummy_samples(data_dir, n=4, T=32)
    from voiceai.training.data.dual_stream import DualStreamDataset, dual_stream_collate
    from torch.utils.data import DataLoader

    ds = DualStreamDataset(data_dir, pad_to_frames=32)
    loader = DataLoader(ds, batch_size=2, collate_fn=dual_stream_collate)
    for batch in loader:
        assert batch["user_codes"].shape == (2, 8, 32)
        assert batch["labels_asst_audio"].shape == (2, 8, 32)
        break


def _smoke_model_forward(tmp: Path) -> None:
    print("[3/4] tiny VoiceAILM forward + backward…")
    import torch

    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(
        backbone=TINY,
        freeze_backbone=True,
        dtype="float32",
        train_user_audio=True,
    )
    model = VoiceAILM(cfg)
    B, K, T = 1, 8, 6
    user_codes = torch.randint(0, cfg.codebook_size, (B, K, T))
    text_ids = torch.full((B, T), 0, dtype=torch.long)
    labels_audio = torch.randint(0, cfg.codebook_size, (B, K, T))
    labels_text = torch.zeros((B, T), dtype=torch.long)
    out = model(
        text_ids=text_ids,
        user_audio_codes=user_codes,
        labels_text=labels_text,
        labels_user_audio=labels_audio,
        labels_asst_audio=labels_audio,
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def _smoke_stage2_run(tmp: Path) -> None:
    print("[4/4] stage2 smoke run (5 steps, tiny model)…")
    data_dir = tmp / "data"
    out_dir = tmp / "run_stage2"
    stage1_dir = tmp / "fake_stage1"

    from voiceai.model.voiceai_lm import VoiceAIConfig, VoiceAILM

    cfg = VoiceAIConfig(backbone=TINY, freeze_backbone=True, dtype="float32")
    m = VoiceAILM(cfg)
    m.save_pretrained(stage1_dir)

    cmd = [
        sys.executable,
        "-m",
        "voiceai.training.stage2_dualstream",
        "--stage1",
        str(stage1_dir),
        "--data-root",
        str(data_dir),
        "--output",
        str(out_dir),
        "--smoke",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--lora-targets",
        "q_proj,v_proj",
        "--wandb-disable",
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("stage2 smoke failed")
    assert (out_dir / "final").exists()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if Path.cwd() != repo_root:
        print(f"chdir → {repo_root}")
        import os

        os.chdir(repo_root)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _smoke_unit_tests()
        _smoke_dual_stream_dataset(tmp)
        _smoke_model_forward(tmp)
        _smoke_stage2_run(tmp)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
