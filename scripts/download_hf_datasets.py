"""Download HuggingFace datasets for all training stages.

Each dataset target:
  - audio_adapter (Stage 1 ASR):  librispeech, common_voice, peoples_speech,
                                  gigaspeech, mls, voxpopuli, emilia
    multilingual: pass --languages en de es ... (one output dir per language)
  - speech_translation seed:  fleurs_ast (--languages = source, --ast-target = into)
  - speech_instruct (continuation/QA): voice_assistant
  - emotion (<emo:x> recognition): crema_d, meld
  - non-speech sounds ([sound: x]): vocalsound, esc50
  - dual_stream (Stage 2):    spokenwoz, intrinsicvoice
  - text_dialog (synth seed): daily_dialog, dailydialog++

Pick what you want via --datasets librispeech common_voice spokenwoz ...

Gated datasets (gigaspeech, common_voice, emilia) need `huggingface-cli login`
plus accepting the terms on the dataset page once.

Outputs:
  - data/hf/<dataset>/raw/  audio files extracted
  - data/hf/<dataset>/manifest.jsonl  unified manifest
  - if --encode-mimi: data/hf/<dataset>/encoded/<id>.npz dual-stream samples
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Dataset adapters: each yields dicts with {audio: np.ndarray, sr: int, text: str, ...}
# ---------------------------------------------------------------------------


_LIBRISPEECH_TRAIN = [("clean", "train.100"), ("clean", "train.360"), ("other", "train.500")]


def _librispeech(splits: list[tuple[str, str]] | None = None, max_hours: float = 100) -> Iterator[dict]:
    """Full LibriSpeech is 960 h (100+360 clean, 500 other) — iterate splits
    until max_hours. 100 h is NOT enough for a Mimi-only adapter to learn
    solid ASR; use 960 (SLAM-ASR scale)."""
    from datasets import load_dataset

    total = 0
    for config, split in splits or _LIBRISPEECH_TRAIN:
        if total / 3600 >= max_hours:
            break
        try:
            ds = load_dataset("openslr/librispeech_asr", config, split=split, streaming=True)
        except Exception as e:
            print(f"[librispeech] {config}/{split} failed: {e}")
            continue
        for ex in ds:
            if total / 3600 >= max_hours:
                break
            a = ex["audio"]
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": ex["text"],
                "source": f"librispeech_{config}_{split}",
            }
            total += len(a["array"]) / a["sampling_rate"]


def _librispeech_dev(max_hours: float = 5) -> Iterator[dict]:
    """Held-out dev-clean — for the stage-1 WER gate, NEVER for training."""
    yield from _librispeech(splits=[("clean", "validation")], max_hours=min(max_hours, 5))


def _common_voice(split: str = "train", max_hours: float = 100, lang: str = "en") -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        lang,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        if not a or "array" not in a:
            continue
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": ex["sentence"],
            "source": f"common_voice_{lang}",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _peoples_speech(split: str = "train", max_hours: float = 100) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "MLCommons/peoples_speech",
        "clean",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": ex.get("text", ""),
            "source": "peoples_speech",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _gigaspeech(split: str = "train", max_hours: float = 100, subset: str = "m") -> Iterator[dict]:
    """GigaSpeech (gated — accept terms on HF first). subset: xs/s/m/l/xl."""
    from datasets import load_dataset

    ds = load_dataset("speechcolab/gigaspeech", subset, split=split, streaming=True, trust_remote_code=True)
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        text = ex.get("text", "")
        # GigaSpeech uses tag tokens for punctuation; map them back
        for tag, repl in (("<COMMA>", ","), ("<PERIOD>", "."), ("<QUESTIONMARK>", "?"), ("<EXCLAMATIONPOINT>", "!")):
            text = text.replace(f" {tag}", repl)
        if "<" in text:  # garbage/music/noise-only utterances
            continue
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": text.capitalize() if text.isupper() else text,
            "source": "gigaspeech",
        }
        total += len(a["array"]) / a["sampling_rate"]


_MLS_LANG = {
    "en": "english", "de": "german", "es": "spanish", "fr": "french",
    "it": "italian", "pt": "portuguese", "pl": "polish", "nl": "dutch",
}


def _mls(split: str = "train", max_hours: float = 100, lang: str = "en") -> Iterator[dict]:
    """Multilingual LibriSpeech (8 langs, CC-BY-4.0; en ~44k h, de ~2k h)."""
    from datasets import load_dataset

    config = _MLS_LANG.get(lang)
    if config is None:
        print(f"[mls] no config for lang={lang} (has: {sorted(_MLS_LANG)})")
        return
    ds = load_dataset("facebook/multilingual_librispeech", config, split=split, streaming=True)
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": ex.get("transcript") or ex.get("text", ""),
            "source": f"mls_{lang}",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _voxpopuli(split: str = "train", max_hours: float = 100, lang: str = "en") -> Iterator[dict]:
    """VoxPopuli (CC0, EU parliament speech — accented/spontaneous, 18 langs)."""
    from datasets import load_dataset

    ds = load_dataset("facebook/voxpopuli", lang, split=split, streaming=True, trust_remote_code=True)
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        a = ex["audio"]
        text = ex.get("normalized_text") or ex.get("raw_text", "")
        if not text.strip():
            continue
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": text,
            "source": f"voxpopuli_{lang}",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _emilia(split: str = "train", max_hours: float = 100, lang: str = "en") -> Iterator[dict]:
    """Emilia (in-the-wild speech, webdataset layout: {json, mp3} pairs).

    Languages: en, zh, de, fr, ja, ko. Gated — accept terms on
    amphion/Emilia-Dataset first. CC-BY-NC for the core set; use
    Emilia-YODAS for CC-BY.
    """
    from datasets import load_dataset

    ds = load_dataset("amphion/Emilia-Dataset", split="train", data_dir=f"Emilia/{lang.upper()}", streaming=True)
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        try:
            meta = ex["json"] if isinstance(ex.get("json"), dict) else json.loads(ex["json"])
            a = ex["mp3"]
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": meta.get("text", ""),
                "source": f"emilia_{lang}",
            }
            total += len(a["array"]) / a["sampling_rate"]
        except Exception:
            continue


_FLEURS_CONFIG = {
    "en": "en_us", "de": "de_de", "es": "es_419", "fr": "fr_fr", "it": "it_it",
    "pt": "pt_br", "nl": "nl_nl", "pl": "pl_pl", "ja": "ja_jp", "ko": "ko_kr",
    "zh": "cmn_hans_cn", "ru": "ru_ru", "hi": "hi_in", "ar": "ar_eg", "tr": "tr_tr",
}


def _fleurs_ast(split: str = "train", max_hours: float = 100, lang: str = "de", target: str = "en") -> Iterator[dict]:
    """FLEURS speech-translation pairs: speak `lang`, target text in `target`.

    FLEURS is n-way parallel (same sentences across 102 languages), so joining
    two language configs by sentence id gives speech→translation pairs.
    Small (~10 h/lang) — a seed for the interpreter task, not the main corpus.
    """
    from datasets import load_dataset

    src_cfg = _FLEURS_CONFIG.get(lang)
    tgt_cfg = _FLEURS_CONFIG.get(target)
    if not src_cfg or not tgt_cfg or lang == target:
        print(f"[fleurs_ast] unsupported pair {lang}->{target}")
        return
    tgt = load_dataset("google/fleurs", tgt_cfg, split=split)
    id2text = {ex["id"]: ex["transcription"] for ex in tgt}
    ds = load_dataset("google/fleurs", src_cfg, split=split, streaming=True)
    total = 0
    for ex in ds:
        if total / 3600 >= max_hours:
            break
        translation = id2text.get(ex["id"], "").strip()
        if not translation:
            continue
        a = ex["audio"]
        yield {
            "audio": np.asarray(a["array"], dtype=np.float32),
            "sr": a["sampling_rate"],
            "text": translation,                  # target: the TRANSLATION
            "transcript": ex.get("transcription", ""),  # source-language transcript
            "source": f"fleurs_ast_{lang}-{target}",
        }
        total += len(a["array"]) / a["sampling_rate"]


def _voice_assistant(max_samples: int = 50000) -> Iterator[dict]:
    """VoiceAssistant-400K (gpt-omni) — spoken question + text answer.

    This is *instruct* data: `text` is the assistant ANSWER, `transcript`
    the spoken question. Train audio→answer for Voxtral-style cross-modal
    continuation instead of pure ASR repetition.
    """
    from datasets import load_dataset

    ds = load_dataset("gpt-omni/VoiceAssistant-400K", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            a = ex.get("question_audio") or ex.get("audio")
            answer = ex.get("answer", "")
            if a is None or not answer.strip():
                continue
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": answer,
                "transcript": ex.get("question", ""),
                "source": "voice_assistant",
            }
        except Exception:
            continue


def _load_first(candidates: list[tuple], **kw):
    """Try multiple (repo_id, config) candidates; return first that loads."""
    from datasets import load_dataset

    last = None
    for repo, config in candidates:
        try:
            a = (repo, config) if config else (repo,)
            return load_dataset(*a, **kw)
        except Exception as e:
            last = e
    raise RuntimeError(f"none of {candidates} loaded: {last}")


_CREMA_EMO = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad",
              "anger": "angry", "disgust": "disgusted", "fear": "fearful", "happy": "happy", "neutral": "neutral", "sad": "sad"}
_MELD_EMO = {"anger": "angry", "disgust": "disgusted", "fear": "fearful", "joy": "happy",
             "neutral": "neutral", "sadness": "sad", "surprise": "surprised"}


def _crema_d(max_samples: int = 8000) -> Iterator[dict]:
    """CREMA-D — acted emotional speech (6 emotions). For <emo:x> input recognition."""
    ds = _load_first([("myleslinder/crema-d", None), ("confit/cremad", None)], split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            a = ex["audio"]
            label = ex.get("label")
            if isinstance(label, int):  # ClassLabel int — order matches _CREMA_EMO keys is NOT guaranteed; skip ints
                emo = None
            else:
                emo = _CREMA_EMO.get(str(label).upper()) or _CREMA_EMO.get(str(label).lower())
            if emo is None:
                continue
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": ex.get("sentence", "") or ex.get("text", ""),
                "emotion": emo,
                "source": "crema_d",
            }
        except Exception:
            continue


def _meld(max_samples: int = 10000) -> Iterator[dict]:
    """MELD — multi-party dialog utterances (Friends) with emotion labels."""
    ds = _load_first(
        [("zrr1999/MELD_Text_Audio", None), ("ajyy/MELD_audio", "MELD_Audio")],
        split="train",
        streaming=True,
    )
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            a = ex["audio"]
            emo = _MELD_EMO.get(str(ex.get("emotion", "")).lower())
            text = ex.get("text", "") or ex.get("utterance", "")
            if emo is None or not text:
                continue
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": text,
                "emotion": emo,
                "source": "meld",
            }
        except Exception:
            continue


def _vocalsound(max_samples: int = 10000) -> Iterator[dict]:
    """VocalSound — laughter/sigh/cough/throat-clearing/sneeze/sniff.

    Teaches the monologue to NOTICE non-speech human sounds in the user
    stream ("[sound: laughter]")."""
    ds = _load_first(
        [("flozi00/VocalSound_audio_16k", None), ("MichaelR207/vocalsound", None)],
        split="train",
        streaming=True,
    )
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            a = ex["audio"]
            label = str(ex.get("label", "")).replace("_", " ").strip()
            if not label:
                continue
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": f"[sound: {label}]",
                "source": "vocalsound",
            }
        except Exception:
            continue


def _esc50(max_samples: int = 2000) -> Iterator[dict]:
    """ESC-50 — environmental sounds (dog, siren, rain, ...) for sound awareness."""
    ds = _load_first([("ashraq/esc50", None), ("confit/esc50", None)], split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            a = ex["audio"]
            cat = str(ex.get("category", "")).replace("_", " ").strip()
            if not cat:
                continue
            yield {
                "audio": np.asarray(a["array"], dtype=np.float32),
                "sr": a["sampling_rate"],
                "text": f"[sound: {cat}]",
                "source": "esc50",
            }
        except Exception:
            continue


def _spokenwoz(split: str = "train", max_dialogs: int = 1000) -> Iterator[dict]:
    """SpokenWOZ — paired task-oriented dialog audio.

    Each example contains channel-separated user and system audio.
    """
    from datasets import load_dataset

    ds = load_dataset("Spoken-WOZ/spokenwoz", split=split, streaming=True, trust_remote_code=True)
    for i, ex in enumerate(ds):
        if i >= max_dialogs:
            break
        try:
            user_a = ex.get("user_audio") or ex.get("audio_user")
            sys_a = ex.get("system_audio") or ex.get("audio_system")
            if user_a is None or sys_a is None:
                continue
            yield {
                "user_audio": np.asarray(user_a["array"], dtype=np.float32),
                "asst_audio": np.asarray(sys_a["array"], dtype=np.float32),
                "sr": user_a["sampling_rate"],
                "dialog_id": ex.get("dialog_id", str(i)),
                "source": "spokenwoz",
            }
        except Exception:
            continue


def _intrinsicvoice(max_samples: int = 5000) -> Iterator[dict]:
    """IntrinsicVoice-500k synth speech-to-speech pairs."""
    from datasets import load_dataset

    try:
        ds = load_dataset("OpenS2S/IntrinsicVoice-500k", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        return
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        try:
            user_a = ex.get("user_audio") or ex.get("input_audio")
            asst_a = ex.get("assistant_audio") or ex.get("output_audio")
            if user_a is None or asst_a is None:
                continue
            yield {
                "user_audio": np.asarray(user_a["array"], dtype=np.float32),
                "asst_audio": np.asarray(asst_a["array"], dtype=np.float32),
                "sr": user_a["sampling_rate"],
                "source": "intrinsicvoice",
            }
        except Exception:
            continue


def _daily_dialog(max_samples: int = 5000) -> Iterator[dict]:
    """DailyDialog — text-only, but we use scripts as seed for TTS rendering."""
    from datasets import load_dataset

    ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        yield {
            "turns": ex["dialog"],
            "source": "daily_dialog",
        }


DATASETS: dict[str, Callable] = {
    "librispeech": _librispeech,
    "librispeech_dev": _librispeech_dev,
    "common_voice": _common_voice,
    "peoples_speech": _peoples_speech,
    "gigaspeech": _gigaspeech,
    "mls": _mls,
    "voxpopuli": _voxpopuli,
    "emilia": _emilia,
    "voice_assistant": _voice_assistant,
    "fleurs_ast": _fleurs_ast,
    "crema_d": _crema_d,
    "meld": _meld,
    "vocalsound": _vocalsound,
    "esc50": _esc50,
    "spokenwoz": _spokenwoz,
    "intrinsicvoice": _intrinsicvoice,
    "daily_dialog": _daily_dialog,
}

ADAPTER_DATASETS = {"librispeech", "common_voice", "peoples_speech", "gigaspeech", "mls", "voxpopuli", "emilia", "fleurs_ast"}
# held-out eval sets: same manifest format, but NEVER merged into the
# combined training manifest
EVAL_DATASETS = {"librispeech_dev"}
INSTRUCT_DATASETS = {"voice_assistant", "crema_d", "meld", "vocalsound", "esc50"}
PAIRED_DATASETS = {"spokenwoz", "intrinsicvoice"}
TEXT_DATASETS = {"daily_dialog"}
# datasets that take a `lang` kwarg and get one output dir per language
LANG_DATASETS = {"common_voice", "mls", "voxpopuli", "emilia", "fleurs_ast"}


def save_adapter(out: Path, items: Iterator[dict]) -> None:
    """Save single-stream (audio, transcript) for Stage 1."""
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)
    manifest = out / "manifest.jsonl"
    with manifest.open("w") as f:
        for i, item in enumerate(tqdm(items, desc=f"writing {out.name}")):
            wav_path = audio_dir / f"{i:08d}.wav"
            sf.write(wav_path, item["audio"], item["sr"])
            text = item.get("text", "")
            if item.get("emotion"):
                # bake the emotion into the text target — the model learns
                # recognition without any pipeline changes ("<emo:angry> ...")
                text = f"<emo:{item['emotion']}> {text}".strip()
            row = {
                "audio": str(wav_path),
                "text": text,
                "duration": len(item["audio"]) / item["sr"],
                "source": item.get("source", ""),
            }
            if item.get("emotion"):
                row["emotion"] = item["emotion"]
            if item.get("transcript"):  # instruct data: text=answer, transcript=spoken question
                row["transcript"] = item["transcript"]
            f.write(json.dumps(row) + "\n")


def save_paired(out: Path, items: Iterator[dict], mimi, device: str, echo_bleed_prob: float = 0.0) -> None:
    """Save dual-stream samples (encoded with Mimi if mimi given).

    echo_bleed_prob: fraction of samples that get residual-echo augmentation
    (attenuated delayed asst audio mixed into the user channel)."""
    import random as _random

    import soundfile as sf

    from voiceai.training.data.mixing import encode_dual_stream, save_dual_stream_sample

    rng = _random.Random(0)

    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    enc_dir = out / "encoded"
    enc_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, item in enumerate(tqdm(items, desc=f"paired {out.name}")):
        sid = f"hf_{out.name}_{i:08d}"
        u = item["user_audio"]
        a = item["asst_audio"]
        if len(u) == 0 or len(a) == 0:
            continue
        sf.write(raw_dir / f"{sid}_user.wav", u, item["sr"])
        sf.write(raw_dir / f"{sid}_asst.wav", a, item["sr"])
        meta = {
            "sample_id": sid,
            "duration_s": float(max(len(u), len(a)) / item["sr"]),
            "source": item.get("source", out.name),
            "category": f"hf_{out.name}",
        }
        metas.append(meta)
        if mimi is not None:
            try:
                bleed = rng.uniform(0.05, 0.2) if rng.random() < echo_bleed_prob else 0.0
                u_codes, a_codes = encode_dual_stream(
                    u, a, mimi, sr=item["sr"], device=device, echo_bleed=bleed
                )
                save_dual_stream_sample(
                    user_codes=u_codes,
                    asst_codes=a_codes,
                    text_ids=np.array([], dtype=np.int32),
                    text_align=np.array([], dtype=np.int32),
                    aux={"source": item.get("source"), "category": meta["category"]},
                    sample_id=sid,
                    out_root=enc_dir,
                    duration_s=meta["duration_s"],
                )
            except Exception as e:
                print(f"encode fail {sid}: {e}")
    (out / "samples.jsonl").write_text("\n".join(json.dumps(m) for m in metas))


def save_text_seed(out: Path, items: Iterator[dict]) -> None:
    """Save text dialog seeds for later TTS rendering by gen_diverse_dialogs."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scripts.jsonl").open("w") as f:
        for item in tqdm(items, desc=f"text {out.name}"):
            turns = item.get("turns", [])
            if not turns:
                continue
            structured = [
                {"role": "user" if i % 2 == 0 else "assistant", "text": t.strip()}
                for i, t in enumerate(turns)
            ]
            f.write(json.dumps({"title": "daily_dialog", "turns": structured, "source": "daily_dialog"}) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/hf"))
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["librispeech", "common_voice"],
        choices=list(DATASETS.keys()),
    )
    p.add_argument("--max-hours", type=float, default=100.0, help="for adapter datasets (per language)")
    p.add_argument("--max-samples", type=int, default=5000, help="for paired/text datasets")
    p.add_argument("--languages", nargs="+", default=["en"],
                   help="for multilingual datasets (common_voice/mls/voxpopuli/emilia/fleurs_ast); "
                        "e.g. --languages en de es")
    p.add_argument("--ast-target", default="en",
                   help="fleurs_ast: translate INTO this language")
    p.add_argument("--encode-mimi", action="store_true")
    p.add_argument("--echo-bleed-prob", type=float, default=0.3,
                   help="fraction of paired samples that get residual-echo augmentation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--combine-adapter-manifest", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    mimi = None
    if args.encode_mimi and any(d in PAIRED_DATASETS for d in args.datasets):
        from voiceai.model.mimi_utils import load_mimi

        mimi = load_mimi(device=args.device, dtype=torch.bfloat16)

    for name in args.datasets:
        langs = args.languages if name in LANG_DATASETS else [None]
        for lang in langs:
            out = args.out / (f"{name}_{lang}" if lang else name)
            if (out / "manifest.jsonl").exists() or (out / "samples.jsonl").exists() or (out / "scripts.jsonl").exists():
                print(f"[{out.name}] already exists, skipping")
                continue
            fn = DATASETS[name]
            kw = {}
            if lang:
                kw["lang"] = lang
            if name == "fleurs_ast":
                kw["target"] = args.ast_target
            try:
                if name in ADAPTER_DATASETS:
                    save_adapter(out, fn(max_hours=args.max_hours, **kw))
                elif name in EVAL_DATASETS:
                    save_adapter(out, fn(max_hours=5, **kw))
                elif name in INSTRUCT_DATASETS:
                    save_adapter(out, fn(max_samples=args.max_samples, **kw))
                elif name in PAIRED_DATASETS:
                    save_paired(out, fn(max_samples=args.max_samples), mimi, args.device,
                                echo_bleed_prob=args.echo_bleed_prob)
                elif name in TEXT_DATASETS:
                    save_text_seed(out, fn(max_samples=args.max_samples))
            except Exception as e:
                print(f"[{out.name}] failed: {e}")

    if args.combine_adapter_manifest:
        combined = args.out / "adapter_manifest.jsonl"
        # exact expected dir names — prefix matching would leak eval sets
        # like librispeech_dev into the training manifest
        expected: set[str] = set()
        for name in ADAPTER_DATASETS & set(args.datasets):
            if name in LANG_DATASETS:
                expected.update(f"{name}_{lang}" for lang in args.languages)
            else:
                expected.add(name)
        n = 0
        with combined.open("w") as out_f:
            for sub in sorted(args.out.iterdir()):
                if not sub.is_dir() or sub.name not in expected:
                    continue
                mpath = sub / "manifest.jsonl"
                if not mpath.exists():
                    continue
                with mpath.open() as in_f:
                    for line in in_f:
                        out_f.write(line)
                        n += 1
        print(f"combined adapter manifest: {n} entries → {combined}")


if __name__ == "__main__":
    main()
