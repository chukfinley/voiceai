# Training: Recherche-Stand (Juni 2026) + Ideen fürs „richtige" Training

Quelle: Online-Recherche zu Speech-LLM-Bridges (Ultravox, Voxtral, SLAM-ASR,
Canary-Qwen, Qwen3-Omni, Granite-Speech) + Training-Effizienz 2025/26.
Was davon schon im Code ist, steht unten unter „Umgesetzt".

## 1. Architektur-Rezept (Konsens 2025/26)

- **Frame-Stacking + MLP ist das Produktions-Rezept.** Whisper gibt 50 Hz aus;
  4–8 Frames concatenieren → 10–12.5 Hz Audio-Tokens. Voxtral hat 12.5 Hz als
  Sweet Spot abliert (+1.5% QA-Acc vs 50 Hz; 6.25 Hz kostet >1% WER).
  Q-Former/Transformer-Projektoren bringen für ASR nichts — Linear/MLP gewinnt.
- **Encoder einfrieren, solange Daten klein.** SLAM-ASR-Ablation: Whisper-Encoder
  mit nur 960 h Daten unfreezen ist katastrophal (WER 4.33→12.79). Kimi-Audio-Regel:
  erst nach ~20% der Trainings-Tokens unfreezen.
- **LLM: frozen reicht für SOTA-ASR** (Canary-Qwen-2.5B = Platz 1 OpenASR-Leaderboard
  mit frozen Qwen3-1.7B + LoRA). LoRA r=64 auf q,v ist der Standard-Mittelweg für
  Instruction-Fähigkeit.
- Typische LRs: Projector-only ~1e-4 (bis 2e-3), Warmup ~1k Steps / 3%, Decay.
  Sobald LLM-Gewichte trainieren: runter auf ~2e-5.

## 2. Daten-Mix: die wichtigste Zahl zum Kopieren

**Voxtral 50/50-Mix:** 50% Audio→Transkript-Wiederholung (ASR) + 50% Cross-Modal
Continuation (Audio→Fortsetzung/Antwort). Nur-ASR killt QA-Fähigkeit, nur-Continuation
gibt ~60% WER. Beides mischen.

Praktisch für uns:
1. ASR-Manifeste wie bisher (text = Transkript).
2. Continuation-Manifeste: gleiche Audios, aber `text` = LLM-generierte Fortsetzung
   des Transkripts (einmalig offline mit dem Qwen-Backbone selbst generieren —
   das ist auch Ultravox' Knowledge-Distillation-Trick: das frozen LLM kann sich
   nicht von sich selbst wegbewegen).
3. `voice_assistant`-Manifest (gesprochene Frage → Text-Antwort) ist jetzt im
   Download-Script; Manifeste lassen sich für Stage 1 kombinieren
   (`--combine-adapter-manifest`).

## 3. Datasets (Download-Script-Support: ✅ = implementiert)

| Dataset | Stunden | HF-ID | Lizenz | |
|---|---|---|---|---|
| LibriSpeech | 960 | openslr/librispeech_asr | CC-BY-4.0 | ✅ |
| Common Voice 17 | ~2.6k en | mozilla-foundation/common_voice_17_0 | CC0 | ✅ |
| People's Speech | ~30k | MLCommons/peoples_speech | CC-BY | ✅ |
| GigaSpeech | 10k | speechcolab/gigaspeech | non-commercial, gated | ✅ |
| MLS (en) | 44k | facebook/multilingual_librispeech | CC-BY-4.0 | ✅ |
| VoxPopuli (en) | 1.8k | facebook/voxpopuli | CC0 | ✅ |
| Emilia | 101k–216k | amphion/Emilia-Dataset | CC-BY-NC (YODAS-Teil CC-BY) | ✅ |
| VoiceAssistant-400K | 470k Samples | gpt-omni/VoiceAssistant-400K | instruct | ✅ |
| Granary (NVIDIA) | ~1M h, 25 Sprachen | nvidia/Granary | open | später (multilingual) |
| YODAS2 | ~400k | espnet/yodas2 | CC-BY-3.0-Teil | später |

Skalierungs-Referenz: SLAM-ASR erreicht 2.6% WER mit nur LibriSpeech-960 und
frozen everything — fürs PoC reichen 100–1000 h locker.

## 4. Effizienz: was wirklich was bringt (Reihenfolge = ROI)

1. **Padding/Längen** (umgesetzt: Trimming + Bucketing + Stacking). War der größte
   Posten: vorher 1500 LLM-Positionen pro Sample egal wie kurz der Clip; jetzt
   ~⌈Clip-Sekunden·12.5⌉. LibriSpeech-Schnitt ~12 s → ~150 statt 1500 Tokens
   Audio-Präfix = grob 10× weniger Attention-Arbeit im Prefix.
2. **Liger-Kernel** (`pip install liger-kernel`, dann
   `apply_liger_kernel_to_qwen2()` vor Model-Load): +20% Durchsatz, −60% Speicher,
   patcht RMSNorm/RoPE/SwiGLU/CE im Qwen. Lohnt ab Stage 2 (LoRA), bei frozen LLM
   weniger kritisch. Alternative für die CE: **Cut Cross-Entropy** (Apple) — bei
  Qwens 151k-Vokabular ist die Logit-Matrix der größte Speicherfresser, CCE macht
   daraus ~1 MB. Für uns aktuell unkritisch (T_text ≤ 121), wird relevant bei
   längeren Targets (Continuation-Daten!).
3. **Sequence Packing** (FA2 varlen + position_ids-Resets): bis 2× Durchsatz.
   Für unser Embedding-Level-Packing ~100 Zeilen Custom-Collator. Nächster
   logischer Schritt nach Bucketing, wenn GPU-Auslastung immer noch <90%.
4. **Mel-Features vorberechnen** (oder gleich frozen-Whisper-Encoder-Outputs
   cachen → Stage 1 wird reines LLM-Training, Whisper fliegt komplett aus dem
   Loop). Lohnt ab mehreren Epochen über demselben Datensatz.
5. **torch.compile**: ~1.1–1.4×, aber Recompile-Stürme bei variablen Shapes —
   erst nach Bucketing-Stabilisierung, Backbone separat kompilieren.
6. **fp8**: nur H100, bei 0.5–8B-Modellen kaum Gewinn. Skip.
7. **Muon/muP**: Pretraining-Story, nicht Finetuning. Skip. **WSD-Schedule**
   (warmup-stable-decay) wäre die Alternative zu Cosine, wenn man Trainings-Horizont
   offen halten will (`lr_scheduler_type="warmup_stable_decay"` in HF).

## 5. Ideen: was man dem Modell noch beibringen könnte (wenn richtig trainiert)

Sortiert nach Aufwand/Nutzen:

1. **Continuation/KD-Stage (Ultravox-Stil)** — kein neues Datenformat nötig:
   Transkripte durchs eigene Qwen jagen, Fortsetzungen als Targets. Macht aus
   dem ASR-Bridge ein „verstehendes" Modell, ohne das LLM anzufassen.
2. **Speech-Instruct-SFT** mit VoiceAssistant-400K (+ TTS-synthetisierte
   OpenHermes-Subsets — Vorsicht: Code/Tabellen vorher rausfiltern, die sind
   nicht TTS-bar). LoRA r=64 auf q,v, LR 2e-5, 10–20% ASR-Daten im Mix behalten
   gegen Regression.
3. **Mehrsprachigkeit**: Whisper-Encoder kann es schon; nur Daten nötig
   (MLS de/fr/es, Common Voice de, Granary). Fürs Produktions-Ziel (Roadmap)
   der billigste Weg zu Deutsch.
4. **Translation (AST)**: CoVoST-2 / Granary-AST-Hälfte; gleiche Pipeline,
   anderes Target. Stabilisiert laut Ultravox auch ASR.
5. **Emotion/Paralinguistik-Tags**: Targets wie `[lacht] ja klar [ironisch]` —
   braucht Datensätze wie MELD/IEMOCAP oder pseudo-gelabelte Emilia-Clips.
   Passt zur Produktions-Roadmap (Emotion).
6. **Timestamps/Diarization-Tokens**: `<ts:3.2>`-Tokens ins Target — ermöglicht
   wort-genaue Alignments fürs Barge-in-Verhalten der Orchestrator-Schicht.
7. **Encoder-Upgrade-Pfad**: whisper-small → large-v3-turbo-Encoder (gleicher
   Encoder wie large-v3, Ultravox' Wahl). Für English-only wäre NVIDIA
   FastConformer (Canary) das stärkste, aber Whisper bleibt die multilinguale Wahl.
8. **Audio-Output wieder andocken**: der Mimi-Pfad (stage1_adapter/stage2) bleibt
   der Weg zu Speech-out. Moderne Alternative: separater Speech-Decoder à la
   LLaMA-Omni (LLM-Hidden → CTC/AR-Decoder → Codec) statt Output-Heads direkt.
9. **In-Context-Spracherwerb** (neue Sprache live verstehen, ohne Training).
   Muss VOR dem großen Training in den Datenmix — nachrüsten ist teuer:
   - **IPA-Bootstrapping**: zusätzlicher Task „transkribiere phonetisch" über
     viele Sprachen → fremde Sprache wird zu Text-Symbolen, ab da übernimmt
     Qwens Text-ICL (MTOB/Kalamang beweist Machbarkeit auf Text-Seite:
     Grammatik+Wörterbuch im Kontext reicht zum Übersetzen).
   - **Meta-Learning-Episoden**: Trainings-Samples mit wenigen
     Audio+Übersetzung-Paaren einer ZURÜCKGEHALTENEN Sprache im Kontext,
     Target = neue Äußerung übersetzen. ~40 Sprachen trainieren, ~10 nur in
     Episoden → „aus Beispielen erschließen" wird gelernter Skill.
   - Orchestrator-Hack: in-session gelernte Vokabeln als TEXT-Notiz im
     Kontext pinnen (überlebt das Sliding Window, kostet kaum Tokens).

## 6. Ziel-Pipeline „perfektes Modell" (volles Budget, Mimi-Pfad kanonisch)

Whisper-Bridge ist degradiert zu Bootstrap/Eval. Der echte Pfad (Emotion-fähig,
Audio-in UND Audio-out):

```
Audio (User) ──Mimi-Encoder──► Codes [8,T@12.5Hz] ──AudioAdapter──► Qwen
                                                                     │
Qwen-Hidden ──MimiDepthTransformer──► Codes [8,T] ──Mimi-Decoder──► Audio (Asst)
             (+ lm_head → Inner Monologue Text mit <emo:x>/<speaker>-Tags)
```

Architektur-Fixes eingebaut (siehe §8): Depth-Transformer statt unabhängiger
Heads, Acoustic-Delay-Pattern, Emotion-Token-Vokabular, Speaker-ID im
Orchestrator.

**Stages mit echtem Budget** (Referenz: Moshi 7M h, wir skalieren runter):
1. **Stage 1** — Adapter + Depth-Transformer auf ASR+TTS, backbone frozen.
   Daten: Emilia (100-200k h) + MLS + GigaSpeech + Emotion-Sets (CREMA-D, MELD)
   + VocalSound/ESC-50 (Sound-Awareness). ~50-100k Steps, 8×H100, Tage.
2. **Stage 2** — Dual-Stream-LoRA (oder Full-FT bei Budget) auf echten Dialogen:
   Fisher/CANDOR-artig + synthetische gen_*-Szenarien + SpokenWOZ. Acoustic-Delay 1.
3. **Stage 3** — Capabilities (Zeit/Barge/Background/Emotion-Reaktion).
4. **Stage 4 (neu, Budget)** — Preference-Tuning (DPO auf Dialog-Qualität,
   Voxtral macht's vor) + Multilingual-Erweiterung (Granary).

Kostenrahmen grob: Stage 1-3 auf 8×H100 ≈ 2-4 Wochen ≈ 15-40k$; Moshi-Klasse
(from scratch, Mio. Stunden) ≈ 7-stellig. Dazwischen skaliert's linear mit Daten.

## 7. Whisper-Bridge: ENTFERNT

Whisper-Encoder ist ASR-trainiert → wirft Prosodie/Emotion/Sprecher weitgehend
weg, und es gibt keinen Audio-Output-Pfad. Entscheidung: komplett raus
(whisper_lm.py, stage1_whisper.py, whisper_asr_quality.py gelöscht), Mimi ist
der einzige Pfad. Die SOTA-Hybrid-Idee (Whisper-Features + Mimi-Codes parallel,
à la Kimi-Audio) bleibt dokumentiert, falls Stage-1-ASR-Qualität mit Mimi-only
enttäuscht — dann aber als *Zusatz*-Feature neben Mimi, nie als Ersatz.

## 8. Umgesetzt in diesem Durchgang

- `whisper_lm.py`: Frame-Stacking (`stack_factor`, Default 4 → 12.5 Hz),
  Audio-Präfix-Trimming + Maske (`audio_frames`), `attn_implementation`-Option,
  Alt-Checkpoint-Kompatibilität (`stack_factor=1` für alte bridge.pt).
- `stage1_whisper.py`: Worker-Sharding-Fix (vorher sah jeder DataLoader-Worker
  den vollen Datensatz → Samples doppelt), Duration-Bucketing, Warmup+Cosine-Decay,
  fused AdamW, TF32, `--grad-checkpointing`, Multi-Manifest-Mixing,
  `--max-audio-s`-Filter.
- `stage1_adapter.py`: Cosine-Decay, fused AdamW, TF32.
- `dual_stream.py`: **Bugfix** — Labels waren nicht geshiftet (Modell hätte
  Frame t aus Frame t „vorhergesagt" = Identität gelernt) und Padding bekam
  Label 0 statt -100. Jetzt echtes Next-Frame-Target mit Maskierung.
- `download_hf_datasets.py`: + gigaspeech, mls, voxpopuli, emilia,
  voice_assistant (Instruct).

Zweiter Durchgang (Pipeline-Umbau auf „perfektes Modell"):
- `audio_adapter.py`: **MimiDepthTransformer** (Moshi/RQ-Stil) — Codebook k
  wird konditioniert auf Codebooks <k desselben Frames gesampelt. Fixt die
  Audio-Artefakte der unabhängigen Linear-Heads. Default an
  (`use_depth_transformer=True`), alte Checkpoints laden als Legacy-Heads.
- `dual_stream.py`: **Acoustic-Delay-Pattern** (`apply_acoustic_delay`,
  Stage-2/3-Flag `--acoustic-delay`, Default 1) — Semantik-Codebook führt,
  Akustik folgt versetzt.
- `voiceai_lm.py`: Emotion-Token-Vokabular (15 `<emo:x>` + `<u:emo>`/`<a:emo>`/
  `<speaker>`-Wrapper) im Tokenizer.
- `download_hf_datasets.py`: + crema_d, meld (Emotion → `<emo:x>`-Tag direkt
  im Text-Target), vocalsound, esc50 (`[sound: x]`-Targets).
- `orchestrator/speaker_id.py`: ECAPA-Voice-Print-Registry (enroll/identify/
  persist) — Sprecher-Erkennung als System-Schicht, nicht im Modell.

Dritter Durchgang („alles fertig bauen"):
- **Whisper-Bridge komplett entfernt** (whisper_lm.py, stage1_whisper.py,
  whisper_asr_quality.py, Tests). Mimi ist der einzige Input-Pfad.
  LR-Schedule nach `training/sched.py` umgezogen.
- **Asst-Stream als Model-Input** (`audio_in_asst`, cfg `asst_audio_input`):
  das Modell hört jetzt, was es selbst gesagt hat (Moshi-Standard — fehlte!).
  Stage 2/3 füttern `asst_audio_codes`; alte Stage-1-Checkpoints warm-starten
  den neuen Adapter aus `audio_in`.
- **`inference/streaming.py`**: Real-time-Engine — 80ms-Frame-Loop mit
  KV-Cache, Depth-Transformer-Sampling, Inner-Monologue-Feedback,
  Acoustic-Delay-Entzerrung vorm Mimi-Decode, Sliding-Window-Re-Prefill
  (Kontext-Deckel graceful statt Crash), `mute()` für Barge-in.
  KV-Cache == Full-Forward per Test bewiesen.
- **Speaker-ID verdrahtet**: `SpeakerIDLoop` (VAD-segmentiert, emittiert
  `SPEAKER_ID`-Events, `only_listen_to`-Gate für „rede nur mit mir").
- **`scripts/gen_volume_dynamics.py`**: Schreien/Flüstern-Augmentation mit
  `<emo:shouting>`/`<emo:whispering>`-Targets.
- **`eval/emotion_recognition.py`**: Accuracy des ersten `<emo:x>`-Tokens
  auf gelabelten Manifests. **`scripts/bench_streaming.py`**: p50/p95/p99
  ms/Frame gegen das 80ms-Budget (inkl. Re-Prefill-Spike).

Selbst-Hören-Problem (Lautsprecher → Mikro), drei Verteidigungs-Schichten:
- **AEC gefixt** (`orchestrator/aec.py`): Endlos-Schleifen-Bug (Loop hat
  eigene republizierte Events re-prozessiert), WebRTC-10ms-Re-Chunking war
  nur Kommentar — jetzt implementiert, sample-genauer Ref-Ringpuffer mit
  Output-Delay-Kompensation statt 1-Chunk-pro-Chunk-Naivität.
- **Architektur**: Modell hört sich über den eigenen `audio_in_asst`-Kanal —
  AEC muss nur den User-Kanal sauber halten, nicht erklären was gesagt wurde.
- **`add_echo_bleed`** (`mixing.py`): Trainings-Augmentation — gedämpftes,
  verzögertes Asst-Audio in den User-Kanal (Default 30% der Paired-Samples,
  `--echo-bleed-prob`). Modell lernt: leise verzögerte Kopie der eigenen
  Stimme = Echo, kein Barge-in.

---

## §N — Speaker-Adaptation auf die eigene Stimme (Idee, später)

Personalisierte ASR: Modell gezielt auf eine Zielstimme adaptieren, damit es
diesen Sprecher deutlich besser versteht.

Pipeline:
1. ~30–60 min Zielstimme aufnehmen, phonetisch vielfältig (Vorlese-Sätze),
   100–500 kurze Clips.
2. **Whisper-large-v3** transkribiert → Pseudo-Labels (~2% WER auf clean
   speech). Whisper nur als Label-Werkzeug, nie zur Laufzeit (wie die TTS-Tools).
3. Mensch prüft/korrigiert Transkripte → saubere Labels.
4. Fine-tune des Audio-Adapters via `stage1_adapter --resume-from <basis-ckpt>`.

Wichtig — catastrophic forgetting vermeiden:
- **NICHT pur** auf die Zielstimme (sonst vergisst Modell den Rest).
- Besser: warm-start vom generischen Stage-1-Modell + **gemischtes** Training
  (Zielstimme + etwas LibriSpeech). Bleibt generell, gebogen auf den Sprecher.
- Reiner Tune nur für reine "versteht-mich"-Demo, akzeptiert generischen Verlust.

Warum stark: schon ~30 min Zielstimme biegen den kleinen Adapter massiv —
WER auf dem Sprecher fällt viel mehr als durch generisches Mehr-Training,
weil Mikro/Akzent/Stimmfarbe gelernt werden.

Reihenfolge: erst generischer Basislauf (Stage 1), DANN Adaptation drauf.
Status: vorgemerkt, noch nicht gebaut. Whisper als datagen-Extra hinzufügen.
