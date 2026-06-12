# PROJECT_STATE.md — Single Source of Truth

Stand: 2026-06-12. Fasst alles zusammen, was über das Projekt entschieden,
recherchiert, gemessen und gebaut wurde. Wenn ein anderes Dokument
(README, PLAN, HOW_TO_TRAIN, TRAINING_IDEAS, COMPUTE, OSS_LANDSCAPE) hiervon
abweicht, gilt dieses hier — die anderen sind älter.

---

## TL;DR — Stand bei Projekt-Pause (2026-06-12)

**Die volle Vollduplex-Pipeline läuft mechanisch end-to-end und ist live
testbar.** Audio rein → Modell → Audio + inner-monologue-Text raus, im Browser
über eine Gradio-Web-UI (`scripts/webui.py`). Das beweist: die Architektur,
das Training und die Inferenz funktionieren als Ganzes.

**Was es NICHT kann: gute Gespräche.** Der Output ist Kauderwelsch (z.B.
`etch irmahood... ~ ~ ~ 娘家`). Grund ist ausschließlich Datenmenge/Training,
kein Architektur-Defekt: nur 1500 synthetische Dialoge, kurzes Stage-2-Training,
und die Trainingssamples hatten leeren inner-monologue-Text. Das ist genau das
erwartete Ergebnis bei dieser Datengröße.

**Bewiesen:** Pipeline trainiert + spricht. **Offen:** Qualität — braucht Geld
für mehr Daten + längeres Training (siehe §11).

- **Modell:** HF `chukfinley/voiceai-duplex-demo` (adapters.pt + backbone +
  zwei Demo-Wavs). Hochgeladen 2026-06-12.
- **Demo-Audio:** `runs/duplex/assistant_response2.wav` (12.3s echtes
  Modell-Audio, RMS 0.034, kein Stille/NaN).
- **Web-UI:** `scripts/webui.py` — `uv run --extra train --extra datagen
  --with "gradio<5" --with "huggingface-hub<1.0" python scripts/webui.py`.
  Auf RunPod hinter dem HTTP-Proxy serven (Port 7860); **uvicorn statt
  `demo.launch()`**, weil gradios localhost-Self-Check im Container scheitert.
- **GPU-Hinweis:** RTX **3090** nehmen, nicht 4090 — zwei 4090-Pods (EUR-IS-2,
  Image `runpod/pytorch:0.7.0-cu1241-torch260`) bekamen nie eine Runtime
  zugewiesen; 3090 bootet mit demselben Image sauber.

**Projekt temporär pausiert**, bis Budget für richtiges Training da ist.

---

## 0. Das eine offene Problem, das alles entscheidet: Grounding

**Empirisch gemessen (2026-06-11), nicht Theorie:**

| Ansatz | Held-out WER | grounded? |
|---|---|---|
| Mimi-only prefix-LM (audio_in-Adapter → frozen Qwen) | 176% | **NEIN** |
| Mimi + LoRA auf Backbone | 209% / 299% (schlechter!) | **NEIN** |
| **Whisper-Encoder + Frame-Stacking-Bridge (SLAM-ASR)** | **10.5%** | **JA** |

Whisper-small-Encoder (frozen) → 4M-Bridge-MLP mit `downsample_k=5`
(5 Frames → 1) → frozen Qwen3-0.6B. 2000 Clips, 3000 Steps, ~9 min, ~$0.15
auf einer 4090. Train-WER 1.9%, Held-out-WER 10.5%. Modell liegt auf HF:
`chukfinley/voiceai-asr-whisper`. Branch `lora-grounding-fix`.

**Warum Mimi-only nicht groundet:** Prefix-LM + Teacher-Forcing + frozen LLM
= das LLM sagt jedes nächste Token aus den vorherigen GROUND-TRUTH-Tokens
vorher (Sprach-Prior). Audio wird nur fürs erste Token gebraucht und dann
ignoriert. Symptom: niedrige Train-Loss, aber autoregressive Ausgabe ist
flüssiger Müll ohne Bezug zum Audio. Rohe Mimi-RVQ-Codes sind außerdem eine
schwache ASR-Repräsentation. **Loss messen reicht nicht — immer Held-out-WER.**

### Der Konflikt, der daraus folgt

Dieses Gespräch hat den **Mimi-only-Pfad** ausgebaut und Whisper **entfernt** —
aus gutem Grund: Whisper ist ASR-trainiert, wirft Emotion/Prosodie/Sprecher
weg und hat **keinen Audio-Output**. Mimi behält all das und kann Audio
erzeugen. Aber die Messung sagt: **Mimi-only versteht nicht** (groundet nicht),
**Whisper versteht** (10.5% WER).

→ Das saubere Entweder-Oder hält der Realität nicht stand:

- **Whisper:** versteht ✓, Emotion ✗, Audio-Output ✗
- **Mimi:** versteht ✗ (roh), Emotion ✓, Audio-Output ✓

### Die Auflösung: Hybrid-Input (= sowieso der SOTA-Weg)

Genau das machen Kimi-Audio, GLM-4-Voice, Step-Audio: **beides parallel als
Input** — kontinuierliche Whisper-Features (fürs Verstehen/Grounding) **plus**
Mimi-Codes (für Emotion/Prosodie/Output). Nicht entweder/oder.

Konkret für uns:
- **Input-Verstehen:** Whisper-Encoder-Features + Frame-Stacking (bewiesen,
  10.5% WER) — das ist der ASR-/Semantik-Pfad.
- **Input-Paralinguistik + Output:** Mimi-Codes — Emotion erkennen, eigene
  Stimme hören, Audio erzeugen über Depth-Transformer.
- Beide Embeddings am Input summiert/concateniert ins Qwen.

**Das ist die wichtigste noch zu treffende Architektur-Entscheidung.** Alles
unten Gebaute (Depth-Transformer, Emotion-Tokens, Streaming, AEC, etc.) bleibt
gültig — es hängt an der Mimi-Seite. Was fehlt, ist die Whisper-Feature-Seite
als zweiter Input-Strang **wieder dranzubauen** (diesmal als Ergänzung neben
Mimi, nicht als Ersatz). Der gelöschte `whisper_lm.py`-Code (frame-stacking,
trimming) ist die Vorlage; in der Git-History / auf HF / Branch
`lora-grounding-fix` vorhanden.

---

## 1. Was das Projekt ist

Open-Source-Nachbau von Thinking Machines' „Interaction Model". Natives,
end-to-end Full-Duplex-Speech-LLM: hört **während** es spricht. Kein
ASR→LLM→TTS-Kaskaden-Stack, sondern zwei parallele Audio-Streams (User +
Assistent) als Mimi-Codes, Frame für Frame interleaved durchs LLM (Moshi-Stil,
12.5 Hz). Plus: zeit-bewusst, visuell-proaktiv (separater VLM-Watcher),
tool-fähig über ein Background-LLM.

**Backbone-Entscheidung:** Qwen3 (Apache 2.0). Frozen in Stage 1, LoRA ab
Stage 2. Der Grund für ein eigenes Modell statt Moshi-Reuse: Moshi = Helium-7B
festgenagelt (englisch-lastig, 2024er-Wissen, schwaches Reasoning, nicht
austauschbar). Qwen gibt uns Intelligenz + Multilingual geschenkt.

### Was das perfekt trainierte Modell könnte
Gleichzeitig hören/sprechen, unterbrechbar (Barge-in), Backchannel ("mhm"
während du redest), Zeitgefühl (Stille-Dauer, eigene Redezeit, "fass in 10s
zusammen"), Live-Kommentar zu laufenden Events, proaktiv bei visuellen Events,
Background-Reasoning mit Überbrückung ("Moment, ich schau nach"), Inner
Monologue (denkt in Text, spricht in Audio). Latenz < ~200ms, kein
Turn-Taking-Zwang.

---

## 2. Mensch-Ähnlichkeit: was fehlt (Architektur-Ebene)

Trennlinie: **Post-Training ändert Verhalten, nicht Kapazität.**

### Gebaut in diesem Gespräch (Architektur-Lücken geschlossen)
1. **Depth-Transformer** (`audio_adapter.py:MimiDepthTransformer`) — Moshi/RQ-
   Stil: Codebook k wird konditioniert auf Codebooks <k desselben Frames
   gesampelt. Fixt Audio-Artefakte der unabhängigen Linear-Heads. Default an.
2. **Acoustic-Delay** (`dual_stream.py:apply_acoustic_delay`) — semantisches
   Codebook 0 führt, akustische 1–7 versetzt. `--acoustic-delay` (Default 1).
3. **Modell hört sich selbst** (`voiceai_lm.py:audio_in_asst`) — eigener
   Adapter für den Assistant-Stream als Input. Fehlte (Moshi-Standard).
4. **Emotion-Vokabular** — 15 `<emo:x>`-Tokens + `<u:emo>`/`<a:emo>`/
   `<speaker>`/`<task>`-Wrapper. Erkennung UND Ausgabe von Emotion.
5. **Streaming-Inferenz** (`inference/streaming.py`) — 80ms-Frame-Loop, KV-
   Cache, Depth-Sampling, Monologue-Feedback, Acoustic-Delay-Entzerrung,
   Sliding-Window-Re-Prefill (Kontext-Deckel graceful), `mute()` für Barge-in.
6. **Speaker-ID** (`orchestrator/speaker_id.py`) — ECAPA-Voice-Prints,
   enroll/identify, `SpeakerIDLoop` mit `only_listen_to` ("rede nur mit mir").
7. **AEC + Echo-Bleed** (`orchestrator/aec.py`, `mixing.py:add_echo_bleed`) —
   Selbst-Hören-Problem, 3 Schichten (s.u. §7).

### Noch offen, Architektur-Decke (nicht per Post-Training)
1. **Grounding/Hybrid-Input** — §0. Die Nummer-1-Lücke.
2. **Langer Kontext** — nach ~45 min Kontextfenster voll. Sliding-Window
   federt ab (vergisst graceful), aber echtes Langzeitgedächtnis = Memory-
   Modul im Orchestrator (Text-Notizen pinnen) oder größeres LLM.
3. **Vision nativ** — aktuell VLM→Text→`<visual_event>`-Token-Umweg. Native
   Vision-Tokens wären die echte Lösung. Bewusst verschoben.
4. **Stimm-/Stil-Conditioning** — kein Style-Embedding-Input. Eine Stimme,
   eine Grundstimmung. PersonaPlex zeigt es geht, war aber NVIDIA-Großprojekt.

### Bewusst verschoben (User-Entscheidung)
Eigeninitiative/Ziele (das „Modelle sind nur Input-Output-Generatoren"-Problem
— bräuchte Ziel-Zustand + kontinuierliches Lernen), Lernen-im-Gespräch (macht
das LLM in-context), Sehen.

---

## 3. Recherche-Stand (Juni 2026) — Kern-Erkenntnisse

Voll in `TRAINING_IDEAS.md`. Die wichtigsten Zahlen:

- **Frame-Stacking + MLP** ist das Produktions-Rezept. 50 Hz → 12.5 Hz
  (Voxtral-Sweet-Spot). Q-Former bringt für ASR nichts.
- **Encoder frozen halten** bei kleinen Daten (SLAM-ASR: unfreezen mit 960h =
  WER 4.3→12.8, katastrophal). Kimi: erst nach ~20% Tokens unfreezen.
- **Voxtral 50/50-Mix**: 50% ASR-Wiederholung + 50% Cross-Modal-Continuation.
  Nur-ASR killt QA, nur-Continuation gibt 60% WER.
- **Ultravox-KD**: Targets = Fortsetzungen vom eigenen Qwen → frozen LLM kann
  nicht von sich wegdriften.
- **Frozen LLM + trainierter Projektor reicht für SOTA-ASR** (Canary-Qwen-2.5B
  = Platz 1 OpenASR mit frozen Qwen3-1.7B + LoRA). ABER: bei uns groundete
  frozen-Mimi nicht — der Unterschied ist der Encoder (FastConformer/Whisper
  vs rohe Mimi-Codes).
- **Effizienz nach ROI:** (1) Padding/Trimming/Stacking, (2) Liger-Kernel
  (+20% Durchsatz, −60% Speicher), (3) Cut-Cross-Entropy (Qwens 151k-Vokabular
  → Logit-Matrix von 24GB auf ~1MB), (4) Sequence-Packing (bis 2×), (5)
  Mel/Feature-Pre-Encoding, (6) torch.compile, (7) bf16+TF32.
- **fp8/Muon/muP/torchtune(tot)/Unsloth(kein Custom-Multimodal): skip.**
- **WSD-Schedule** statt Cosine, wenn Horizont offen bleiben soll.

---

## 4. Moshi/PersonaPlex Post-Training (Kyutai, 2026-06-10)

Blogpost: GRPO-RL-Post-Training auf fertigen Full-Duplex-Modellen, ohne
Pretraining anzufassen. 4.000h Seamless Interaction (Meta, öffentlich auf HF:
`facebook/seamless-interaction`), per VAD in 4 Achsen zerlegt: Pausen,
Turn-Taking, Backchannel, Unterbrechungen. Reward pro Achse + LLM-Judge gegen
Qualitäts-Degradation. Ergebnisse deutlich (Moshi Turn-Taking 0.74→0.96 etc.).

**Was per Post-Training in ein bestehendes Modell geht:** die ganze
Interaktions-Schicht (Turn-Taking, Pausen, Backchannel, Barge-in, Emotion-Tags,
Zeit, Background-Query, Speaker-Tags, Sound-Awareness, Echo-Robustheit).

**Was NICHT geht:** Intelligenz (LLM festgenagelt), Multilingual, Vision nativ,
langer Kontext, Stimm-Freiheit, Audio-Verständnis-Tiefe.

**Plan zweigleisig:**
- **Track A:** PersonaPlex + Kyutai-Rezept nachbauen → schnelle Demo + baut die
  RL-Infrastruktur (Rollouts, VAD-Rewards, LLM-Judge), die wir eh brauchen.
- **Track B:** unser Modell. Kyutai-Rezept wird unsere **Stage 4** (GRPO). RL-
  Infra aus Track A 1:1 wiederverwendbar, nur Modell getauscht.

`training/moshi_ft.py` existiert als Stub (NotImplementedError), `refs/moshi`
+ `refs/personaplex` liegen im Repo.

---

## 5. Multilingual / Übersetzung / Dolmetschen

Architektonisch ja — Simultandolmetschen ist die Paradedisziplin dieser
Architektur (Beweis: Kyutais **Hibiki** = Moshi-Architektur als FR→EN-Live-
Dolmetscher). Input- und Output-Sprache komplett entkoppelt (Mimi sprach-
agnostisch, Qwen3 stark in Top-10). „Deutsch rein, Spanisch raus live" geht;
im Dolmetscher-Modus ist Barge-in invertiert (User redet = Input, keine
Unterbrechung) — trainierbares Verhalten.

**Gebaut:** Download-Pipeline mehrsprachig (`--languages en de es ...`), ein
Dir pro Sprache. `fleurs_ast` (n-way-parallel → Speech→Übersetzung-Paare).
`<task>`-Tokens für Moduswechsel. CV/MLS/VoxPopuli/Emilia jetzt
sprach-parametrisiert.

**Fehlt (Daten, keine Architektur):** Skala (CoVoST-2, CVSS), Hibiki-Trick für
Simultan-Timing (parallele Texte beidseitig TTS + Wort-Alignments), sprecher-
konsistente multilinguale TTS-Stimme.

### In-Context-Spracherwerb (neue Sprache live, ohne Training)
Geht — aber als **trainierter Skill**, nicht gratis. Beweis auf Text-Seite:
MTOB/Kalamang (Grammatik+Wörterbuch im Kontext → übersetzen). Audio-Brücke
braucht: (1) **IPA-Bootstrapping** (phonetisch transkribieren über viele
Sprachen → fremde Sprache wird zu Text-Symbolen, dann Qwen-Text-ICL), (2)
**Meta-Learning-Episoden** (Trainings-Samples mit Audio+Übersetzung-Paaren
einer zurückgehaltenen Sprache im Kontext, Target = neue Äußerung übersetzen;
~40 Sprachen trainieren, ~10 nur in Episoden). Muss VOR dem großen Training in
den Datenmix. Grenze: Kontextfenster = Minuten Audio → Wörter/Phrasen, kein
fließendes Sprechen. Orchestrator-Hack: gelernte Vokabeln als Text-Notiz pinnen
(überlebt Sliding-Window, billig).

---

## 6. Hardware & Kosten

### Modellgröße (winzig): Qwen3-8B bf16 ≈ 16GB, 1.7B ≈ 3.4GB, Adapter+Depth
wenige hundert MB, Mimi 145M. Speicher geht in Daten, nie ins Modell.

### Speicher-Daten: Mimi-Codes sind 160× kleiner als WAV
~0.7 MB/h (Codes) vs ~115 MB/h (WAV). 5k h = ~4GB Codes. 200k h = ~150GB.
Strategie: streamen → encodieren → Roh-Audio (außer gated Sets + ~50-100h
Eval) wegwerfen. Codes wegwerfen = auf Mimi festgenagelt; gated Sets (Emilia,
GigaSpeech) zusätzlich als FLAC archivieren.

### Training-Dauer Consumer-Hardware
| Stufe | Hardware | Dauer |
|---|---|---|
| PoC (1.7B, ~960h en) | 1× RTX 4090 | ~10-14 Tage |
| PoC | 3× 4090 | ~4-5 Tage |
| „Gut" (4B, DE+EN, 20-50k h) | 8× 4090/5090 | 1-3 Monate |
| „Perfekt" (8B, Top-10, 100-200k h) | Consumer | 6-12 Monate → **mieten** |

**DGX Spark: Vorsicht** — 273 GB/s Bandbreite (4090: 1000, 5090: 1800). „1
PFLOP" ist fp4-Inferenz-Marketing. Fürs Training real unter einer 4090 (2-3×
langsamer). Gut zum Debuggen großer Modelle (128GB unified), schlecht zum
Trainieren. **Empfehlung: 1-2× 4090/5090 kaufen für PoC+Debugging, großes
Training mieten** (Lambda/Runpod/Voltage Park, ~$2-2.50/h H100, Spot ~$1.50).

### Kosten „perfektes 8B-Modell, komplett gemietet"
LLM selbst pretrainen wäre $1.5-2M+ (sinnlos, Qwen schenkt's). Unsere Pipeline
auf fertigem Qwen3-8B, 100-200k h:

| Posten | Kosten |
|---|---|
| Mimi-Encoding (billige GPU) | ~$0.5-1k |
| Synthese-Daten | ~$1k |
| Stage 1 (10-18B Tokens) | ~$4-6k |
| Stage 2 (Dual-Stream) | ~$1k |
| Stage 3 (Capabilities) | ~$250 |
| Stage 4 (GRPO, rollout-teuer) | ~$1.5-5k |
| LLM-Judge/API + Storage | ~$1.5-4k |

**Single-Pass: ~$10-17k. Realistisch (2-3× für Ablationen/Bugs/Retries):
$30-60k, sparsam $20-30k.** Größter Unsicherheitsfaktor: Stage-4-Rollout-
Effizienz (Faktor 5). Klug: alle Experimente auf 1.7B (~10× billiger), nur
validierte Configs auf 8B. Spot + `--resume-from`. Pilot zuerst (~$1-2k).

---

## 7. Selbst-Hören-Problem (Lautsprecher → Mikro), 3 Schichten

1. **AEC** (`aec.py`) — WebRTC, subtrahiert Lautsprecher-Signal vorm Mimi-
   Encode. Gefixt: Endlos-Schleifen-Bug (Loop re-prozessierte eigene Events),
   echtes 10ms-Re-Chunking, sample-genauer Ref-Ringpuffer mit Output-Delay-
   Kompensation.
2. **Architektur** — Modell hört sich über eigenen `audio_in_asst`-Kanal; AEC
   muss nur User-Kanal sauber halten.
3. **Training** — `add_echo_bleed`: gedämpftes verzögertes Asst-Audio in den
   User-Kanal (30% der Paired-Samples, `--echo-bleed-prob`). Modell lernt:
   leise verzögerte Kopie = Echo, kein Barge-in.

---

## 8. Trainings-Pipeline (Code-Stand)

### Ein-Befehl-Rezepte: `scripts/train_recipe.py`
```bash
uv run python scripts/train_recipe.py --recipe poc --dry-run   # Plan
uv run python scripts/train_recipe.py --recipe poc             # <€500, 1×4090
uv run python scripts/train_recipe.py --recipe full            # 8B, 10 Sprachen
```
Phasen: download → encode → stage1 → **gate1** → synth → stage2 → stage3 →
bench. Jede einzeln via `--phase` wiederholbar, Resume-Hinweis bei Fehler.

### Gate-System (Go/No-Go nach Stage 1)
`gate1` misst WER auf strikt separatem `librispeech_dev` (Leak-Schutz: exakter
Dir-Name-Match statt Prefix). **<15% super, 15-30% brauchbar, >30% STOPP** —
mehr Stage-1-Daten/Steps vor Stage-2-Geld. Max. Verlustrisiko bis erste
Messung: ~$60-80, dann auf Zahlen entscheiden statt blind.

### PoC-Rezept (<€500, ~150-250€ real)
Qwen3-1.7B, volle 960h LibriSpeech (clean-100+360 + other-500; der Downloader
konnte vorher nur 100h!) + VoiceAssistant + CREMA-D + VocalSound + ESC-50.
60k Steps Stage 1, dann gate, dann 9 synth-Szenarien + Volume-Dynamics, Stage
2 (25k) + Stage 3 (8k) + bench. Ergebnis: englische Full-Duplex-Demo, nicht
klug, nicht multilingual — beweist jede Komponente.

### Stage-Übersicht
- **Stage 1** (`stage1_adapter.py`): Audio-Adapter + Depth-Transformer auf
  ASR+TTS, Backbone frozen. **`--lora-backbone`** neu (versucht Grounding-Fix
  durch trainierbare LoRA — Messung sagt: half bei Mimi nicht, machte schlimmer;
  s. §0). Cosine-Decay, fused AdamW, TF32.
- **Stage 2** (`stage2_dualstream.py`): Dual-Stream LoRA, hört eigenen Stream,
  Acoustic-Delay.
- **Stage 3** (`stage3_capabilities.py`): Zeit/Barge/Background/Emotion.
- **Stage 4** (geplant, Infra fehlt): GRPO, Kyutai-Rezept.

### Datasets (`download_hf_datasets.py`)
ASR: librispeech (960h, multi-split), common_voice, peoples_speech, gigaspeech,
mls, voxpopuli, emilia (alle sprach-parametrisiert). Übersetzung: fleurs_ast.
Instruct: voice_assistant. Emotion: crema_d, meld (→`<emo:x>` ins Target).
Sounds: vocalsound, esc50 (→`[sound: x]`). Paired: spokenwoz, intrinsicvoice.
Eval (nie ins Training): librispeech_dev.

---

## 9. Bugs gefixt in diesem Gespräch (sonst stille Trainings-Killer)

1. **dual_stream Label-Shift** — Labels waren `codes.clone()` statt geshiftet →
   Modell hätte Identität gelernt (Frame t aus Frame t). + Padding bekam Label
   0 (echter Code!) statt -100. **Kritisch — jedes Stage-2-Training wertlos.**
2. **Worker-Duplikate** (Whisper-Stage, jetzt entfernt) — jeder DataLoader-
   Worker sah vollen Datensatz.
3. **combine-manifest Leak** — Prefix-Match hätte `librispeech_dev` ins
   Training gemischt. Jetzt exakter Name-Match.
4. **DualStreamDataset flach** — `glob` statt `rglob`, hätte synth-Szenarien
   in Unterordnern (`synth/<x>/encoded/`) nicht gefunden → 0 Samples.
5. Kein LR-Decay, kein fused AdamW, kein TF32 (alle Stages).

---

## 10. Tests
27 Tests grün (`uv run pytest tests/ -q`). Decken ab: audio_adapter (inkl.
Depth-Transformer Kausalität/Sampling/masked frames), voiceai_lm (Legacy +
Depth + Self-Hearing), dual_stream (Shift, Acoustic-Delay-Roundtrip), streaming
(KV-Cache == Full-Forward, Sliding-Window, mute), aec (Ref-Buffer, keine Endlos-
Schleife, Echo-Bleed). Smoke-Benchmark: p95 41ms < 80ms-Budget (real-time OK).

---

## 11. Nächste Schritte (Empfehlung)

1. **Grounding lösen = Hybrid-Input** (§0). Whisper-Feature-Strang neben Mimi
   wieder dranbauen. Ohne das versteht das Modell nicht — alles andere ist
   Verzierung an einem tauben Modell. **Höchste Priorität.**
2. **PoC fahren** mit Hybrid-Input + gate1. ~$60-80 bis zur ersten WER-Zahl.
3. **Echte Dialog-Daten** für Stage 2 (Seamless Interaction, Fisher/CANDOR).
4. **Stage-4-RL-Infra** (Kyutai-Rezept) — auch als Track A auf PersonaPlex.
5. Sequence-Packing + Liger/CCE wenn GPU-Auslastung <90%.

Offene Daten-Beschaffung (separates Thema, später): gated Sets, CoVoST-2,
multilinguale TTS-Stimme, Emotions-Verlauf-Dialoge.
