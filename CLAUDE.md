# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VOSK 2.0 is a **Vosk ASR Biasing Evaluation System** that experiments with constraining the Vosk speech recognizer to prefer domain-specific vocabulary (bias words), then measures whether this improves recognition of those words.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

`requirements.txt` now includes: `vosk soundfile numpy datasets jiwer pandas`.

The Vosk model must be downloaded and placed at `model/vosk-model-small-en-us-0.15/`. An `output/` directory is needed for generated files.

## Running the Pipeline

Run steps in order:

```bash
# Step 1: Baseline transcription (no bias)
python scripts/transcribe.py

# Step 2: Biased transcription (constrained to bias_words.txt)
python scripts/transcribe_with_bias.py

# Step 3: Error analysis — WER/CER vs ground truth
python scripts/error_analysis.py

# Step 4: Bias word accuracy on baseline predictions
python scripts/bias_word_analysis.py

# Step 5: Compare baseline vs biased recall on bias keywords
python scripts/recall_comparision.py
```

## Architecture

### Pipeline Flow

```
audio/test_audio/*.wav  ──────────────────────────────────────┐
bias_words.txt          ──────────────────────────────────────┤
model/vosk-model-*/     ──────────────────────────────────────┤
                                                               ↓
                        transcribe.py → output/predictions.json
                        transcribe_with_bias.py → output/predictions_with_bias.json
                                                               ↓
audio/ground_truth/84-121123.trans.txt ────────────────────────┤
                                                               ↓
                        error_analysis.py → output/error_metrics.csv
                        bias_word_analysis.py → stdout
                        recall_comparision.py → stdout
```

### Key Design Decisions

- **Biasing mechanism**: `transcribe_with_bias.py` calls `rec.SetGrammar(json.dumps(bias_words))` on the Kaldi recognizer, which constrains the decoder to recognize only the listed words. This is a hard constraint — predictions outside the grammar are suppressed.
- **Ground truth format**: `84-121123.trans.txt` uses the format `audioID sentence` (space-separated, ID has no extension). The analysis scripts map these to `.wav` filenames by appending `.wav`.
- **Metrics**: WER/CER via `jiwer`; bias keyword recall computed manually by set intersection across ground truth and predictions.

### Hardcoded Paths

All scripts use relative paths assuming execution from the repo root:
- `MODEL_PATH = "../model/vosk-model-small-en-us-0.15"` (relative to `scripts/`)
- Audio input: `../audio/test_audio/`
- Ground truth: `../audio/ground_truth/84-121123.trans.txt`
- Output: `../output/`

---

## Finance Domain Pipeline (Earnings-22)

A second, improved pipeline demonstrating vocabulary biasing on real earnings-call audio.

### Domain & Dataset

- **Domain**: Finance (earnings calls)  — rich in acronyms (EBITDA, CAPEX, non-GAAP) that Vosk's general LM consistently fails on
- **Dataset**: [`distil-whisper/earnings22`](https://huggingface.co/datasets/distil-whisper/earnings22) — CC-BY-SA 4.0, 119 h of real English earnings calls chunked into ~20-second segments, already 16 kHz

### Running the Finance Pipeline

```bash
# Step 1: Download 150 audio chunks + ground truth from Hugging Face
python scripts/download_earnings22.py

# Step 2: Baseline Vosk transcription (no bias)
python scripts/transcribe_finance.py

# Step 3: Post-processing vocabulary bias
python scripts/transcribe_finance_biased.py

# Step 4: WER / CER / KRR comparison table + example pairs
python scripts/evaluate_finance.py
```

### Finance Pipeline Architecture

```
HuggingFace distil-whisper/earnings22  ─────────────────────────────┐
                                                                      ↓
                         download_earnings22.py
                           audio/finance/*.wav  (PCM-16, 16 kHz, mono)
                           audio/finance_gt/ground_truth.json
                                                                      ↓
model/vosk-model-small-en-us-0.15/  ────────────────────────────────┤
                                                                      ↓
                         transcribe_finance.py
                           output/finance/predictions_baseline.json
                                                                      ↓
bias_words_finance.txt  ─────────────────────────────────────────────┤
                                                                      ↓
                         transcribe_finance_biased.py
                           output/finance/predictions_biased.json
                                                                      ↓
                         evaluate_finance.py
                           stdout: WER / CER / KRR table + examples
                           output/finance/evaluation_results.csv
```

### Biasing Mechanism (why SetGrammar was abandoned)

`SetGrammar()` hard-constrains the decoder to only emit listed words, which destroys fluency for natural speech (see existing pipeline results). The finance pipeline uses a two-layer **post-processing** approach instead:

1. **Confusion-map substitution** — regex replaces known Vosk misrecognitions of finance acronyms (e.g. `"e bit da"` → `"ebitda"`, `"cap ex"` → `"capex"`, `"a creative"` → `"accretive"`).
2. **Fuzzy phonetic matching** — `difflib.SequenceMatcher` replaces non-function words whose string similarity to a bias term exceeds 0.88.

This preserves fluent general transcription while recovering domain vocabulary.

### Key Files

| File | Purpose |
|------|---------|
| `bias_words_finance.txt` | 30 finance terms used for biasing and KRR |
| `scripts/download_earnings22.py` | Download + preprocess dataset |
| `scripts/transcribe_finance.py` | Baseline Vosk (no bias) |
| `scripts/transcribe_finance_biased.py` | Post-processing bias |
| `scripts/evaluate_finance.py` | WER / CER / KRR evaluation |
