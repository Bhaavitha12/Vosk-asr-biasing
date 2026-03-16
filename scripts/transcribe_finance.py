"""
Stage 2 – Baseline Vosk transcription on finance audio (no bias).

Reads every WAV in audio/finance/, runs Vosk with no grammar constraint,
and writes predictions to output/finance/predictions_baseline.json.

Run from repo root:
    python scripts/transcribe_finance.py
"""

import os
import json
import wave
from vosk import Model, KaldiRecognizer


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/vosk-model-small-en-us-0.15"
AUDIO_DIR  = "audio/finance"
OUT_FILE   = "output/finance/predictions_baseline.json"
CHUNK      = 4000   # frames per read — same as existing pipeline
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── Load model once (expensive) ───────────────────────────────────────────────
print("Loading Vosk model...")
model = Model(MODEL_PATH)

# ── Transcribe each WAV file ──────────────────────────────────────────────────
predictions = {}
audio_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
print(f"Transcribing {len(audio_files)} files...")

for idx, fname in enumerate(audio_files):
    fpath = os.path.join(AUDIO_DIR, fname)

    with wave.open(fpath, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)      # enable word-level output

        # Stream audio in fixed-size chunks; accumulate partial results
        text_parts = []
        while True:
            data = wf.readframes(CHUNK)
            if not data:
                break
            if rec.AcceptWaveform(data):
                # Partial utterance complete — grab intermediate text
                text_parts.append(json.loads(rec.Result()).get("text", ""))

        # Flush any remaining frames at end of file
        text_parts.append(json.loads(rec.FinalResult()).get("text", ""))

    predictions[fname] = " ".join(text_parts).strip()

    if (idx + 1) % 50 == 0:
        print(f"  {idx + 1}/{len(audio_files)} done")

# ── Persist predictions ───────────────────────────────────────────────────────
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print(f"\nBaseline transcription complete -> {OUT_FILE}")
