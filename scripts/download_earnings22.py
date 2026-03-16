"""
Stage 1 – Download the Earnings-22 finance speech corpus.

Source  : distil-whisper/earnings22 (CC-BY-SA 4.0)
          119-hour corpus of English earnings calls, ~20-second chunks.
Output  :
  audio/finance/*.wav              — PCM-16, 16 kHz, mono (Vosk-ready)
  audio/finance_gt/ground_truth.json — {filename: lowercase transcript}

Run from repo root:
    python scripts/download_earnings22.py
"""

import io
import os
import json
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio


# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES = 150                          # number of ~20-sec chunks to fetch
AUDIO_DIR = "audio/finance"
GT_FILE   = "audio/finance_gt/ground_truth.json"
TARGET_SR = 16_000                       # Vosk requires 16 kHz
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GT_FILE), exist_ok=True)

# ── Stream chunked test split from Hugging Face (avoids full download) ────────
print(f"Streaming Earnings-22 (chunked), first {N_SAMPLES} samples...")
ds = load_dataset(
    "distil-whisper/earnings22",
    name="chunked",
    split="test",
    streaming=True,          # fetch on-the-fly — no full corpus download
)
# Disable automatic decoding so we get raw audio bytes; decode with soundfile
# instead (avoids torchcodec / FFmpeg dependency on Windows).
ds = ds.cast_column("audio", Audio(decode=False))

# ── Convert each sample to WAV and collect ground-truth ──────────────────────
ground_truth = {}
for i, sample in enumerate(ds.take(N_SAMPLES)):
    raw = sample["audio"]
    # raw["bytes"] contains the encoded audio; decode with soundfile
    array, sr = sf.read(io.BytesIO(raw["bytes"]))

    # Flatten to mono if stereo
    if array.ndim > 1:
        array = array.mean(axis=1)

    # Resample to 16 kHz if needed (dataset is typically already 16 kHz)
    if sr != TARGET_SR:
        # Use numpy linear interpolation for a lightweight resample
        duration   = len(array) / sr
        n_out      = int(duration * TARGET_SR)
        array      = np.interp(
            np.linspace(0, len(array) - 1, n_out),
            np.arange(len(array)),
            array,
        )

    array = array.astype(np.float32)

    # Write 16-bit PCM mono WAV — the exact format Vosk expects
    fname = f"earnings22_{i:04d}.wav"
    sf.write(os.path.join(AUDIO_DIR, fname), array, TARGET_SR, subtype="PCM_16")

    # Store lowercased transcript keyed by filename
    ground_truth[fname] = sample["transcription"].strip().lower()

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{N_SAMPLES} files saved")

# ── Persist ground truth ──────────────────────────────────────────────────────
with open(GT_FILE, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print(f"\nDone.\n  Audio  -> {AUDIO_DIR}/\n  GT     -> {GT_FILE}")
print(f"  Unique transcripts: {len(ground_truth)}")
