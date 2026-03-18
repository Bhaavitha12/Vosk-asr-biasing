import os
import json
import wave
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
AUDIO_DIR = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/audio/finance"
GT_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/audio/finance_gt/ground_truth.json"
EVAL_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/evaluation_results.csv"
OUT_DIR = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/finance"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
if not os.path.exists(GT_PATH):
    raise FileNotFoundError("❌ ground_truth.json missing")

ground_truth = json.load(open(GT_PATH))

# ---------------- ANALYSIS ----------------
results = []

for file, text in ground_truth.items():
    path = os.path.join(AUDIO_DIR, file)

    if not os.path.exists(path):
        continue

    try:
        # Duration
        wf = wave.open(path, "rb")
        duration = wf.getnframes() / wf.getframerate()

        # Load audio
        y, sr = librosa.load(path, sr=None)

        # Energy
        energy = float(np.mean(y ** 2))

        # Silence ratio
        silence = np.sum(np.abs(y) < 0.01)
        silence_ratio = float(silence / len(y))

        # Speech rate
        words = len(text.split())
        speech_rate = words / duration if duration > 0 else 0

        results.append({
            "file": file,
            "duration": duration,
            "energy": energy,
            "silence_ratio": silence_ratio,
            "speech_rate": speech_rate
        })

    except Exception as e:
        print(f"Skipping {file}: {e}")

# ---------------- SAVE AUDIO FEATURES ----------------
audio_df = pd.DataFrame(results)
audio_df.to_csv(f"{OUT_DIR}/audio_features.csv", index=False)

print("✅ Audio features saved")

# ---------------- CORRELATION WITH METRICS ----------------
if os.path.exists(EVAL_PATH):
    eval_df = pd.read_csv(EVAL_PATH)

    # Merge on file
    df = pd.merge(eval_df, audio_df, on="file", how="inner")

    corr = df.corr(numeric_only=True)

    corr.to_csv(f"{OUT_DIR}/audio_correlation.csv")

    print("\n📊 Correlation Matrix:\n", corr)

    # ---------------- VISUALIZATION ----------------
    plt.figure()
    plt.scatter(df["speech_rate"], df["wer_baseline"])
    plt.xlabel("Speech Rate")
    plt.ylabel("WER Baseline")
    plt.title("Speech Rate vs WER")
    plt.show()

    plt.figure()
    plt.scatter(df["energy"], df["wer_baseline"])
    plt.xlabel("Energy")
    plt.ylabel("WER Baseline")
    plt.title("Energy vs WER")
    plt.show()

else:
    print("⚠️ evaluation_results.csv not found → skipping correlation")

print("Audio analysis complete")