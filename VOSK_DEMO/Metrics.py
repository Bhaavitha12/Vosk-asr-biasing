import json
import os
import re
from jiwer import wer, cer

# ---------------- PATHS ----------------
GT_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/audio/finance_gt/ground_truth.json"
BASE_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/predictions_baseline.json"
BIAS_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/predictions_biased.json"
KEYWORDS_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/biaswords_earnings22.txt"

# ---------------- LOAD DATA ----------------
def load_json(path):
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_keywords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

gt = load_json(GT_PATH)
baseline = load_json(BASE_PATH)
biased = load_json(BIAS_PATH)
keywords = load_keywords(KEYWORDS_PATH)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# ---------------- METRIC FUNCTION ----------------
def compute_all_metrics(pred_dict, model_name):
    all_gt = []
    all_pred = []

    TP = 0  # correct keyword matches
    FP = 0  # predicted but not in GT
    FN = 0  # missed keywords

    total_gt_keywords = 0

    for file, gt_text in gt.items():
        pred_text = pred_dict.get(file, "")

        gt_clean = clean_text(gt_text)
        pred_clean = clean_text(pred_text)

        all_gt.append(gt_clean)
        all_pred.append(pred_clean)

        for kw in keywords:
            gt_present = kw in gt_clean
            pred_present = kw in pred_clean

            if gt_present:
                total_gt_keywords += 1

            if gt_present and pred_present:
                TP += 1
            elif pred_present and not gt_present:
                FP += 1
            elif gt_present and not pred_present:
                FN += 1

    # ---------------- WER / CER ----------------
    gt_full = " ".join(all_gt)
    pred_full = " ".join(all_pred)

    overall_wer = wer(gt_full, pred_full)
    overall_cer = cer(gt_full, pred_full)

    # ---------------- KEYWORD METRICS ----------------
    recall = TP / total_gt_keywords if total_gt_keywords > 0 else 0  # KRR
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    ker = (FP + FN) / total_gt_keywords if total_gt_keywords > 0 else 0

    # ---------------- PRINT ----------------
    print(f"\n===== {model_name} =====")
    print(f"WER : {overall_wer:.4f}")
    print(f"CER : {overall_cer:.4f}")
    print(f"KRR (Recall) : {recall:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"KER : {ker:.4f}")

    return overall_wer, overall_cer, recall, precision, f1, ker


# ---------------- RUN ----------------
print("\n📊 FULL METRIC EVALUATION")

if baseline:
    compute_all_metrics(baseline, "BASELINE")

if biased:
    compute_all_metrics(biased, "BIASED")

print("\nDone")
