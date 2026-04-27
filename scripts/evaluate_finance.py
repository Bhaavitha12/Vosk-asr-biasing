"""
Stage 4 – Evaluate baseline vs. biased finance transcriptions.

Metrics
-------
WER  — Word Error Rate       (lower is better; via jiwer)
CER  — Character Error Rate  (lower is better; via jiwer)
KRR  — Keyword Recall Rate   (higher is better; manual set-intersection)
         = fraction of bias-vocab words present in ground truth that were
           also present in the predicted transcript

KRR is the primary metric for this experiment: it measures how well the
model recovers domain-specific finance vocabulary that a general ASR
system tends to miss or mangle.

Prints: summary table + up to 8 transcript pairs where bias improved KRR.
Saves : output/finance/evaluation_results.csv

Run from repo root:
    python scripts/evaluate_finance.py
"""

import os
import re
import json
import csv
from jiwer import wer, cer


# ── Config ────────────────────────────────────────────────────────────────────
GT_FILE       = "audio/finance_gt/ground_truth.json"
BASELINE_FILE = "output/finance/predictions_baseline.json"
BIASED_FILE   = "output/finance/predictions_biased.json"
BIAS_FILE     = "bias_words_finance.txt"
OUT_CSV       = "output/finance/evaluation_results.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── Load all data ─────────────────────────────────────────────────────────────
with open(GT_FILE,       encoding="utf-8") as f: ground_truth = json.load(f)
with open(BASELINE_FILE, encoding="utf-8") as f: baseline     = json.load(f)
with open(BIASED_FILE,   encoding="utf-8") as f: biased       = json.load(f)
with open(BIAS_FILE,     encoding="utf-8") as f:
    bias_words = {
        ln.strip().lower()
        for ln in f
        if ln.strip() and not ln.startswith("#")
    }

# Only evaluate samples present in all three sources
common = sorted(set(ground_truth) & set(baseline) & set(biased))
print(f"Evaluating on {len(common)} samples...\n")

refs_list, base_list, bias_list = [], [], []
for fname in common:
    refs_list.append(ground_truth[fname].lower())
    base_list.append(baseline[fname].lower())
    bias_list.append(biased[fname].lower())


# ── WER / CER (aggregate over all samples) ───────────────────────────────────
base_wer = wer(refs_list, base_list)
bias_wer = wer(refs_list, bias_list)
base_cer = cer(refs_list, base_list)
bias_cer = cer(refs_list, bias_list)


# ── KRR — Keyword Recall Rate ─────────────────────────────────────────────────
def _words(text):
    """Extract lowercase word tokens, stripping punctuation (handles 'ebitda.' etc.)."""
    return set(re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower()))


def compute_krr(refs, preds, keywords):
    """
    For each sample, count bias words present in the reference.
    Of those, count how many also appear in the prediction.
    Returns (recall_rate, hits, total).
    """
    total, hits = 0, 0
    for ref, pred in zip(refs, preds):
        ref_words  = _words(ref)
        pred_words = _words(pred)
        for kw in keywords:
            if kw in ref_words:
                total += 1
                if kw in pred_words:
                    hits += 1
    rate = hits / total if total > 0 else 0.0
    return rate, hits, total

base_krr, base_hits, kw_total = compute_krr(refs_list, base_list, bias_words)
bias_krr, bias_hits, _        = compute_krr(refs_list, bias_list, bias_words)


# ── Print summary table ───────────────────────────────────────────────────────
def pct(v):
    return f"{v * 100:.2f}%"

def delta(before, after, lower_is_better=True):
    d = after - before
    arrow  = "v" if d < 0 else "^"
    better = (d < 0) == lower_is_better
    badge  = "[+]" if better else "[-]"
    return f"{arrow}{abs(d) * 100:.2f}pp {badge}"

W = 65
print("=" * W)
print(f"  Finance ASR Biasing Evaluation - Earnings-22 ({len(common)} samples)")
print("=" * W)
print(f"  {'Metric':<10}  {'Baseline':>12}  {'Biased':>12}  {'Change':>16}")
print("-" * W)
print(f"  {'WER':<10}  {pct(base_wer):>12}  {pct(bias_wer):>12}  {delta(base_wer, bias_wer):>16}")
print(f"  {'CER':<10}  {pct(base_cer):>12}  {pct(bias_cer):>12}  {delta(base_cer, bias_cer):>16}")
print(f"  {'KRR':<10}  {pct(base_krr):>12}  {pct(bias_krr):>12}  {delta(base_krr, bias_krr, lower_is_better=False):>16}")
print("=" * W)
print(f"\n  Keyword hits: {base_hits}/{kw_total} baseline  ->  {bias_hits}/{kw_total} biased\n")


# ── Example transcript pairs where bias improved KRR ─────────────────────────
def sample_krr(ref, pred, keywords):
    """Returns (hits, total) for one sample."""
    rw = _words(ref)
    pw = _words(pred)
    kw_present = [k for k in keywords if k in rw]
    return sum(1 for k in kw_present if k in pw), len(kw_present)

print("-" * W)
print("  Examples where bias improved keyword recall:\n")
shown = 0
for ref, base_pred, bias_pred in zip(refs_list, base_list, bias_list):
    bh, bt = sample_krr(ref, base_pred, bias_words)
    ah, at = sample_krr(ref, bias_pred, bias_words)
    if bt > 0 and ah > bh:
        kws = [kw for kw in bias_words if kw in ref.split()]
        print(f"  GT      : {ref[:110]}")
        print(f"  Baseline: {base_pred[:110]}")
        print(f"  Biased  : {bias_pred[:110]}")
        print(f"  KW      : {kws}  |  hits {bh}/{bt} -> {ah}/{at}")
        print()
        shown += 1
        if shown >= 8:
            break

if shown == 0:
    print("  (no samples found where bias improved keyword recall)\n")


# ── Save per-sample results to CSV ────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file", "wer_baseline", "wer_biased",
        "cer_baseline", "cer_biased",
        "krr_baseline", "krr_biased",
        "ground_truth", "prediction_baseline", "prediction_biased"
    ])
    writer.writeheader()
    for fname, ref, bp, bip in zip(common, refs_list, base_list, bias_list):
        bh, bt = sample_krr(ref, bp,  bias_words)
        ah, at = sample_krr(ref, bip, bias_words)
        writer.writerow({
            "file":                fname,
            "wer_baseline":        round(wer(ref, bp),  4),
            "wer_biased":          round(wer(ref, bip), 4),
            "cer_baseline":        round(cer(ref, bp),  4),
            "cer_biased":          round(cer(ref, bip), 4),
            "krr_baseline":        f"{bh}/{bt}" if bt else "0/0",
            "krr_biased":          f"{ah}/{at}" if at else "0/0",
            "ground_truth":        ref,
            "prediction_baseline": bp,
            "prediction_biased":   bip,
        })

print(f"  Per-sample CSV saved -> {OUT_CSV}")
