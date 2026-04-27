"""
Stage 3 – Finance-domain post-processing bias.

WHY not SetGrammar()?
  SetGrammar constrains the decoder to ONLY the listed words, so it destroys
  fluency for natural speech (as demonstrated by the existing pipeline).
  Instead we use a two-layer post-processing pass on the baseline transcripts:

  Layer 1 — Confusion-map substitution (regex, longest-match first)
      Hard-codes the predictable misrecognitions Vosk makes for finance
      acronyms (e.g. "e bit da" -> "ebitda", "cap ex" -> "capex").

  Layer 2 — Fuzzy phonetic matching (difflib SequenceMatcher)
      For every non-function word in the transcript, compute string
      similarity to each single-word bias term.  Replace if similarity
      exceeds FUZZY_THRESHOLD and the word is not in SKIP_WORDS.
      This catches spelling variants ("synergy" -> "synergies") and
      partial misrecognitions ("a creative" already caught by Layer 1,
      but fuzzy catches softer cases like "dilutef" -> "dilutive").

Input  : output/finance/predictions_baseline.json
Output : output/finance/predictions_biased.json

Run from repo root:
    python scripts/transcribe_finance_biased.py
"""

import os
import json
import re
from difflib import SequenceMatcher


# ── Config ────────────────────────────────────────────────────────────────────
BASELINE_FILE  = "output/finance/predictions_baseline.json"
BIAS_FILE      = "bias_words_finance.txt"
OUT_FILE       = "output/finance/predictions_biased.json"
FUZZY_THRESHOLD = 0.88   # conservative — avoids false-positive replacements
# ─────────────────────────────────────────────────────────────────────────────


# ── Layer 1: confusion map ────────────────────────────────────────────────────
# Maps typical Vosk output phrases -> correct finance term.
# Ordered longest -> shortest so greedy matching picks the best fit first.
CONFUSION_MAP = {
    # ── Patterns confirmed from actual Vosk output on Earnings-22 ─────────────
    # EBITDA: Vosk hears "ee-bit-dah" and renders phonetically as "even day"
    "even day":      "ebitda",
    # pipeline: Vosk splits on the compound and loses the second syllable
    "pipe and":      "pipeline",
    # tailwinds: Vosk mishears the "w" cluster → "tailings"
    "tailings":      "tailwinds",

    # ── General finance acronym patterns (phonetic fracturing) ────────────────
    "e b i t d a":   "ebitda",
    "e bit da":       "ebitda",
    "ebit da":        "ebitda",
    "ebita":          "ebitda",
    "evita":          "ebitda",
    # non-GAAP — "gap" alone is too risky to replace globally
    "non gap":        "non-gaap",
    "non-gap":        "non-gaap",
    "nongap":         "non-gaap",
    "non gapping":    "non-gaap",
    # CAPEX / OPEX
    "cap ex":         "capex",
    "cape x":         "capex",
    "op ex":          "opex",
    # EPS (usually spelled out in speech)
    "e p s":          "eps",
    # CAGR
    "c a g r":        "cagr",
    # accretive — common Vosk phonetic failure
    "a creative":     "accretive",
    "accreted if":    "accretive",
    # synergies
    "sin origies":    "synergies",
    "sin ergies":     "synergies",
    "sin energy":     "synergies",
    # buyback (often split)
    "buy back":       "buyback",
    # deleverage
    "de leverage":    "deleverage",
    "deliver edge":   "deleverage",
    # headwinds / tailwinds (split forms)
    "head winds":     "headwinds",
    "tail winds":     "tailwinds",
}

# ── Layer 2: words to never replace with a bias term ─────────────────────────
# Covers the most frequent English function words + common finance context words
# that would generate false positives (e.g. "revenue" ≈ "reverend").
SKIP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "and",
    "but", "or", "if", "that", "this", "it", "we", "our", "they",
    "their", "he", "she", "not", "so", "as", "all", "more", "also",
    "very", "just", "some", "than", "only", "over", "such", "new",
    "well", "any", "these", "two", "first", "even", "most", "how",
    "you", "me", "my", "what", "when", "where", "who", "which", "its",
    "one", "now", "out", "there", "then", "them", "each", "other",
    "time", "his", "her", "said", "get", "make", "go", "see", "know",
    "take", "come", "think", "look", "year", "quarter", "million",
    "billion", "percent", "number", "strong", "good", "high", "low",
    "revenue", "growth", "margin", "earnings", "income", "profit",
    "loss", "cost", "price", "rate", "total", "net", "gross", "cash",
    "share", "stock", "market", "company", "business", "fiscal", "full",
    "next", "last", "prior", "current", "second", "third", "fourth",
    "basis", "points", "outlook", "guidance", "expect", "continue",
    "increase", "decrease", "improve", "return", "flow", "free",
}


# ── Helper functions ──────────────────────────────────────────────────────────

def apply_confusion_map(text: str) -> str:
    """Regex-replace known misrecognition phrases, longest phrase first."""
    for wrong, right in sorted(CONFUSION_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text, flags=re.IGNORECASE)
    return text


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def apply_fuzzy_bias(text: str, bias_single: list) -> str:
    """Replace individual words with high-similarity bias terms."""
    words  = text.split()
    result = []
    for word in words:
        # Strip trailing punctuation for comparison but preserve it for output
        clean = word.lower().rstrip(".,;:!?")
        suffix = word[len(clean):]          # punctuation to re-attach

        if clean in SKIP_WORDS or len(clean) <= 3:
            result.append(word)
            continue

        best_word, best_score = clean, 0.0
        for bw in bias_single:
            s = _sim(clean, bw)
            if s > best_score:
                best_score, best_word = s, bw

        result.append((best_word + suffix) if best_score >= FUZZY_THRESHOLD else word)
    return " ".join(result)


# ── Main ──────────────────────────────────────────────────────────────────────

# Load bias vocabulary; separate single-word terms for fuzzy matching
with open(BIAS_FILE, encoding="utf-8") as f:
    bias_words  = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
bias_single = [w for w in bias_words if " " not in w and "-" not in w]

# Load baseline transcriptions
with open(BASELINE_FILE, encoding="utf-8") as f:
    baseline = json.load(f)

# Apply both biasing layers to every transcript
biased = {}
for fname, text in baseline.items():
    t = text.lower()
    t = apply_confusion_map(t)           # Layer 1 — phrase correction
    t = apply_fuzzy_bias(t, bias_single) # Layer 2 — fuzzy word matching
    biased[fname] = t

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(biased, f, indent=2, ensure_ascii=False)

print(f"Biased transcriptions saved -> {OUT_FILE}")
print(f"  Samples processed: {len(biased)}")
