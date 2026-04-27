import os
import json
import re
from difflib import SequenceMatcher


# ── Config ────────────────────────────────────────────────────────────────────
BASELINE_FILE  = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/predictions_baseline.json"
BIAS_FILE      = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/biaswords_earnings22.txt"
OUT_FILE       = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/predictions_biased.json"
FUZZY_THRESHOLD = 0.82   
# ─────────────────────────────────────────────────────────────────────────────


# ── Layer 1: confusion map ────────────────────────────────────────────────────
# Maps typical Vosk output phrases -> correct finance term.
# Ordered longest -> shortest so greedy matching picks the best fit first.
CONFUSION_MAP = {
    # ── CONFIRMED from actual Vosk output on Earnings-22 corpus ──────────────
    # TeamViewer — Vosk consistently splits/mangles this
    "team fewer":           "teamviewer",
    "team beyond me":       "teamviewer",
    "team you are":         "teamviewer",
    "team yeah":            "teamviewer",
    "team viewer":          "teamviewer",
    # billings / billing — Vosk hears "buildings" / "building"
    "buildings":            "billings",
    "building the pipeline":"billing the pipeline",  # protect this common phrase
    # APAC — split into syllables
    "a pack":               "apac",
    "a pic":                "apac",
    # EMEA — sounds like gibberish to Vosk
    "in the yard":          "emea",
    "a me up":              "emea",
    # ABB — spelled out
    "ab be":                "abb",
    "a b b":                "abb",
    # NRR — "in our" phonetically
    "in our our":           "nrr",
    # Q4 / Q3 — "queue for" / "que three"
    "queue for":            "q4",
    "key for":              "q4",
    "que three":            "q3",
    "key three":            "q3",
    # EBITDA — multiple Vosk renderings
    "chatty every everyday":"ebitda",
    "chatty every":         "ebitda",
    "every everyday":       "ebitda",
    "even day":             "ebitda",
    "e b i t d a":          "ebitda",
    "e bit da":             "ebitda",
    "ebit da":              "ebitda",
    "ebita":                "ebitda",
    "evita":                "ebitda",
    # M&A — "emanate"
    "emanate":              "m&a",
    "em and a":             "m&a",
    # trajectory — "statutory" / "factory"
    "statutory":            "trajectory",
    # margin — "marching" / "martin"
    "marching":             "margin",
    "in terms of marching": "in terms of margin",
    "all martin":           "our margin",
    # marketing — "mocking" / "martin"
    "mocking area":         "marketing area",
    "martin partnerships":  "marketing partnerships",
    # covid — "corbett" / "covert"
    "corbett":              "covid",
    "covert nineteen":      "covid-19",
    # digitalize — "ditch to life"
    "ditch to life":        "digitalize",
    # sponsorships — "bundle shit"
    "bundle shit":          "sponsorships",
    # partnerships — "publish it"
    "publish it with":      "partnerships with",
    # augmented — "op augment"
    "the op augment":       "the augmented",
    # m&a pursuit — "pursue emanate"
    "pursue emanate":       "pursue m&a",
    "pursue":               "pursue",   # keep as-is; just ensuring not replaced
    # ── General finance acronym patterns ─────────────────────────────────────
    "non gap":              "non-gaap",
    "non-gap":              "non-gaap",
    "nongap":               "non-gaap",
    "non gapping":          "non-gaap",
    "cap ex":               "capex",
    "cape x":               "capex",
    "op ex":                "opex",
    "e p s":                "eps",
    "c a g r":              "cagr",
    "a creative":           "accretive",
    "accreted if":          "accretive",
    "sin origies":          "synergies",
    "sin ergies":           "synergies",
    "sin energy":           "synergies",
    "buy back":             "buyback",
    "de leverage":          "deleverage",
    "deliver edge":         "deleverage",
    "head winds":           "headwinds",
    "tail winds":           "tailwinds",
    "tailings":             "tailwinds",
    "pipe and":             "pipeline",
}

# ── Layer 2: words to never replace with a bias term ─────────────────────────
# Covers the most frequent English function words + common finance context words
# that would generate false positives (e.g. "revenue" ≈ "reverend").
SKIP_WORDS = {
    # Pure function words only — do NOT add finance/domain terms here
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
    "take", "come", "think", "look",
    # Safe generic words unlikely to be confused with bias terms
    "million", "billion", "percent", "number", "strong", "good",
    "high", "low", "total", "share", "stock", "fiscal", "full",
    "last", "prior", "current", "third", "fourth", "basis", "points",
    "increase", "decrease", "improve", "return", "flow", "free",
    # NOTE: removed from original skip list — these are KRR targets:
    # margin, guidance, cash, growth, quarter, outlook, net, gross,
    # revenue, earnings, income, profit, loss, cost, price, rate,
    # company, business, next, second, expect, continue
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
