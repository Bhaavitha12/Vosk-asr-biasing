import json
from collections import Counter   # Counter helps us keep track of counts easily

# File paths
BIAS_FILE = "bias_words.txt"                      # file containing bias words
GT_FILE = "audio/ground_truth/84-121123.trans.txt" # ground truth transcription file
BASE_FILE = "output/predictions.json"             # predictions from normal model
BIAS_PRED_FILE = "output/predictions_with_bias.json" # predictions from biased model


# -------------------- LOAD BIAS WORDS --------------------

# Open the bias words file and store them in a set
# Using a set makes checking faster
with open(BIAS_FILE) as f:
    bias_words = set(w.strip().lower() for w in f if w.strip())


# -------------------- LOAD GROUND TRUTH --------------------

# Dictionary to store correct transcription for each audio file
ground_truth = {}

with open(GT_FILE) as f:
    for line in f:
        # Each line has format:  audioID sentence
        parts = line.strip().split(" ", 1)

        # We check if the format is correct
        if len(parts) == 2:
            # Add ".wav" because our prediction files use wav filenames
            ground_truth[parts[0] + ".wav"] = parts[1].lower()


# -------------------- LOAD MODEL PREDICTIONS --------------------

# Load baseline predictions (without bias)
with open(BASE_FILE) as f:
    base_preds = json.load(f)

# Load predictions after applying bias
with open(BIAS_PRED_FILE) as f:
    bias_preds = json.load(f)


# -------------------- CREATE STATS COUNTER --------------------

# We use counters to keep track of totals and correct predictions
stats = {
    "baseline": Counter(),
    "biased": Counter()
}


# -------------------- COMPARE PREDICTIONS --------------------

# Loop through each audio file in ground truth
for wav, gt_text in ground_truth.items():

    # Skip if prediction is missing in either file
    if wav not in base_preds or wav not in bias_preds:
        continue

    # Split ground truth sentence into words
    gt_words = gt_text.split()

    # Convert predictions into word sets
    # Sets make it easy to check if a word exists
    base_words = set(base_preds[wav].lower().split())
    bias_words_pred = set(bias_preds[wav].lower().split())


    # Now we check only the bias words
    for w in gt_words:

        # If this word is one of the bias words
        if w in bias_words:

            # Count how many bias words appear in the ground truth
            stats["baseline"]["total"] += 1
            stats["biased"]["total"] += 1

            # Check if baseline predicted it correctly
            if w in base_words:
                stats["baseline"]["correct"] += 1

            # Check if biased model predicted it correctly
            if w in bias_words_pred:
                stats["biased"]["correct"] += 1


# -------------------- CALCULATE RECALL --------------------

# Recall = correctly predicted bias words / total bias words
baseline_recall = stats["baseline"]["correct"] / stats["baseline"]["total"]
biased_recall = stats["biased"]["correct"] / stats["biased"]["total"]


# -------------------- PRINT RESULTS --------------------

print("Baseline Bias Keyword Recall :", baseline_recall)
print("Biased Bias Keyword Recall   :", biased_recall)

# Calculate percentage improvement after biasing
print("Relative Improvement (%)     :", 
      ((biased_recall - baseline_recall) / baseline_recall) * 100)