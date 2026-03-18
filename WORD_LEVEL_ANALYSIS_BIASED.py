import json
from jiwer import process_words

GT_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/audio/finance_gt/ground_truth.json"
PRED_PATH = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/predictions_biased.json"

with open(GT_PATH) as f:
    ground_truth = json.load(f)

with open(PRED_PATH) as f:
    predictions = json.load(f)

analysis = []

for file in predictions:
    if file in ground_truth:
        gt = ground_truth[file]
        pred = predictions[file]

        result = process_words(gt, pred)

        analysis.append({
            "file": file,
            "substitutions": result.substitutions,
            "insertions": result.insertions,
            "deletions": result.deletions
        })

with open("C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/word_error_analysis_biased.json", "w") as f:
    json.dump(analysis, f, indent=2)

print("Word-level error analysis saved.")