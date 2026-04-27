import os, json, statistics

DIR = "C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/INTERMEDIATE_ANALYSIS"
scores = []

for f in os.listdir(DIR):
    if not f.endswith(".json"):
        continue

    data = json.load(open(os.path.join(DIR, f)))

    for block in data.get("results", []):
        for w in block.get("result", []):
            if "conf" in w:
                scores.append(w["conf"])

    for w in data.get("final", {}).get("result", []):
        if "conf" in w:
            scores.append(w["conf"])

if not scores:
    print("⚠️ No confidence found")
else:
    stats = {
        "avg": statistics.mean(scores),
        "min": min(scores),
        "max": max(scores)
    }

    json.dump(stats, open("C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/confidence.json", "w"), indent=2)
    print("✅ Confidence:", stats)