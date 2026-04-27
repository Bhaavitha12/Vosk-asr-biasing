import json

# Load JSON files
with open("C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/word_error_analysis_baseline.json", "r") as f:
    baseline = json.load(f)

with open("C:/Desktop/PROJECT_PART_II/FinanceEarnings22_Execution/output/word_error_analysis_biased.json", "r") as f:
    biased = json.load(f)


# Function to aggregate errors
def aggregate(data):
    sub, ins, dele = 0, 0, 0
    for item in data:
        sub += item["substitutions"]
        ins += item["insertions"]
        dele += item["deletions"]
    total = sub + ins + dele
    return sub, ins, dele, total


# Compute totals
b_sub, b_ins, b_del, b_total = aggregate(baseline)
bi_sub, bi_ins, bi_del, bi_total = aggregate(biased)


# Compute percentages
def percent(val, total):
    return (val / total * 100) if total != 0 else 0


# Improvement calculation
def improvement(before, after):
    return ((before - after) / before * 100) if before != 0 else 0


# Print results
print("\n===== ERROR ANALYSIS =====\n")

print("Baseline Totals:")
print(f"Substitutions: {b_sub}")
print(f"Insertions:    {b_ins}")
print(f"Deletions:     {b_del}")
print(f"Total Errors:  {b_total}\n")

print("Biased Totals:")
print(f"Substitutions: {bi_sub}")
print(f"Insertions:    {bi_ins}")
print(f"Deletions:     {bi_del}")
print(f"Total Errors:  {bi_total}\n")


print("===== PERCENTAGE BREAKDOWN =====\n")
print(f"Baseline -> Sub: {percent(b_sub, b_total):.2f}%, Ins: {percent(b_ins, b_total):.2f}%, Del: {percent(b_del, b_total):.2f}%")
print(f"Biased   -> Sub: {percent(bi_sub, bi_total):.2f}%, Ins: {percent(bi_ins, bi_total):.2f}%, Del: {percent(bi_del, bi_total):.2f}%")


print("\n===== IMPROVEMENT (%) =====\n")
print(f"Substitution Reduction: {improvement(b_sub, bi_sub):.2f}%")
print(f"Insertion Reduction:    {improvement(b_ins, bi_ins):.2f}%")
print(f"Deletion Reduction:     {improvement(b_del, bi_del):.2f}%")


print("\n===== PPT TABLE =====\n")
print("Error Type     Baseline   Biased")
print(f"Substitution   {b_sub}         {bi_sub}")
print(f"Insertion      {b_ins}         {bi_ins}")
print(f"Deletion       {b_del}         {bi_del}")