import json
import re
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

json_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res/clang_preds.json"
csv_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res/metrics.csv"

# Load the predictions
with open(json_path, "r") as f:
    data = json.load(f)

y_true = []
y_pred = []

# Process each item
for item in data:
    # Extract specific bugs from the stderr output
    bugs_text = item.get("bugs", "")
    
    # We look for lines matching "warning: message" or "error: message"
    # To avoid matching too long strings, we can just split by line and look for "warning:" or "error:"
    bug_list = []
    for line in bugs_text.split("\n"):
        match = re.search(r"(warning|error):\s*(.+)", line)
        if match:
            # e.g., "error: use of undeclared identifier 'NULL'"
            bug_str = f"{match.group(1)}: {match.group(2)}"
            if bug_str not in bug_list:
                bug_list.append(bug_str)
                
    item["extracted_bugs"] = bug_list
    
    y_true.append(item.get("ground", item.get("ground_truth")))
    y_pred.append(item.get("pred", item.get("prediction")))

# Re-save the JSON with extracted bugs
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Save to CSV
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Tool", "Accuracy", "Precision", "Recall", "F1_Score"])
    writer.writerow(["Clang", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

print(f"Updated JSON saved to {json_path}")
print(f"Metrics saved to {csv_path}")
