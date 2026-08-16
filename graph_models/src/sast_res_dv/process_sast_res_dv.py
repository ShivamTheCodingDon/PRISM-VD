import json
import os
import subprocess
import re
import csv
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

jsonlines_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/data/processed/Devign/test_uscp.jsonlines"
# Wait, this is for devign! The user said @sast_res_dv
# Is the file inside sast_res_dv or do they just want to process sast_res_dv's jsonlines?
# Let's read from sast_res_dv/test_uscp.jsonlines!
jsonlines_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res_dv/test_uscp.jsonlines"
output_json = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res_dv/clang_preds.json"
csv_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res_dv/metrics.csv"
output_dir = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res_dv/temp_clang_files"

def process_item(item):
    i, line = item
    data = json.loads(line.strip())
    id_str = data["id"]
    code = data["code"]
    label = data["label"]
    
    file_path = os.path.join(output_dir, f"{id_str}.c")
    with open(file_path, "w") as out_f:
        out_f.write(code)
        
    pred = 0
    bugs_text = ""
    try:
        res = subprocess.run(["clang", "--analyze", file_path], capture_output=True, text=True, timeout=5)
        bugs_text = res.stderr
        if "warning:" in bugs_text or "error:" in bugs_text:
            pred = 1
    except Exception as e:
        bugs_text = str(e)
        pred = 0
        
    # Extract specific bugs from the stderr output
    bug_list = []
    for line in bugs_text.split("\n"):
        match = re.search(r"(warning|error):\s*(.+)", line)
        if match:
            bug_str = f"{match.group(1)}: {match.group(2)}"
            if bug_str not in bug_list:
                bug_list.append(bug_str)
                
    return {
        "id": id_str,
        "code": code,
        "ground": label,
        "pred": pred,
        "bugs": bugs_text,
        "extracted_bugs": bug_list
    }

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(jsonlines_path, "r") as f:
        lines = f.readlines()
        
    results = []
    print(f"Processing {len(lines)} files using ThreadPoolExecutor...")
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        for i, res in enumerate(executor.map(process_item, enumerate(lines))):
            results.append(res)
            if (i+1) % 500 == 0:
                print(f"Processed {i+1}/{len(lines)} files.")
                
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {output_json}")
    
    y_true = [item["ground"] for item in results]
    y_pred = [item["pred"] for item in results]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Tool", "Accuracy", "Precision", "Recall", "F1_Score"])
        writer.writerow(["Clang", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])
        
    print(f"Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
