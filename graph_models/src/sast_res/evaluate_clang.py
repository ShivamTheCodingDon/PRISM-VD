import json
import subprocess
import os
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def main():
    jsonlines_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/data/processed/Devign/test_uscp.jsonlines"
    output_dir = "temp_clang_files"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    y_true = []
    clang_preds = []
    
    print("Reading dataset and generating files...")
    with open(jsonlines_path, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        id_str = data["id"]
        code = data["code"]
        label = data["label"]
        
        file_path = os.path.join(output_dir, f"{id_str}.c")
        with open(file_path, "w") as out_f:
            out_f.write(code)
            
        y_true.append(label)
        
        # Clang Static Analyzer
        try:
            res = subprocess.run(["clang", "--analyze", file_path], capture_output=True, text=True, timeout=5)
            output = res.stderr
            if "warning:" in output or "error:" in output:
                clang_preds.append(1)
            else:
                clang_preds.append(0)
        except Exception as e:
            clang_preds.append(0)
            
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(lines)} files.")
            
    print(f"Finished processing all {len(lines)} files.\n")
    
    print("=== Clang SAST Metrics ===")
    acc, prec, rec, f1 = calc_metrics(y_true, clang_preds)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}\n")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    main()
