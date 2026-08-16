import json
import os
import subprocess
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def main():
    jsonlines_path = "test_uscp.jsonlines"
    output_dir = "temp_c_files"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    y_true = []
    flawfinder_preds = []
    cppcheck_preds = []
    
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
        
        # Flawfinder
        try:
            ff_res = subprocess.run(["flawfinder", "--csv", "--quiet", file_path], capture_output=True, text=True)
            ff_output = ff_res.stdout.strip().split("\n")
            if len(ff_output) > 1: # Header is one line
                flawfinder_preds.append(1)
            else:
                flawfinder_preds.append(0)
        except Exception as e:
            flawfinder_preds.append(0)
            
        # Cppcheck
        try:
            cpp_res = subprocess.run(["cppcheck", "--quiet", "--template={severity}", file_path], capture_output=True, text=True)
            err_output = cpp_res.stderr.strip()
            if err_output and ("error" in err_output or "warning" in err_output):
                cppcheck_preds.append(1)
            else:
                cppcheck_preds.append(0)
        except Exception as e:
            cppcheck_preds.append(0)
            
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(lines)} files.")
            
    print(f"Finished processing all {len(lines)} files.\n")
    
    print("=== Flawfinder Metrics ===")
    acc, prec, rec, f1 = calc_metrics(y_true, flawfinder_preds)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}\n")
    
    print("=== Cppcheck Metrics ===")
    acc, prec, rec, f1 = calc_metrics(y_true, cppcheck_preds)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}\n")
    
    # Clean up the temporary folder after execution
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    main()
