import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

jsonlines_path = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/data/processed/Devign/test_uscp.jsonlines"
output_json = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/sast_res/clang_preds.json"
output_dir = "temp_clang_files"

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
    bugs = ""
    try:
        res = subprocess.run(["clang", "--analyze", file_path], capture_output=True, text=True, timeout=5)
        bugs = res.stderr
        if "warning:" in bugs or "error:" in bugs:
            pred = 1
    except Exception as e:
        bugs = str(e)
        pred = 0
        
    return {
        "id": id_str,
        "code": code,
        "ground": label,
        "pred": pred,
        "bugs": bugs
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

if __name__ == "__main__":
    main()
