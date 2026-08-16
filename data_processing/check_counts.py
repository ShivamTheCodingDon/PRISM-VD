import os
import json
import sys
import subprocess

# Increase recursion limit for complex ATLAS graphs (though not needed for just counting)
sys.setrecursionlimit(5000)

def count_lines_fast(path):
    if not os.path.exists(path):
        return 0
    try:
        # Use wc -l for speed on large files
        output = subprocess.check_output(['wc', '-l', path]).split()[0]
        return int(output)
    except Exception as e:
        # Fallback to manual count if wc fails
        count = 0
        try:
            with open(path, 'rb') as f:
                for line in f:
                    count += 1
        except:
            pass
        return count

def count_json_array(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
    except:
        pass
    return 0

def print_table(data, headers):
    # Simple table printing without pandas
    widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
    
    header_line = " | ".join(f"{str(h).ljust(w)}" for h, w in zip(headers, widths))
    print(header_line)
    print("-|-".join("-" * w for w in widths))
    
    for row in data:
        print(" | ".join(f"{str(r).ljust(w)}" for r, w in zip(row, widths)))

def main():
    base_dir = "/home/azure/PRISM-VD/PRISM-VD-Enhanced"
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")

    # Pre-calculate original counts to avoid redundant file scans (especially for the 11GB BigVul file)
    print("Pre-calculating original dataset counts...")
    
    # 1. Reveal
    reveal_vul = os.path.join(data_dir, "Reveal", "vulnerables.json")
    reveal_non_vul = os.path.join(data_dir, "Reveal", "non-vulnerables.json")
    reveal_orig_count = count_json_array(reveal_vul) + count_json_array(reveal_non_vul)
    
    # 2. Devign
    devign_orig = os.path.join(base_dir, "devign.json")
    devign_orig_count = count_json_array(devign_orig)
    
    # 3. BigVul
    bigvul_orig = os.path.join(data_dir, "MSR_data_cleaned.jsonlines")
    print(f"Counting lines in {bigvul_orig} (this may take a minute)...")
    bigvul_orig_count = count_lines_fast(bigvul_orig)
    
    orig_counts = {
        "Reveal": reveal_orig_count,
        "Devign": devign_orig_count,
        "BigVul": bigvul_orig_count
    }

    for graph_type in ["causal", "uscp"]:
        results = []
        
        for dataset in ["Reveal", "Devign", "BigVul"]:
            proc_dir = os.path.join(processed_dir, dataset)
            
            # Count each split individually
            counts = {s: count_lines_fast(os.path.join(proc_dir, f"{s}_{graph_type}.jsonlines")) 
                     for s in ['train', 'valid', 'test', 'unmapped']}
            
            proc_count = sum(counts.values())
            orig_count = orig_counts[dataset]
            
            # Calculate percentages
            if proc_count > 0:
                tr_p = (counts['train'] / proc_count) * 100
                va_p = (counts['valid'] / proc_count) * 100
                te_p = (counts['test'] / proc_count) * 100
                split_str = f"Tr:{tr_p:2.0f}% | Va:{va_p:2.0f}% | Te:{te_p:2.0f}%"
            else:
                split_str = "N/A"
            
            results.append([
                dataset, 
                orig_count, 
                proc_count, 
                orig_count - proc_count,
                counts['train'],
                counts['valid'],
                counts['test'],
                split_str
            ])
            
        print(f"\n--- Processing Counts Comparison ({graph_type.capitalize()}) ---")
        headers = ["Dataset", "Original", "Processed", "Missing", "Train", "Val", "Test", "Distribution"]
        print_table(results, headers)
        
        print("\nDetailed breakdown for missing samples:")
        for res in results:
            missing = res[3]
            if missing > 0:
                print(f"- {res[0]}: {missing} samples missing (likely due to ATLAS/USCP extraction errors or pruning).")
            elif missing < 0:
                print(f"- {res[0]}: {abs(missing)} extra samples found (?)")

if __name__ == "__main__":
    main()
