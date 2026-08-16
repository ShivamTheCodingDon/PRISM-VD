import json
import os
import glob
from collections import defaultdict

models = ["Codestral-Agent", "gpt-4.1-2", "DeepSeek-V3.2"]
datasets = ["qemu", "xen", "ffmpeg", "openssl", "libav"]
years_of_interest = ['2026', '2025', '2024', '2023', '2022']

for model in models:
    print(f"\n# {model} - Latest CVEs (Last 5 Years) Aggregated")
    print("| Dataset | Year | Samples | Actual Vul | Pred Vul | TP | FP | TN | FN | Accuracy |")
    print("|---------|------|---------|------------|----------|----|----|----|----|----------|")
    
    for dataset in datasets:
        pred_file = f"results_{model}/{dataset}/predictions.jsonl"
        if not os.path.exists(pred_file):
            continue
            
        # Aggregate stats
        year_stats = defaultdict(lambda: {"total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "act": 0, "prd": 0})
        
        with open(pred_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                year = rec.get("year", "")
                if year not in years_of_interest:
                    continue
                    
                true_label = rec["true_label"]
                pred_label = rec["pred_label"]
                
                y = year_stats[year]
                y["total"] += 1
                if true_label == 1: y["act"] += 1
                if pred_label == 1: y["prd"] += 1
                
                if true_label == 1 and pred_label == 1: y["tp"] += 1
                elif true_label == 0 and pred_label == 1: y["fp"] += 1
                elif true_label == 0 and pred_label == 0: y["tn"] += 1
                elif true_label == 1 and pred_label == 0: y["fn"] += 1
                
        for year in years_of_interest:
            if year not in year_stats: continue
            ys = year_stats[year]
            if ys["total"] == 0: continue
            
            acc = (ys["tp"] + ys["tn"]) / ys["total"]
            
            print(f"| **{dataset.upper()}** | {year} | {ys['total']} | {ys['act']} | {ys['prd']} | {ys['tp']} | {ys['fp']} | {ys['tn']} | {ys['fn']} | {acc:.4f} |")

print("\nAggregation complete!")
