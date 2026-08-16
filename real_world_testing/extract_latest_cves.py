import json
import re
from collections import defaultdict
import os

with open('real_world_metrics_summary.json', 'r') as f:
    data = json.load(f)

years_of_interest = ['2026', '2025', '2024', '2023', '2022']

table_lines = [
    "| Dataset | Year | # CVEs | Samples | Actual Vul | Pred Vul | TP | FP | TN | FN | Accuracy |",
    "|---------|------|--------|---------|------------|----------|----|----|----|----|----------|"
]

for dataset, metrics in data.items():
    cve_breakdown = metrics.get('cve_breakdown', {})
    
    # Group by year
    year_stats = defaultdict(lambda: {"num_cves": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "act": 0, "prd": 0})
    
    for cve, stats in cve_breakdown.items():
        if cve == "UNKNOWN_CVE":
            continue
        parts = cve.split('-')
        if len(parts) >= 2:
            year = parts[1]
            if year in years_of_interest:
                y = year_stats[year]
                y["num_cves"] += 1
                y["total"] += stats.get("total", 0)
                y["tp"] += stats.get("tp", 0)
                y["fp"] += stats.get("fp", 0)
                y["tn"] += stats.get("tn", 0)
                y["fn"] += stats.get("fn", 0)
                y["act"] += stats.get("actual_vul", 0)
                y["prd"] += stats.get("pred_vul", 0)
                
    if not year_stats:
        continue
        
    for year in years_of_interest:
        if year not in year_stats:
            continue
        ys = year_stats[year]
        if ys["total"] == 0:
            continue
            
        acc = (ys["tp"] + ys["tn"]) / ys["total"]
        
        line = f"| **{dataset.upper()}** | {year} | {ys['num_cves']} | {ys['total']} | {ys['act']} | {ys['prd']} | {ys['tp']} | {ys['fp']} | {ys['tn']} | {ys['fn']} | {acc:.4f} |"
        table_lines.append(line)

# output to stdout
print("\n".join(table_lines))

