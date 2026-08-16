import os
import json

base = "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/Devign_Abl_gnn_rev"
results = []
for d in os.listdir(base):
    dp = os.path.join(base, d)
    if not os.path.isdir(dp): continue
    for f in os.listdir(dp):
        if f.endswith(".json") or f.endswith(".txt"):
            try:
                with open(os.path.join(dp, f)) as jf:
                    data = json.load(jf)
                    auc = data.get("AUC", 0) or data.get("auc", 0) or 0
                    f1 = data.get("F1", 0) or data.get("f1", 0) or data.get("f1_score", 0) or 0
                    if auc > 0 or f1 > 0:
                        results.append({"model": d, "auc": float(auc), "f1": float(f1)})
            except:
                pass

results.sort(key=lambda x: (x["auc"], x["f1"]), reverse=True)
for r in results:
    print(f"Model: {r['model']}, AUC: {r['auc']}, F1: {r['f1']}")
