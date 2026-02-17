# coding: utf-8
import sys, io, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from tusas.core.laminate_optimizer import LaminateOptimizer

configs = [
    ({0:12, 90:8, 45:8, -45:8}, "36ply"),
    ({0:18, 90:12, 45:14, -45:14}, "58ply"),
    ({0:6, 90:4, 45:4, -45:4}, "18ply"),
]

for ply_counts, label in configs:
    opt = LaminateOptimizer(ply_counts)
    seq, score, details, _ = opt.run_hybrid_optimization()
    rules = details.get("rules", {})
    # Write to json for reliable reading
    out = {"label": label, "total": score, "seq": seq}
    for r in ["R1","R2","R3","R4","R5","R6","R7","R8"]:
        d = rules.get(r, {})
        out[r] = {"score": d.get("score",0), "weight": d.get("weight",0), "loss": round(d.get("weight",0)-d.get("score",0), 2)}
    gs = opt._grouping_stats(seq)
    out["grouping"] = gs
    with open(f"analysis_{label}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved analysis_{label}.json")
