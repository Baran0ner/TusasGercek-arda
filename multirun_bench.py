# coding: utf-8
"""Multi-run benchmark: her config 5 kez calistirilir, istatistik raporlanir."""
import sys, io, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from tusas.core.laminate_optimizer import LaminateOptimizer

configs = [
    ({0:12, 90:8, 45:8, -45:8}, "36ply"),
    ({0:18, 90:12, 45:14, -45:14}, "58ply"),
]

results = {}
N_RUNS = 3

for ply_counts, label in configs:
    scores = []
    for run in range(N_RUNS):
        opt = LaminateOptimizer(ply_counts)
        seq, score, details, _ = opt.run_hybrid_optimization()
        rules = details.get("rules", {})
        r5 = rules.get("R5", {}).get("score", 0)
        r7 = rules.get("R7", {}).get("score", 0)
        r8 = rules.get("R8", {}).get("score", 0)
        scores.append({"total": score, "R5": r5, "R7": r7, "R8": r8})
        print(f"  {label} run {run+1}: {score:.2f} (R5={r5:.2f}, R7={r7:.2f}, R8={r8:.2f})")

    avg_total = sum(s["total"] for s in scores) / len(scores)
    best_total = max(s["total"] for s in scores)
    results[label] = {"avg": round(avg_total, 2), "best": round(best_total, 2), "runs": scores}
    print(f"\n  {label}: avg={avg_total:.2f}, best={best_total:.2f}\n")

with open("multirun_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved multirun_results.json")
