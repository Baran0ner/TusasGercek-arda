"""
Skor Analizi: Hangi kurallar puan kaybettiriyor?
"""
import sys, io
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tusas.core.laminate_optimizer import LaminateOptimizer

configs = [
    ({0: 12, 90: 8, 45: 8, -45: 8}, "36 ply"),
    ({0: 18, 90: 12, 45: 14, -45: 14}, "58 ply"),
    ({0: 18, 90: 18, 45: 18, -45: 18}, "72 ply (büyük)"),
    ({0: 6, 90: 4, 45: 4, -45: 4}, "18 ply (küçük)"),
    ({0: 10, 90: 6, 45: 4, -45: 4}, "24 ply (dengesiz)"),
]

for ply_counts, label in configs:
    print("=" * 60)
    print(f"CONFIG: {label} — {ply_counts}")
    print(f"Toplam: {sum(ply_counts.values())} ply")
    print("=" * 60)

    opt = LaminateOptimizer(ply_counts)
    seq, score, details, _ = opt.run_hybrid_optimization()

    print(f"\nFinal Skor: {score:.2f}/100")
    print(f"Dizilim: {seq}")

    # Kural bazlı analiz
    rules = details.get("rules", {})
    loss_rules = []
    for rule_name in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
        r = rules.get(rule_name, {})
        weight = r.get("weight", 0)
        sc = r.get("score", 0)
        penalty = r.get("penalty", 0)
        reason = r.get("reason", "")
        loss = weight - sc
        pct = (sc / weight * 100) if weight > 0 else 100
        marker = "  ✗ KAYIP" if loss > 0.5 else ""
        print(f"  {rule_name}: {sc:.2f}/{weight:.1f} ({pct:.0f}%) — kayıp: {loss:.2f}{marker}  {reason}")
        if loss > 0.1:
            loss_rules.append((rule_name, loss))

    if loss_rules:
        loss_rules.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  En çok puan kaybettiren kurallar:")
        for r, l in loss_rules:
            print(f"    {r}: -{l:.2f} puan")

    gstats = opt._grouping_stats(seq)
    print(f"\n  Grouping: 2'li={gstats['groups_len_2']}, 3'lü={gstats['groups_len_3']}, 4+={gstats['groups_len_ge4']}, max_run={gstats['max_run']}")
    print()
