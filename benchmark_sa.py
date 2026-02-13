"""
Simulated Annealing vs Hill Climbing Benchmark
===============================================
Aynı ply konfigürasyonunu hem SA hem Hill Climbing ile çalıştırarak
sonuçları karşılaştırır.
"""

import sys
import io
import time

# Windows UTF-8 desteği
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tusas.core.laminate_optimizer import LaminateOptimizer


def run_benchmark(ply_counts: dict, label: str):
    """Tek bir konfigürasyon için SA vs Hill Climbing benchmark."""
    print("=" * 70)
    print(f"BENCHMARK: {label}")
    print(f"Config: {ply_counts}")
    print(f"Total plies: {sum(ply_counts.values())}")
    print("=" * 70)

    # ---- Phase 1 & 2: Ortak (Skeleton + GA) ----
    optimizer = LaminateOptimizer(ply_counts)

    print("\n--- Ortak Fazlar (Phase 1 + Phase 2) ---")
    skeleton = optimizer._build_smart_skeleton()
    n_runs = 4 if optimizer.total_plies <= 40 else 5
    ga_best_seq, ga_best_score = optimizer._multi_start_ga(skeleton, n_runs=n_runs)
    print(f"GA sonucu: {ga_best_score:.2f}/100\n")

    # ---- Test 1: Hill Climbing (Local Search) ----
    print("--- TEST 1: Hill Climbing (Local Search) ---")
    t1 = time.time()
    hc_seq, hc_score = optimizer._local_search(ga_best_seq[:], max_iter=30)
    hc_time = time.time() - t1
    _, hc_details = optimizer.calculate_fitness(hc_seq)
    hc_grouping_stats = optimizer._grouping_stats(hc_seq)

    # ---- Test 2: Simulated Annealing ----
    print("\n--- TEST 2: Simulated Annealing ---")
    t2 = time.time()
    sa_seq, sa_score = optimizer._simulated_annealing(ga_best_seq[:])
    sa_time = time.time() - t2
    _, sa_details = optimizer.calculate_fitness(sa_seq)
    sa_grouping_stats = optimizer._grouping_stats(sa_seq)

    # ---- Sonuçlar ----
    print("\n" + "=" * 70)
    print("SONUÇLAR")
    print("=" * 70)
    diff = sa_score - hc_score
    winner = "SA" if diff > 0 else ("HC" if diff < 0 else "Berabere")

    print(f"  Hill Climbing:         {hc_score:.2f}/100  ({hc_time:.2f}s)")
    print(f"  Simulated Annealing:   {sa_score:.2f}/100  ({sa_time:.2f}s)")
    print(f"  Fark:                  {diff:+.2f} puan ({winner} lehine)")
    print()

    # Kural bazlı karşılaştırma
    print("  Kural Bazlı Karşılaştırma:")
    print(f"  {'Kural':<8} {'HC Skor':>10} {'SA Skor':>10} {'Fark':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for rule in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
        hc_r = hc_details.get("rules", {}).get(rule, {})
        sa_r = sa_details.get("rules", {}).get(rule, {})
        hc_s = hc_r.get("score", 0)
        sa_s = sa_r.get("score", 0)
        d = sa_s - hc_s
        marker = " ✓" if d > 0 else (" ✗" if d < 0 else "")
        print(f"  {rule:<8} {hc_s:>10.2f} {sa_s:>10.2f} {d:>+10.2f}{marker}")

    print(f"\n  Grouping (2'li/3'lü/4+):")
    print(f"    HC: 2'li={hc_grouping_stats['groups_len_2']}, 3'lü={hc_grouping_stats['groups_len_3']}, 4+={hc_grouping_stats['groups_len_ge4']}")
    print(f"    SA: 2'li={sa_grouping_stats['groups_len_2']}, 3'lü={sa_grouping_stats['groups_len_3']}, 4+={sa_grouping_stats['groups_len_ge4']}")
    print()

    return {
        "hc_score": hc_score, "sa_score": sa_score,
        "hc_time": hc_time, "sa_time": sa_time,
        "diff": diff, "winner": winner
    }


if __name__ == "__main__":
    configs = [
        ({"0": 12, "90": 8, "45": 8, "-45": 8}, "36 ply (küçük)"),
        ({"0": 18, "90": 12, "45": 14, "-45": 14}, "58 ply (orta)"),
    ]

    results = []
    for config, label in configs:
        ply_counts = {int(k): int(v) for k, v in config.items()}
        r = run_benchmark(ply_counts, label)
        results.append(r)

    # Genel özet
    print("\n" + "=" * 70)
    print("GENEL ÖZET")
    print("=" * 70)
    sa_wins = sum(1 for r in results if r["winner"] == "SA")
    hc_wins = sum(1 for r in results if r["winner"] == "HC")
    ties = sum(1 for r in results if r["winner"] == "Berabere")
    avg_diff = sum(r["diff"] for r in results) / len(results)
    print(f"  SA kazandı: {sa_wins}/{len(results)}")
    print(f"  HC kazandı: {hc_wins}/{len(results)}")
    print(f"  Berabere:   {ties}/{len(results)}")
    print(f"  Ortalama fark: {avg_diff:+.2f} puan")
