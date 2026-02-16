"""
Test optimized performance improvements
"""
import time
import sys
import io

# Windows UTF-8 fix
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tusas.core.laminate_optimizer import LaminateOptimizer


if __name__ == '__main__':
    # Test case: 36-ply composite
    ply_counts = {
        0: 12,
        90: 8,
        45: 8,
        -45: 8
    }

    print("=" * 70)
    print("OPTIMIZED PERFORMANCE TEST")
    print("=" * 70)
    print("Improvements:")
    print("  - Reduced population: 100 -> 80")
    print("  - Adaptive early stopping (stops faster for good solutions)")
    print("  - Reduced stagnation limit: 25 -> 20")
    print()

    optimizer = LaminateOptimizer(ply_counts)

    # Run 5 tests to get average
    times = []
    scores = []

    for test_num in range(5):
        print(f"\n--- Test {test_num + 1}/5 ---")
        start = time.time()
        skeleton = optimizer._build_smart_skeleton()
        seq, score = optimizer._multi_start_ga(skeleton, n_runs=7, parallel=False)
        elapsed = time.time() - start
        times.append(elapsed)
        scores.append(score)
        print(f"Time: {elapsed:.2f}s, Score: {score:.2f}/100")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Average time:  {sum(times)/len(times):.2f}s")
    print(f"Min time:      {min(times):.2f}s")
    print(f"Max time:      {max(times):.2f}s")
    print(f"Average score: {sum(scores)/len(scores):.2f}/100")
    print(f"Min score:     {min(scores):.2f}/100")
    print(f"Max score:     {max(scores):.2f}/100")

    avg_time = sum(times) / len(times)
    avg_score = sum(scores) / len(scores)

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print(f"OLD: ~3.2s, score ~92/100")
    print(f"NEW: {avg_time:.2f}s, score {avg_score:.2f}/100")

    speedup = 3.2 / avg_time
    quality_change = (avg_score - 92) / 92 * 100

    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Quality change: {quality_change:+.1f}%")

    if speedup > 1.2 and quality_change > -3:
        print("\n[SUCCESS] Significant speedup with acceptable quality!")
    elif speedup > 1.0 and quality_change > -1:
        print("\n[OK] Moderate improvement")
    else:
        print("\n[WARNING] Limited improvement")
