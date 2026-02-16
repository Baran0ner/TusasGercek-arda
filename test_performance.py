"""
Performance test: Paralel vs Serial GA comparison
"""
import time
import sys
import io

# Windows UTF-8 fix
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tusas.core.laminate_optimizer import LaminateOptimizer


def verify_sequence(seq, label):
    """Verify sequence validity"""
    print(f"\n{label}:")
    print(f"  Length: {len(seq)}")
    print(f"  First 6: {seq[:6]}")
    print(f"  Last 6:  {seq[-6:]}")

    # Check symmetry
    is_sym = seq == seq[::-1]
    print(f"  Symmetric: {is_sym}")

    # Check first/last 2 are ±45
    first_two = set(seq[:2])
    last_two = set(seq[-2:])
    valid_external = first_two.issubset({45, -45}) and last_two.issubset({45, -45})
    print(f"  Valid external: {valid_external}")

    return is_sym and valid_external


if __name__ == '__main__':
    # Test case: 36-ply composite
    ply_counts = {
        0: 12,
        90: 8,
        45: 8,
        -45: 8
    }

    print("=" * 70)
    print("PERFORMANCE TEST: Paralel vs Serial GA")
    print("=" * 70)
    print(f"Test case: 36-ply composite (0deg:12, 90deg:8, +/-45deg:8 each)")
    print()

    optimizer = LaminateOptimizer(ply_counts)

    # Test 1: Serial (parallel=False)
    print("\n" + "=" * 70)
    print("TEST 1: SERIAL MODE (parallel=False)")
    print("=" * 70)
    start_serial = time.time()
    skeleton = optimizer._build_smart_skeleton()
    seq_serial, score_serial = optimizer._multi_start_ga(skeleton, n_runs=7, parallel=False)
    time_serial = time.time() - start_serial
    print(f"\n[OK] Serial completed in {time_serial:.2f}s")
    print(f"  Best score: {score_serial:.2f}/100")

    # Test 2: Parallel (parallel=True)
    print("\n" + "=" * 70)
    print("TEST 2: PARALLEL MODE (parallel=True)")
    print("=" * 70)
    start_parallel = time.time()
    skeleton = optimizer._build_smart_skeleton()
    seq_parallel, score_parallel = optimizer._multi_start_ga(skeleton, n_runs=7, parallel=True)
    time_parallel = time.time() - start_parallel
    print(f"\n[OK] Parallel completed in {time_parallel:.2f}s")
    print(f"  Best score: {score_parallel:.2f}/100")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Serial time:   {time_serial:.2f}s  (score: {score_serial:.2f})")
    print(f"Parallel time: {time_parallel:.2f}s  (score: {score_parallel:.2f})")
    speedup = time_serial / time_parallel if time_parallel > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x faster with parallel processing")

    # Verify correctness: Both should find similar quality solutions
    score_diff = abs(score_serial - score_parallel)
    print(f"\nScore difference: {score_diff:.2f} (should be small)")

    if score_diff < 5.0:
        print("[OK] Both methods found similar quality solutions")
    else:
        print("[WARNING] Large score difference detected")

    # Verify sequences are valid
    print("\n" + "=" * 70)
    print("SEQUENCE VALIDATION")
    print("=" * 70)
    valid_serial = verify_sequence(seq_serial, "Serial sequence")
    valid_parallel = verify_sequence(seq_parallel, "Parallel sequence")

    if valid_serial and valid_parallel:
        print("\n[OK] All sequences are valid and follow design rules!")
    else:
        print("\n[ERROR] Some sequences are invalid!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("TEST PASSED [OK]")
    print("=" * 70)
