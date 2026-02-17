
import sys
import collections
from tusas.core.laminate_optimizer import LaminateOptimizer
from tusas.core.dropoff_optimizer import DropOffOptimizer

# Redirect stdout to a file to capture clean output if needed, 
# but for now we rely on simple prints.

def test_dropoff():
    print("TEST_START")
    # Master: 8, 8, 8, 8. Total 32.
    counts_master = {0: 8, 45: 8, -45: 8, 90: 8}
    opt_master = LaminateOptimizer(counts_master)
    master_seq = opt_master._build_smart_skeleton()
    # Force strict symmetry if skeleton is random? Skeleton is built symmetric.
    print(f"Master Seq: {master_seq}")
    print(f"Master Counts: {dict(collections.Counter(master_seq))}")

    drop_opt = DropOffOptimizer(master_seq, opt_master)
    
    # Target: 8, 7, 7, 8
    target_counts = {0: 8, 45: 7, -45: 7, 90: 8}
    print(f"Target Counts: {target_counts}")

    try:
        new_seq, score, dropped = drop_opt.optimize_drop_with_angle_targets(target_counts)
        print(f"Result Seq: {new_seq}")
        res_counts = dict(collections.Counter(new_seq))
        print(f"Result Counts: {res_counts}")
        print(f"Result Score: {score}")

        if new_seq == master_seq:
            print("FAILURE: Master sequence returned.")
        else:
            print("SUCCESS: New sequence generated.")
            # Verify counts
            if res_counts.get(45) == 7 and res_counts.get(-45) == 7:
                 print("VERIFIED: 45 and -45 counts are correct.")
            else:
                 print("MISMATCH: Counts are not as requested.")
                 
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_dropoff()
