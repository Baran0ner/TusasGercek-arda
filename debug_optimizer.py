
from tusas.core.laminate_optimizer import LaminateOptimizer
from tusas.core.dropoff_optimizer import DropOffOptimizer
import collections

def test_laminate_optimizer_8778():
    print("--- Testing LaminateOptimizer with 8-7-7-8 ---")
    # 0:8, 45:7, -45:7, 90:8. Total 30.
    counts = {0: 8, 45: 7, -45: 7, 90: 8}
    try:
        opt = LaminateOptimizer(counts)
        seq = opt._create_symmetric_individual()
        print("Sequence generated:", seq)
        print("Counts:", dict(collections.Counter(seq)))
        print("Is Symmetric:", opt._is_symmetric(seq))
    except Exception as e:
        print("Optimization Failed with error:", str(e))

def test_dropoff_8888_to_8778():
    print("\n--- Testing DropOffOptimizer 8-8-8-8 to 8-7-7-8 ---")
    # Master: 8, 8, 8, 8 (Strict Symmetric). Total 32.
    # Target: 0:8, 45:7, -45:7, 90:8. Total 30.
    
    # Create a valid master sequence
    counts_master = {0: 8, 45: 8, -45: 8, 90: 8}
    opt_master = LaminateOptimizer(counts_master)
    master_seq = opt_master._build_smart_skeleton() # Should work
    print(f"Master Sequence (len {len(master_seq)}): {master_seq}")
    print("Master Counts:", dict(collections.Counter(master_seq)))
    
    drop_opt = DropOffOptimizer(master_seq, opt_master)
    
    target_counts = {0: 8, 45: 7, -45: 7, 90: 8}
    
    try:
        new_seq, score, dropped = drop_opt.optimize_drop_with_angle_targets(target_counts)
        print(f"Result Sequence (len {len(new_seq)}): {new_seq}")
        print("Result Counts:", dict(collections.Counter(new_seq)))
        print("Result Score:", score)
        
        if new_seq == master_seq:
            print("ISSUE REPRODUCED: Returned Master Sequence (Copied Upper Part)")
        else:
            print("Successfully dropped.")
            
    except Exception as e:
        print("Dropoff Failed with error:", str(e))

if __name__ == "__main__":
    test_laminate_optimizer_8778()
    test_dropoff_8888_to_8778()
