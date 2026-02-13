
from tusas.core.laminate_optimizer import LaminateOptimizer
from tusas.core.dropoff_optimizer import DropOffOptimizer
from collections import Counter
import random

def test_multizone_fail():
    print("--- Simulating MultiZone Failure ---")
    
    # Zone 1 (40 ply): 16, 8, 8, 8
    # We simulate a completed Zone 2 first
    
    # Zone 2 Target: 12, 8, 8, 8 (36 ply)
    z2_target = {0: 12, 90: 8, 45: 8, -45: 8}
    z2_opt = LaminateOptimizer(z2_target)
    # Create a synthetic Zone 2 sequence that is symmetric
    z2_seq = z2_opt._create_symmetric_individual() 
    print(f"Zone 2 Seq (len {len(z2_seq)}): {dict(Counter(z2_seq))}")
    
    # Zone 3 Target: 8, 7, 7, 8 (30 ply)
    # This is the problematic one
    z3_target = {0: 8, 90: 8, 45: 7, -45: 7}
    print(f"Zone 3 Target: {z3_target}")
    
    drop_opt = DropOffOptimizer(z2_seq, z2_opt)
    
    try:
        new_seq, score, dropped = drop_opt.optimize_drop_with_angle_targets(z3_target)
        print(f"SUCCESS: Zone 3 generated with len {len(new_seq)}")
        print(f"Counts: {dict(Counter(new_seq))}")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multizone_fail()
