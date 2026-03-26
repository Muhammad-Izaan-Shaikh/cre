from system1 import SubconsciousEngine
from system2 import ConsciousEngine
import matplotlib.pyplot as plt
import json
import torch

def run_single_iteration(sys1, sys2, goal, iteration_num):
    """Run one generation cycle"""
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*50}")
    print(f"Goal: {goal[:80]}..." if len(goal) > 80 else f"Goal: {goal}")
    
    # System 1: Incubation
    final_vector, energy_history = sys1.incubate(goal, steps=40, noise_level=0.05)
    
    # System 2: Decoding
    result = sys2.verify(final_vector, goal)
    
    # Calculate drift for logging ONLY (not stopping)
    drift = sys1.calculate_dual_anchor_divergence(
        final_vector, goal, goal  # Same goal = local drift only
    )
    
    return {
        "iteration": iteration_num,
        "goal": goal,
        "idea": result['idea'],
        "anchors": result['anchors'],
        "score": result['economic_score'],
        "passed": result['passed'],
        "drift": drift['global'],  # For logging only
        "energy": energy_history[-1],
        "vector": final_vector
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys1 = SubconsciousEngine(device=device)
    sys2 = ConsciousEngine(sys1)
    
    original_goal = input("Enter a Goal (e.g., 'Sustainable Food Security'): ")
    
    # Track the full journey
    trace = []
    current_goal = original_goal
    best_iteration = None
    best_score = 0
    
    # Dynamic iteration limits
    max_iterations = 8
    min_iterations = 2
    decline_count = 0
    
    print(f"\n{'='*50}")
    print("COGNITIVE RESONANCE ENGINE v3.0")
    print("Score-Only Trajectory Tracking")
    print(f"{'='*50}")
    
    for i in range(max_iterations):
        iteration_result = run_single_iteration(
            sys1, sys2, current_goal, i + 1
        )
        trace.append(iteration_result)
        
        # Track best iteration
        if iteration_result['score'] > best_score:
            best_score = iteration_result['score']
            best_iteration = iteration_result
            print(f"\n🏆 NEW BEST: Score {best_score}/10")
        
        # Show iteration output
        print(f"\n{'='*50}")
        print("OUTPUT")
        print(f"{'='*50}")
        print(f"Idea: {iteration_result['idea'].get('idea_name', 'N/A')}")
        print(f"Description: {iteration_result['idea'].get('description', 'N/A')[:150]}...")
        print(f"Score: {iteration_result['score']}/10")
        print(f"Drift (info only): {iteration_result['drift']:.2f}")
        
        # === SCORE-ONLY STOPPING LOGIC ===
        
        if i >= min_iterations - 1:
            # Excellence threshold
            if iteration_result['score'] >= 9.0:
                print(f"\n✅ EXCELLENCE REACHED (≥9.0). Stopping.")
                break
            
            # Compare to previous score
            if i > 0:
                prev_score = trace[-2]['score']
                curr_score = iteration_result['score']
                score_change = curr_score - prev_score
                
                if score_change >= 0.3:
                    print(f"\n📈 Score improving ({prev_score:.1f} → {curr_score:.1f}). Continuing...")
                    decline_count = 0  # Reset decline counter
                elif score_change >= -0.2:
                    print(f"\n➡️ Score stable ({prev_score:.1f} → {curr_score:.1f}). One more iteration...")
                else:
                    decline_count += 1
                    print(f"\n⚠️ Score decreased ({prev_score:.1f} → {curr_score:.1f}). Decline #{decline_count}")
                    
                    if decline_count >= 2:
                        print(f"\n🛑 CONFIRMED DECLINE. Stopping.")
                        break
                    elif score_change < -0.5:
                        print(f"\n🛑 SHARP DECLINE (>0.5). Stopping.")
                        break
        
        # Goal refinement (based on score, not drift)
        if iteration_result['score'] >= 7.0 and i < max_iterations - 1:
            refinement = sys1.propose_goal_refinement(
                original_goal, current_goal, iteration_result['idea'], iteration_result['anchors']
            )
            
            if refinement['divergent']:
                current_goal = refinement['refined_goal']
                print(f"\n🔄 Goal refined: {current_goal[:80]}...")
            else:
                print(f"\n➡️ Goal stable.")
    
    # Final Summary - Use BEST iteration
    final_output = best_iteration if best_iteration else trace[-1]
    
    print(f"\n{'='*50}")
    print("FULL TRACE SUMMARY")
    print(f"{'='*50}")
    print(f"Original Goal: {original_goal}")
    print(f"Iterations Run: {len(trace)}")
    print(f"Best Iteration: {best_iteration['iteration'] if best_iteration else 'N/A'}")
    print(f"Best Score: {best_score}/10")
    
    if len(trace) > 1:
        print(f"\nScore Trajectory:")
        for t in trace:
            marker = "🏆" if t == best_iteration else "  "
            print(f"  {marker} Iter {t['iteration']}: Score {t['score']}/10")
    
    print(f"\n🏆 FINAL IDEA (Best Iteration):")
    print(f"Name: {final_output['idea'].get('idea_name', 'N/A')}")
    print(f"Description: {final_output['idea'].get('description', 'N/A')}")
    print(f"Revenue Model: {final_output['idea'].get('revenue_model', 'N/A')}")
    print(f"Final Score: {final_output['score']}/10")
    
    # Save trace
    trace_clean = [{k: v for k, v in t.items() if k != 'vector'} for t in trace]
    with open("generation_trace.json", "w") as f:
        json.dump(trace_clean, f, indent=2)
    print(f"\nSaved full trace to 'generation_trace.json'")

if __name__ == "__main__":
    main()
