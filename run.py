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
    print(f"Goal: {goal}")
    
    # System 1: Incubation
    final_vector, energy_history = sys1.incubate(goal, steps=40, noise_level=0.05)
    
    # System 2: Decoding
    result = sys2.verify(final_vector, goal)
    
    # Goal Refinement Check
    refinement = sys1.propose_goal_refinement(goal, result['idea'], result['anchors'])
    
    return {
        "iteration": iteration_num,
        "goal": goal,
        "idea": result['idea'],
        "anchors": result['anchors'],
        "score": result['economic_score'],
        "passed": result['passed'],
        "refinement": refinement,
        "energy": energy_history[-1]
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys1 = SubconsciousEngine(device=device)
    sys2 = ConsciousEngine(sys1)
    
    original_goal = input("Enter a Goal (e.g., 'Sustainable Food Security'): ")
    
    # Track the full journey
    trace = []
    current_goal = original_goal
    max_iterations = 3
    
    for i in range(max_iterations):
        iteration_result = run_single_iteration(sys1, sys2, current_goal, i + 1)
        trace.append(iteration_result)
        
        # Show iteration output
        print(f"\n{'='*50}")
        print("OUTPUT")
        print(f"{'='*50}")
        print(f"Idea: {iteration_result['idea'].get('idea_name', 'N/A')}")
        print(f"Description: {iteration_result['idea'].get('description', 'N/A')[:150]}...")
        print(f"Score: {iteration_result['score']}/10")
        print(f"Divergence: {iteration_result['refinement']['divergence_score']:.2f}")
        
        # Decide whether to continue
        if iteration_result['refinement']['divergent'] and i < max_iterations - 1:
            print(f"\n⚠️  GOAL DIVERGENCE DETECTED")
            print(f"Refined Goal: {iteration_result['refinement']['refined_goal']}")
            print(f"Reasoning: {iteration_result['refinement']['reasoning'][:150]}...")
            
            # Auto-accept refinement (you can change this to manual approval)
            current_goal = iteration_result['refinement']['refined_goal']
            print(f"\n→ Continuing with refined goal...")
        else:
            print(f"\n✅ Goal stable. Stopping iteration.")
            break
    
    # Final Summary
    print(f"\n{'='*50}")
    print("FULL TRACE SUMMARY")
    print(f"{'='*50}")
    print(f"Original Goal: {original_goal}")
    print(f"Iterations: {len(trace)}")
    
    if len(trace) > 1:
        print(f"Final Goal: {trace[-1]['goal']}")
        print(f"\nGoal Evolution:")
        for t in trace:
            print(f"  {t['iteration']}. {t['goal']}")
    
    print(f"\nFinal Idea: {trace[-1]['idea'].get('idea_name', 'N/A')}")
    print(f"Description: {trace[-1]['idea'].get('description', 'N/A')}")
    print(f"Final Score: {trace[-1]['score']}/10")
    
    # Save trace to file
    with open("generation_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nSaved full trace to 'generation_trace.json'")

if __name__ == "__main__":
    main()
