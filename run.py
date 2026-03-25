from system1 import SubconsciousEngine
from system2 import ConsciousEngine
import matplotlib.pyplot as plt
import torch

def main():
    # 1. Initialize Architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys1 = SubconsciousEngine(device=device)
    sys2 = ConsciousEngine(sys1)
    
    # 2. User Input
    goal = input("Enter a Goal (e.g., 'Sustainable Food Security'): ")
    
    # 3. Run System 1 (Incubation)
    final_vector, energy_history = sys1.incubate(goal, steps=50, noise_level=0.2)
    
    # 4. Run System 2 (Verification)
    result = sys2.verify(final_vector, goal)
    
    # 5. Output
    print("\n" + "="*30)
    print("COGNITIVE RESONANCE OUTPUT")
    print("="*30)
    print(f"Goal: {goal}")
    print(f"Emergent Concept: {result['concept']}")
    print(f"Semantic Confidence: {result['confidence']:.2f}")
    print(f"Economic Feasibility: {result['economic_score']}/10")
    print(f"Status: {'ACCEPTED' if result['passed'] else 'REJECTED'}")
    print("="*30 + "\n")
    
    # 6. Visualization
    plt.plot(energy_history)
    plt.title("Energy Minimization During Incubation")
    plt.xlabel("Step")
    plt.ylabel("Energy (Lower is Better)")
    plt.savefig("energy_plot.png")
    print("Saved energy plot to 'energy_plot.png'")

if __name__ == "__main__":
    main()
