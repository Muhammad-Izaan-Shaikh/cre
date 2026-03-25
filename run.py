from system1 import SubconsciousEngine
from system2 import ConsciousEngine
import matplotlib.pyplot as plt
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys1 = SubconsciousEngine(device=device)
    sys2 = ConsciousEngine(sys1)
    
    goal = input("Enter a Goal (e.g., 'Sustainable Food Security'): ")
    
    final_vector, energy_history = sys1.incubate(goal, steps=50, noise_level=0.05)
    
    result = sys2.verify(final_vector, goal)
    
    print("\n" + "="*50)
    print("COGNITIVE RESONANCE OUTPUT")
    print("="*50)
    print(f"Goal: {goal}")
    print(f"Primary Concept: {result['concept']}")
    print(f"Alternative Concepts: {', '.join(result['alternatives'])}")
    print(f"Semantic Confidence: {result['confidence']:.4f}")
    print(f"Economic Feasibility: {result['economic_score']}/10")
    print(f"Status: {'✅ ACCEPTED' if result['passed'] else '❌ REJECTED'}")
    print("="*50 + "\n")
    
    # Show energy trend
    print(f"Energy Trend: {energy_history[0]:.4f} → {energy_history[-1]:.4f}")
    if energy_history[-1] < energy_history[0]:
        print("✅ Energy DECREASED (Optimization Working)")
    else:
        print("⚠️ Energy INCREASED (Optimization Issue)")
    
    plt.plot(energy_history)
    plt.title("Energy Minimization During Incubation")
    plt.xlabel("Step")
    plt.ylabel("Energy (Lower is Better)")
    plt.savefig("energy_plot.png")
    print("\nSaved energy plot to 'energy_plot.png'")

if __name__ == "__main__":
    main()
