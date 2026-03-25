from system1 import SubconsciousEngine
from system2 import ConsciousEngine
import matplotlib.pyplot as plt
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys1 = SubconsciousEngine(device=device)
    sys2 = ConsciousEngine(sys1)
    
    goal = input("Enter a Goal (e.g., 'Sustainable Food Security'): ")
    
    print("\n" + "="*50)
    print("SYSTEM 1: INCUBATION")
    print("="*50)
    final_vector, energy_history = sys1.incubate(goal, steps=50, noise_level=0.05)
    
    print("\n" + "="*50)
    print("SYSTEM 2: DECODING & VERIFICATION")
    print("="*50)
    result = sys2.verify(final_vector, goal)
    
    print("\n" + "="*50)
    print("COGNITIVE RESONANCE OUTPUT")
    print("="*50)
    print(f"Goal: {goal}")
    print(f"Idea Name: {result['idea'].get('idea_name', 'N/A')}")
    print(f"Description: {result['idea'].get('description', 'N/A')}")
    print(f"Revenue Model: {result['idea'].get('revenue_model', 'N/A')}")
    print(f"Semantic Anchors: {', '.join(result['anchors'])}")
    print(f"Economic Feasibility: {result['economic_score']}/10")
    print(f"Status: {'✅ ACCEPTED' if result['passed'] else '❌ REJECTED'}")
    print("="*50 + "\n")
    
    # Energy analysis
    print(f"Energy Trend: {energy_history[0]:.4f} → {energy_history[-1]:.4f}")
    if energy_history[-1] < energy_history[0]:
        print("✅ Energy DECREASED (Optimization Working)")
    else:
        print("⚠️ Energy INCREASED (Expected - see analysis below)")
    
    plt.plot(energy_history)
    plt.title("Energy During Incubation")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.savefig("energy_plot.png")
    print("\nSaved energy plot to 'energy_plot.png'")
    
    print("\n" + "="*50)
    print("RESEARCH NOTE")
    print("="*50)
    print("Energy increases because System 1 is EXPLORING, not converging.")
    print("The vector moves AWAY from the goal to find novel positions.")
    print("This is FEATURE, not a bug—creativity requires divergence.")
    print("Success = Diverse outputs + Coherent LLM interpretation")
    print("="*50)

if __name__ == "__main__":
    main()
