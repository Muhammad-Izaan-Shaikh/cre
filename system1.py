import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class SubconsciousEngine:
    def __init__(self, device='cpu'):
        self.device = device
        print("Loading Semantic Latent Space (System 1)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.dim = 384
        
        # Define "Cliché Centroid" (Vectors we want to avoid)
        cliches = ["startup app", "blockchain solution", "AI platform", "marketplace", "SaaS"]
        self.cliche_vectors = self.model.encode(cliches, convert_to_tensor=True, device=device)
        self.cliche_centroid = torch.mean(self.cliche_vectors, dim=0)
        
    def embed(self, text):
        return self.model.encode(text, convert_to_tensor=True, device=self.device)
    
    def calculate_energy(self, vector, goal_vector, lambda_repel=0.5):
        """
        Energy Function:
        1. Minimize distance to Goal (Alignment)
        2. Maximize distance from Cliché Centroid (Novelty/Inversion)
        """
        # Alignment Energy (Lower is better)
        align_loss = 1 - torch.cosine_similarity(vector.unsqueeze(0), goal_vector.unsqueeze(0))[0]
        
        # Repulsion Energy (Higher distance from cliché is better -> Lower Energy)
        repel_loss = torch.cosine_similarity(vector.unsqueeze(0), self.cliche_centroid.unsqueeze(0))[0]
        
        # Total Energy
        energy = align_loss - (lambda_repel * repel_loss)
        return energy

    def incubate(self, goal_text, steps=50, noise_level=0.1, lambda_repel=0.5):
        """
        The Dreaming Loop:
        Starts at Goal, adds noise, iteratively minimizes Energy.
        """
        print(f"Incubating goal: '{goal_text}'...")
        goal_vec = self.embed(goal_text)
        
        # Initialize current vector with gradient tracking
        current_vec = goal_vec.clone().detach()
        current_vec.requires_grad = True
        
        history = []
        optimizer = torch.optim.Adam([current_vec], lr=0.05)
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Calculate Energy BEFORE adding noise (preserve gradient graph)
            energy = self.calculate_energy(current_vec, goal_vec, lambda_repel)
            
            # Backprop to find direction of lower energy
            energy.backward()
            
            # Manual gradient step (gives us more control)
            with torch.no_grad():
                # Apply gradient
                current_vec -= optimizer.param_groups[0]['lr'] * current_vec.grad
                
                # Add Stochastic Noise (Simulating Subconscious Drift)
                if i > 0:
                    noise = torch.randn_like(current_vec) * noise_level
                    current_vec += noise
                
                # Clear gradients
                current_vec.grad = None
                
                # Normalize vector (keep on hypersphere)
                current_vec = torch.nn.functional.normalize(current_vec, dim=0)
                
                # Re-enable gradients for next iteration
                current_vec.requires_grad = True
            
            history.append(energy.item())
            
            if i % 10 == 0:
                print(f"Step {i}: Energy = {energy.item():.4f}")
        
        return current_vec.detach(), history
