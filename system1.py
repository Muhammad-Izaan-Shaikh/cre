import torch
import numpy as np
import subprocess
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
    
    def calculate_energy(self, vector, goal_vector, lambda_repel=0.3):
        """
        FIXED Energy Function:
        1. Minimize distance to Goal (Alignment) - PRIMARY
        2. Maximize distance from Cliché Centroid (Novelty) - SECONDARY
        3. Keep vector on valid semantic manifold
        """
        # Alignment Energy (Lower is better) - Weight: 1.0
        align_loss = 1 - torch.cosine_similarity(vector.unsqueeze(0), goal_vector.unsqueeze(0))[0]
        
        # Repulsion Energy - Weight: 0.3 (less aggressive)
        # We want to AVOID clichés, so HIGH similarity = HIGH energy (bad)
        repel_loss = torch.cosine_similarity(vector.unsqueeze(0), self.cliche_centroid.unsqueeze(0))[0]
        repel_penalty = torch.relu(repel_loss - 0.5)  # Only penalize if too close to cliché
        
        # Total Energy (Alignment is primary, novelty is secondary constraint)
        energy = align_loss + (lambda_repel * repel_penalty)
        return energy

    def incubate(self, goal_text, steps=50, noise_level=0.05, lambda_repel=0.3):
        """
        The Dreaming Loop:
        Starts at Goal, adds SMALL noise, iteratively minimizes Energy.
        """
        print(f"Incubating goal: '{goal_text}'...")
        goal_vec = self.embed(goal_text)
        
        # Initialize current vector with gradient tracking
        current_vec = goal_vec.clone().detach()
        current_vec.requires_grad = True
        
        history = []
        optimizer = torch.optim.Adam([current_vec], lr=0.02)  # Reduced learning rate
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Calculate Energy
            energy = self.calculate_energy(current_vec, goal_vec, lambda_repel)
            
            # Backprop
            energy.backward()
            
            # Manual gradient step
            with torch.no_grad():
                # Apply gradient
                current_vec -= optimizer.param_groups[0]['lr'] * current_vec.grad
                
                # Add SMALL Stochastic Noise (reduced from 0.2 to 0.05)
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
    
    def propose_goal_refinement(self, original_goal, idea_dict, anchors):
        """
        Compare generated idea to original goal and propose refinement.
        """
        original_vec = self.embed(original_goal)
        
        # Embed the generated idea description
        idea_text = f"{idea_dict.get('idea_name', '')} {idea_dict.get('description', '')}"
        idea_vec = self.embed(idea_text)
        
        # Calculate divergence
        similarity = torch.cosine_similarity(original_vec.unsqueeze(0), idea_vec.unsqueeze(0))[0].item()
        divergence = 1 - similarity
        
        # Determine if refinement is warranted
        if divergence > 0.35:
            # Ask LLM to articulate the refinement - SIMPLER PROMPT
            prompt = f"""Original Goal: {original_goal}
Generated Idea: {idea_text}
Anchors: {', '.join(anchors)}
Divergence: {divergence:.2f}

What deeper need does this idea address that the original goal missed?
Propose a refined goal in ONE sentence (max 20 words).

Format:
REFINED_GOAL: [your refined goal]
REASONING: [your reasoning]
"""
            result = subprocess.run(
                ['ollama', 'run', 'llama3.2', prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse with simple string extraction (more robust than JSON)
            output = result.stdout.strip()
            refined_goal = self._extract_line(output, "REFINED_GOAL:")
            reasoning = self._extract_line(output, "REASONING:")
            
            if refined_goal and len(refined_goal) > 10:
                return {
                    "divergent": True,
                    "divergence_score": divergence,
                    "refined_goal": refined_goal,
                    "reasoning": reasoning or "Goal refinement based on semantic drift"
                }
            else:
                # Fallback: use divergence alone
                return {
                    "divergent": True,
                    "divergence_score": divergence,
                    "refined_goal": f"{original_goal} (explored via {anchors[0]})",
                    "reasoning": f"High divergence ({divergence:.2f}) suggests alternative framing"
                }
        else:
            return {
                "divergent": False,
                "divergence_score": divergence,
                "refined_goal": original_goal,
                "reasoning": "Idea aligns well with original goal"
            }
    
    def _extract_line(self, text, prefix):
        """Extract value after a prefix like 'REFINED_GOAL:'"""
        for line in text.split('\n'):
            if line.startswith(prefix):
                return line.replace(prefix, '').strip()
        return None
