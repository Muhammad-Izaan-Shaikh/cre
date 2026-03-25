import torch
from sentence_transformers import util

class ConsciousEngine:
    def __init__(self, system1):
        self.system1 = system1
        # A small database of "Known Concepts" to decode vectors into text
        # In a real system, this would be a vector DB with 10k+ ideas
        self.knowledge_base = [
            "vertical farming", "food delivery app", "community garden", 
            "crypto currency", "supply chain blockchain", "subscription box",
            "micro-finance", "barter system", "group buying", "waste management",
            "solar energy", "hydroponics", "local currency", "sms service"
        ]
        self.kb_vectors = self.system1.model.encode(self.knowledge_base, convert_to_tensor=True, device=self.system1.device)

    def decode(self, latent_vector):
        """Find the nearest semantic concept to the latent vector"""
        scores = util.cos_sim(latent_vector.unsqueeze(0), self.kb_vectors)[0]
        top_idx = torch.argmax(scores)
        return self.knowledge_base[top_idx], scores[top_idx].item()

    def critique(self, idea_text, goal_text):
        """
        Simple Heuristic Critic (Replace with LLM API for production)
        Checks for cliché keywords and length.
        """
        score = 5.0 # Base score
        cliches = ["app", "blockchain", "platform", "AI"]
        
        for word in cliches:
            if word in idea_text.lower():
                score -= 1.5
        
        if len(idea_text) < 10:
            score -= 2.0
            
        return max(0, min(10, score))

    def verify(self, latent_vector, goal_text):
        concept, confidence = self.decode(latent_vector)
        econ_score = self.critique(concept, goal_text)
        return {
            "concept": concept,
            "confidence": confidence,
            "economic_score": econ_score,
            "passed": econ_score > 6.0
        }
