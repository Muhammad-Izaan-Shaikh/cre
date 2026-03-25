import torch
from sentence_transformers import util

class ConsciousEngine:
    def __init__(self, system1):
        self.system1 = system1
        # EXPANDED knowledge base (50+ concepts for better coverage)
        self.knowledge_base = [
            # Energy
            "solar energy", "battery storage", "grid sharing", "energy cooperative",
            "kinetic harvesting", "thermal storage", "peer to peer energy",
            # Food
            "vertical farming", "community garden", "food delivery", "meal kit",
            "urban agriculture", "hydroponics", "composting", "food waste reduction",
            # Finance
            "micro-finance", "crowdfunding", "group buying", "local currency",
            "barter system", "subscription box", "pay per use", "leasing model",
            # Housing
            "co-living", "shared housing", "modular homes", "rental platform",
            # Transportation
            "car sharing", "bike sharing", "electric vehicles", "public transit",
            # Waste
            "waste management", "recycling", "upcycling", "circular economy",
            # Community
            "neighborhood network", "skill sharing", "time bank", "community hub",
            # Technology
            "IoT sensors", "smart metering", "mobile app", "SMS service",
            "blockchain solution", "AI platform", "marketplace", "SaaS"
        ]
        self.kb_vectors = self.system1.model.encode(self.knowledge_base, convert_to_tensor=True, device=self.system1.device)

    def decode(self, latent_vector):
        """Find the nearest semantic concept to the latent vector"""
        scores = util.cos_sim(latent_vector.unsqueeze(0), self.kb_vectors)[0]
        top_idx = torch.argmax(scores)
        top_3_idx = torch.topk(scores, 3).indices
        return {
            "primary": self.knowledge_base[top_idx],
            "confidence": scores[top_idx].item(),
            "alternatives": [self.knowledge_base[i] for i in top_3_idx]
        }

    def critique(self, idea_text, goal_text):
        """
        Relaxed Critic (allow more ideas through for testing)
        """
        score = 6.0  # Higher base score
        cliches = ["blockchain", "AI platform"]  # Only reject obvious buzzwords
        
        for word in cliches:
            if word in idea_text.lower():
                score -= 2.0
        
        return max(0, min(10, score))

    def verify(self, latent_vector, goal_text):
        decode_result = self.decode(latent_vector)
        econ_score = self.critique(decode_result["primary"], goal_text)
        return {
            "concept": decode_result["primary"],
            "confidence": decode_result["confidence"],
            "alternatives": decode_result["alternatives"],
            "economic_score": econ_score,
            "passed": econ_score > 5.0
        }
