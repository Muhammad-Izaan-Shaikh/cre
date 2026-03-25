import torch
from sentence_transformers import util
import subprocess
import json
import re

class ConsciousEngine:
    def __init__(self, system1):
        self.system1 = system1
        self.knowledge_base = [
            "solar energy", "battery storage", "grid sharing", "energy cooperative",
            "kinetic harvesting", "thermal storage", "peer to peer energy",
            "vertical farming", "community garden", "food delivery", "meal kit",
            "urban agriculture", "hydroponics", "composting", "food waste reduction",
            "micro-finance", "crowdfunding", "group buying", "local currency",
            "barter system", "subscription box", "pay per use", "leasing model",
            "co-living", "shared housing", "modular homes", "rental platform",
            "car sharing", "bike sharing", "electric vehicles", "public transit",
            "waste management", "recycling", "upcycling", "circular economy",
            "neighborhood network", "skill sharing", "time bank", "community hub",
            "IoT sensors", "smart metering", "mobile app", "SMS service",
            "blockchain solution", "AI platform", "marketplace", "SaaS"
        ]
        self.kb_vectors = self.system1.model.encode(self.knowledge_base, convert_to_tensor=True, device=self.system1.device)

    def decode_with_llm(self, latent_vector, goal_text):
        """Use LLM to interpret the latent vector as a full idea"""
        scores = util.cos_sim(latent_vector.unsqueeze(0), self.kb_vectors)[0]
        top_5_idx = torch.topk(scores, 5).indices
        anchor_concepts = [self.knowledge_base[i] for i in top_5_idx]
        anchor_scores = [scores[i].item() for i in top_5_idx]
        
        prompt = f"""You are interpreting a semantic vector from a creative AI system.

GOAL: {goal_text}

SEMANTIC ANCHORS (the vector is positioned near these concepts):
{chr(10).join([f"- {c} (relevance: {s:.2f})" for c, s in zip(anchor_concepts, anchor_scores)])}

TASK: Generate ONE specific, actionable startup idea that:
1. Addresses the GOAL
2. Is semantically related to the ANCHORS but NOT identical to them
3. Is concrete (has a mechanism, revenue model, target user)
4. Is NOT a generic app, platform, or blockchain solution

Output ONLY valid JSON, no markdown, no explanations:
{{"idea_name": "Short name", "description": "2-3 sentences", "revenue_model": "How it makes money", "novelty_explanation": "Why this is not a cliché"}}
"""
        
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2', prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout.strip()
        idea = self._parse_json_robust(output)
        return idea, anchor_concepts

    def _parse_json_robust(self, text):
        """Try multiple strategies to extract JSON from messy LLM output"""
        
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Strategy 2: Extract JSON from markdown blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Strategy 3: Find first { and last }
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
        
        # Strategy 4: Extract fields with regex (fallback)
        idea = {
            "idea_name": self._extract_field(text, "idea_name"),
            "description": self._extract_field(text, "description"),
            "revenue_model": self._extract_field(text, "revenue_model"),
            "novelty_explanation": self._extract_field(text, "novelty_explanation")
        }
        return idea

    def _extract_field(self, text, field_name):
        """Extract a field value using regex"""
        patterns = [
            rf'"{field_name}"\s*:\s*"([^"]+)"',
            rf'"{field_name}"\s*:\s*([^\n,}}]+)',
            rf'{field_name}\s*[:=]\s*([^"\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Unknown"

    def critique(self, idea_dict, goal_text):
        """Evaluate the full idea"""
        score = 7.0
        
        desc = idea_dict.get("description", "").lower()
        revenue = idea_dict.get("revenue_model", "").lower()
        name = idea_dict.get("idea_name", "")
        
        # Penalize vagueness
        if len(desc) < 30:
            score -= 1.5
        if "blockchain" in desc:
            score -= 2.0
        if revenue == "unknown" or len(revenue) < 5:
            score -= 1.0  # Reduced penalty (was 1.5)
            
        # Reward specificity
        if name and len(name) > 3:
            score += 0.5
        if any(word in desc for word in ["subscription", "fee", "commission", "lease", "rent", "per unit"]):
            score += 1.0
        if any(word in desc for word in ["apartment", "resident", "building", "unit"]):
            score += 0.5  # Goal alignment
            
        return max(0, min(10, score))

    def verify(self, latent_vector, goal_text):
        idea, anchors = self.decode_with_llm(latent_vector, goal_text)
        econ_score = self.critique(idea, goal_text)
        return {
            "idea": idea,
            "anchors": anchors,
            "economic_score": econ_score,
            "passed": econ_score > 5.0  # Lowered threshold
        }
