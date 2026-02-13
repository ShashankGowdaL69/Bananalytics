import numpy as np

class BananaIntelligence:
    def __init__(self):
        # 1. DEFINE THE LABELS (Must match your training output!)
        # Your training said: {'overripe': 0, 'ripe': 1, 'rotten': 2, 'unripe': 3}
        self.labels = ["Overripe", "Ripe", "Rotten", "Unripe"]

        # 2. THE KNOWLEDGE BASE
        self.knowledge = {
            "Unripe": {
                "desc": "Green and firm. Not ready yet.",
                "days": "3-6 days",
                "use": "Store at room temperature. Don't refrigerate yet.",
                "risk": "Low",
                "nutrition": "High in resistant starch (prebiotic fiber).",
                "color": "#90EE90" # Light Green
            },
            "Ripe": {
                "desc": "Perfectly yellow. Sweet and creamy.",
                "days": "1-2 days",
                "use": "Eat fresh, fruit salads, or refrigerate to keep.",
                "risk": "Moderate",
                "nutrition": "High in antioxidants and natural sugars.",
                "color": "#FFD700" # Gold
            },
            "Overripe": {
                "desc": "Brown spots. Very soft and sweet.",
                "days": "0-1 days",
                "use": "Banana bread, smoothies, pancakes (Do not waste!).",
                "risk": "High",
                "nutrition": "Maximum sugar content, easy to digest.",
                "color": "#DAA520" # Goldenrod
            },
            "Rotten": {
                "desc": "Black and mushy. Unsafe.",
                "days": "0 days",
                "use": "Compost or discard immediately.",
                "risk": "Critical",
                "nutrition": "Nutrient degradation.",
                "color": "#000000" # Black
            }
        }

    def analyze_prediction(self, prediction_array):
        """
        Takes the raw model output (e.g., [[0.1, 0.8, 0.0, 0.1]])
        and returns a human-readable dictionary.
        """
        # Convert prediction to simple list
        probs = prediction_array[0] 
        
        # Get the index of the highest probability
        max_index = np.argmax(probs)
        confidence = probs[max_index]
        label = self.labels[max_index]
        
        # Get the static info for this label
        info = self.knowledge[label]

        # --- 7️⃣ CALCULATE RIPENESS SCORE (0-100) ---
        # Weights: Unripe(25), Ripe(50), Overripe(75), Rotten(100)
        # Note: We must map these weights to the specific indices [0, 1, 2, 3]
        # Index 0 (Overripe) -> 75
        # Index 1 (Ripe)     -> 50
        # Index 2 (Rotten)   -> 100
        # Index 3 (Unripe)   -> 25
        weights = [75, 50, 100, 25] 
        
        # Calculate dot product
        raw_score = sum(p * w for p, w in zip(probs, weights))
        
        # --- 8️⃣ VISUAL BAR ---
        filled = int(raw_score / 10)
        bar = "█" * filled + "░" * (10 - filled)

        return {
            "label": label,
            "confidence": f"{confidence*100:.1f}%",
            "score": int(raw_score),
            "bar": bar,
            "days_remaining": info['days'],
            "action": info['use'],
            "risk": info['risk'],
            "nutrition": info['nutrition'],
            "color": info['color']
        }