"""Nudge design tools based on behavioral economics and choice architecture."""

from typing import Dict, List

from pydantic import BaseModel, Field


class Nudge(BaseModel):
    """Represents a behavioral nudge."""

    name: str
    category: str  # default, framing, salience, social, etc.
    description: str
    ease_of_implementation: float = Field(ge=0.0, le=1.0)
    expected_effectiveness: float = Field(ge=0.0, le=1.0)
    ethical_score: float = Field(ge=0.0, le=1.0)  # Transparency and autonomy preservation


class NudgeDesigner:
    """Designs behavioral nudges using choice architecture principles."""

    NUDGE_CATALOG = [
        {
            "name": "Default Option",
            "category": "defaults",
            "description": "Set desired behavior as default (opt-out rather than opt-in)",
            "ease": 0.8,
            "effectiveness": 0.9,
            "ethical": 0.7,
        },
        {
            "name": "Social Proof",
            "category": "social",
            "description": "Show what others are doing (e.g., '90% of people choose...')",
            "ease": 0.9,
            "effectiveness": 0.7,
            "ethical": 0.8,
        },
        {
            "name": "Loss Framing",
            "category": "framing",
            "description": "Frame in terms of losses avoided rather than gains",
            "ease": 0.9,
            "effectiveness": 0.6,
            "ethical": 0.9,
        },
        {
            "name": "Salience",
            "category": "salience",
            "description": "Make desired option more visible and attention-grabbing",
            "ease": 0.8,
            "effectiveness": 0.6,
            "ethical": 0.8,
        },
        {
            "name": "Commitment Device",
            "category": "commitment",
            "description": "Enable pre-commitment to future behavior",
            "ease": 0.6,
            "effectiveness": 0.7,
            "ethical": 0.9,
        },
    ]

    def __init__(self):
        """Initialize nudge designer."""
        pass

    def recommend_nudges(
        self, target_behavior: str, context: Dict[str, any] = None, top_k: int = 3
    ) -> List[Nudge]:
        """
        Recommend nudges for a target behavior.

        Args:
            target_behavior: Description of target behavior
            context: Context information
            top_k: Number of nudges to recommend

        Returns:
            List of recommended nudges
        """
        nudges = [
            Nudge(
                name=n["name"],
                category=n["category"],
                description=n["description"],
                ease_of_implementation=n["ease"],
                expected_effectiveness=n["effectiveness"],
                ethical_score=n["ethical"],
            )
            for n in self.NUDGE_CATALOG
        ]

        # Sort by combined score (effectiveness * ease * ethical)
        nudges.sort(
            key=lambda n: n.expected_effectiveness * n.ease_of_implementation * n.ethical_score,
            reverse=True,
        )

        return nudges[:top_k]
