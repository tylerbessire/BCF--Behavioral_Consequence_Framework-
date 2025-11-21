"""
BJ Fogg Behavior Model Implementation

The Fogg Behavior Model states that behavior (B) occurs when Motivation (M),
Ability (A), and Prompt (P) converge at the same moment: B = MAP

Reference:
Fogg, B. J. (2009). A behavior model for persuasive design.
Proceedings of the 4th International Conference on Persuasive Technology.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from bcf.models.base import Behavior, BehavioralModel


class PromptType(str, Enum):
    """Types of prompts in Fogg model."""

    SPARK = "spark"  # High ability, low motivation - provide motivation
    FACILITATOR = "facilitator"  # High motivation, low ability - make it easier
    SIGNAL = "signal"  # High motivation and ability - just remind


class Prompt(BaseModel):
    """Represents a prompt/trigger for behavior."""

    prompt_type: PromptType
    timing: float = Field(
        ge=0.0, le=1.0, description="Timing appropriateness (0-1)"
    )
    salience: float = Field(ge=0.0, le=1.0, description="How noticeable (0-1)")
    relevance: float = Field(
        ge=0.0, le=1.0, description="Relevance to user (0-1)"
    )
    context_fit: float = Field(
        ge=0.0, le=1.0, description="Contextual appropriateness (0-1)"
    )

    @property
    def effectiveness(self) -> float:
        """Calculate overall prompt effectiveness."""
        return np.mean([self.timing, self.salience, self.relevance, self.context_fit])


class FoggBehaviorModel(BehavioralModel):
    """
    BJ Fogg's Behavior Model: B = MAP (Motivation × Ability × Prompt).

    Key insight: For behavior to occur, a person must have:
    1. Sufficient motivation
    2. Sufficient ability
    3. An effective prompt at the right moment
    """

    def __init__(
        self,
        motivation_threshold: float = 0.4,
        ability_threshold: float = 0.4,
        prompt_threshold: float = 0.3,
    ):
        """
        Initialize Fogg Behavior Model.

        Args:
            motivation_threshold: Minimum motivation needed (default: 0.4)
            ability_threshold: Minimum ability needed (default: 0.4)
            prompt_threshold: Minimum prompt effectiveness needed (default: 0.3)
        """
        self.motivation_threshold = motivation_threshold
        self.ability_threshold = ability_threshold
        self.prompt_threshold = prompt_threshold

    def predict_behavior(
        self,
        motivation: float,
        ability: float,
        prompt_effectiveness: float,
        use_threshold: bool = True,
    ) -> float:
        """
        Predict likelihood of behavior occurring.

        Args:
            motivation: Motivation level (0-1)
            ability: Ability level (0-1)
            prompt_effectiveness: Prompt effectiveness (0-1)
            use_threshold: If True, apply thresholds; else use continuous model

        Returns:
            Probability of behavior (0-1)
        """
        if use_threshold:
            # Binary threshold model: all must exceed thresholds
            if (
                motivation >= self.motivation_threshold
                and ability >= self.ability_threshold
                and prompt_effectiveness >= self.prompt_threshold
            ):
                # Above threshold: probability increases with each component
                return motivation * ability * prompt_effectiveness
            else:
                # Below threshold: low probability
                return 0.1 * motivation * ability * prompt_effectiveness
        else:
            # Continuous multiplicative model
            return motivation * ability * prompt_effectiveness

    def diagnose(self, behavior: Behavior, prompt: Optional[Prompt] = None) -> Dict[str, Any]:
        """
        Diagnose behavior using Fogg's MAP framework.

        Args:
            behavior: Behavior to analyze
            prompt: Optional prompt information

        Returns:
            Diagnostic information
        """
        motivation = behavior.motivation.overall
        ability = behavior.capability.overall  # Map capability to ability

        prompt_effectiveness = prompt.effectiveness if prompt else 0.0

        # Determine what's limiting behavior
        limiting_factors = []
        if motivation < self.motivation_threshold:
            limiting_factors.append("motivation")
        if ability < self.ability_threshold:
            limiting_factors.append("ability")
        if prompt_effectiveness < self.prompt_threshold:
            limiting_factors.append("prompt")

        # Determine recommended prompt type
        if motivation < ability:
            recommended_prompt = PromptType.SPARK
        elif ability < motivation:
            recommended_prompt = PromptType.FACILITATOR
        else:
            recommended_prompt = PromptType.SIGNAL

        # Calculate behavior likelihood
        behavior_likelihood = self.predict_behavior(
            motivation, ability, prompt_effectiveness
        )

        return {
            "motivation": motivation,
            "ability": ability,
            "prompt_effectiveness": prompt_effectiveness,
            "limiting_factors": limiting_factors,
            "recommended_prompt_type": recommended_prompt,
            "behavior_likelihood": behavior_likelihood,
            "above_activation_threshold": behavior_likelihood > 0.5,
            "simplicity_factors": self._analyze_simplicity(behavior),
        }

    def recommend_interventions(self, behavior: Behavior) -> List[Dict[str, Any]]:
        """
        Recommend interventions based on Fogg model.

        Args:
            behavior: Target behavior

        Returns:
            List of intervention recommendations
        """
        diagnosis = self.diagnose(behavior)
        interventions = []

        # Motivation interventions
        if diagnosis["motivation"] < self.motivation_threshold:
            interventions.append({
                "type": "increase_motivation",
                "target": "Motivation",
                "strategies": [
                    "Connect to core values and desires",
                    "Provide social proof (others doing it)",
                    "Highlight immediate benefits",
                    "Create fear of loss (FOMO)",
                    "Use aspirational messaging",
                ],
                "priority": 1.0 - diagnosis["motivation"],
                "prompt_type": PromptType.SPARK,
            })

        # Ability interventions (Simplicity)
        if diagnosis["ability"] < self.ability_threshold:
            simplicity_factors = diagnosis["simplicity_factors"]

            interventions.append({
                "type": "increase_ability",
                "target": "Ability/Simplicity",
                "strategies": [
                    f"Reduce time required (current concern: {simplicity_factors['time']})",
                    f"Reduce financial cost (current concern: {simplicity_factors['money']})",
                    f"Reduce physical effort (current concern: {simplicity_factors['physical_effort']})",
                    f"Simplify mental process (current concern: {simplicity_factors['brain_cycles']})",
                    "Make it routine/habitual",
                    "Provide clear step-by-step guidance",
                ],
                "priority": 1.0 - diagnosis["ability"],
                "prompt_type": PromptType.FACILITATOR,
                "simplicity_breakdown": simplicity_factors,
            })

        # Prompt interventions
        if diagnosis["prompt_effectiveness"] < self.prompt_threshold:
            interventions.append({
                "type": "improve_prompts",
                "target": "Prompts/Triggers",
                "strategies": [
                    "Deliver at optimal time (when M&A are high)",
                    "Make prompts more noticeable",
                    "Increase personal relevance",
                    "Use contextual triggers",
                    "Implement implementation intentions (if-then rules)",
                ],
                "priority": 1.0 - diagnosis["prompt_effectiveness"],
                "recommended_prompt_type": diagnosis["recommended_prompt_type"],
            })

        # Sort by priority
        interventions.sort(key=lambda x: x["priority"], reverse=True)

        return interventions

    def _analyze_simplicity(self, behavior: Behavior) -> Dict[str, float]:
        """
        Analyze simplicity across Fogg's 6 simplicity factors.

        The 6 factors that affect simplicity:
        1. Time
        2. Money
        3. Physical effort
        4. Brain cycles (mental effort)
        5. Social deviance
        6. Non-routine
        """
        # Extract from barriers if available
        barriers = [b.lower() for b in behavior.barriers]

        return {
            "time": 1.0 - behavior.opportunity.temporal,
            "money": 1.0 - behavior.opportunity.physical * 0.5,  # Proxy
            "physical_effort": 1.0 - behavior.capability.physical,
            "brain_cycles": 1.0 - behavior.capability.psychological,
            "social_deviance": 1.0
            - behavior.opportunity.social,  # How socially acceptable
            "non_routine": 0.8 if "new" in barriers or "unfamiliar" in barriers else 0.3,
        }

    def calculate_activation_threshold(
        self, motivation: float, ability: float
    ) -> float:
        """
        Calculate the activation threshold (minimum prompt needed).

        Higher motivation and ability = lower prompt threshold needed.

        Args:
            motivation: Motivation level (0-1)
            ability: Ability level (0-1)

        Returns:
            Minimum prompt effectiveness needed for behavior
        """
        # Inverse relationship: high M&A means low prompt needed
        return max(0.0, 1.0 - (motivation * ability))

    def optimize_prompt_timing(
        self, motivation_pattern: List[float], ability_pattern: List[float]
    ) -> int:
        """
        Find optimal time to deliver prompt based on M&A patterns.

        Args:
            motivation_pattern: Motivation levels over time
            ability_pattern: Ability levels over time

        Returns:
            Index of optimal timing
        """
        if len(motivation_pattern) != len(ability_pattern):
            raise ValueError("Motivation and ability patterns must have same length")

        # Optimal time is when M × A is highest
        combined = [m * a for m, a in zip(motivation_pattern, ability_pattern)]
        return int(np.argmax(combined))

    def fogg_curve_position(self, motivation: float, ability: float) -> str:
        """
        Determine position relative to Fogg's behavior activation curve.

        Args:
            motivation: Motivation level (0-1)
            ability: Ability level (0-1)

        Returns:
            Description of position relative to activation threshold
        """
        # Simple linear activation threshold for illustration
        # In practice, this would be a curved threshold
        threshold_line = 0.8 - ability * 0.5

        if motivation > threshold_line:
            return "Above activation threshold - will act with appropriate prompt"
        elif motivation > threshold_line * 0.7:
            return "Near activation threshold - close to acting"
        else:
            return "Below activation threshold - unlikely to act"
