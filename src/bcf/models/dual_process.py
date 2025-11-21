"""
Dual-Process Theory Implementation

Dual-process theories distinguish between two types of cognitive processing:
- System 1: Fast, automatic, intuitive, emotional
- System 2: Slow, deliberative, analytical, effortful

Reference:
Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.
"""

from enum import Enum
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

from bcf.models.base import Behavior, BehavioralModel


class CognitiveSystem(str, Enum):
    """Cognitive processing systems."""

    SYSTEM_1 = "system_1"  # Automatic, fast
    SYSTEM_2 = "system_2"  # Deliberative, slow


class CognitiveLoad(str, Enum):
    """Cognitive load levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OVERWHELMING = "overwhelming"


class DualProcessProfile(BaseModel):
    """Profile of dual-process characteristics for a behavior."""

    system_1_strength: float = Field(
        ge=0.0, le=1.0, description="Strength of automatic processing (0-1)"
    )
    system_2_strength: float = Field(
        ge=0.0, le=1.0, description="Strength of deliberative processing (0-1)"
    )
    cognitive_load: float = Field(
        ge=0.0, le=1.0, description="Current cognitive load (0-1)"
    )
    emotional_valence: float = Field(
        ge=-1.0, le=1.0, description="Emotional association (-1=negative, 1=positive)"
    )
    habit_strength: float = Field(
        ge=0.0, le=1.0, description="Strength of habitual response (0-1)"
    )

    @property
    def dominant_system(self) -> CognitiveSystem:
        """Determine which system dominates."""
        if self.system_1_strength > self.system_2_strength:
            return CognitiveSystem.SYSTEM_1
        else:
            return CognitiveSystem.SYSTEM_2

    @property
    def load_level(self) -> CognitiveLoad:
        """Categorize cognitive load."""
        if self.cognitive_load < 0.3:
            return CognitiveLoad.LOW
        elif self.cognitive_load < 0.6:
            return CognitiveLoad.MEDIUM
        elif self.cognitive_load < 0.9:
            return CognitiveLoad.HIGH
        else:
            return CognitiveLoad.OVERWHELMING


class DualProcessModel(BehavioralModel):
    """
    Dual-Process Model for understanding behavior through System 1 and System 2.

    System 1: Automatic, fast, emotional, intuitive
    System 2: Deliberative, slow, logical, effortful

    Key insight: Under cognitive load or time pressure, System 1 dominates.
    """

    def __init__(self):
        """Initialize Dual-Process Model."""
        pass

    def predict_behavior(
        self,
        system_1_activation: float,
        system_2_activation: float,
        cognitive_load: float,
        time_pressure: float = 0.5,
    ) -> float:
        """
        Predict behavior based on dual-process activation.

        Args:
            system_1_activation: Automatic system activation (0-1)
            system_2_activation: Deliberative system activation (0-1)
            cognitive_load: Current cognitive load (0-1)
            time_pressure: Time pressure level (0-1)

        Returns:
            Predicted behavior likelihood (0-1)
        """
        # Under high cognitive load or time pressure, System 1 dominates
        load_factor = (cognitive_load + time_pressure) / 2

        # Weight System 1 more under load
        system_1_weight = 0.5 + (load_factor * 0.4)  # 0.5-0.9
        system_2_weight = 1.0 - system_1_weight  # 0.5-0.1

        return (system_1_weight * system_1_activation +
                system_2_weight * system_2_activation)

    def diagnose(self, behavior: Behavior) -> Dict[str, Any]:
        """
        Diagnose behavior using dual-process framework.

        Args:
            behavior: Behavior to analyze

        Returns:
            Dual-process diagnostic information
        """
        # Map behavior components to dual-process
        system_1_strength = self._assess_system_1(behavior)
        system_2_strength = self._assess_system_2(behavior)

        # Estimate cognitive load based on behavior complexity
        cognitive_load = 1.0 - behavior.capability.psychological

        profile = DualProcessProfile(
            system_1_strength=system_1_strength,
            system_2_strength=system_2_strength,
            cognitive_load=cognitive_load,
            emotional_valence=self._assess_emotional_valence(behavior),
            habit_strength=behavior.motivation.automatic,
        )

        # Identify cognitive biases that may be affecting behavior
        biases = self._identify_relevant_biases(behavior, profile)

        return {
            "dual_process_profile": profile.model_dump(),
            "dominant_system": profile.dominant_system,
            "cognitive_load_level": profile.load_level,
            "system_1_strength": system_1_strength,
            "system_2_strength": system_2_strength,
            "system_balance": abs(system_1_strength - system_2_strength),
            "relevant_biases": biases,
            "intervention_approach": self._recommend_approach(profile),
        }

    def recommend_interventions(self, behavior: Behavior) -> List[Dict[str, Any]]:
        """
        Recommend interventions based on dual-process analysis.

        Args:
            behavior: Target behavior

        Returns:
            List of interventions tailored to dominant cognitive system
        """
        diagnosis = self.diagnose(behavior)
        interventions = []

        profile = DualProcessProfile(**diagnosis["dual_process_profile"])

        # System 1 interventions (when automatic processing dominates)
        if profile.dominant_system == CognitiveSystem.SYSTEM_1:
            interventions.append({
                "type": "system_1_intervention",
                "target": "Automatic/Intuitive Processing",
                "rationale": "System 1 dominates - use automatic, emotional approaches",
                "strategies": [
                    "Build strong positive habits and routines",
                    "Use emotional appeals and storytelling",
                    "Leverage defaults and choice architecture",
                    "Create environmental cues and triggers",
                    "Make desired behavior the path of least resistance",
                    "Use vivid imagery and concrete examples",
                ],
                "priority": profile.system_1_strength,
            })

        # System 2 interventions (when deliberative processing possible)
        if profile.system_2_strength > 0.4 and profile.load_level != CognitiveLoad.OVERWHELMING:
            interventions.append({
                "type": "system_2_intervention",
                "target": "Deliberative/Analytical Processing",
                "rationale": "System 2 available - use rational, analytical approaches",
                "strategies": [
                    "Provide detailed information and evidence",
                    "Support goal-setting and planning",
                    "Use implementation intentions (if-then plans)",
                    "Offer decision aids and comparison tools",
                    "Encourage cost-benefit analysis",
                    "Support metacognition (thinking about thinking)",
                ],
                "priority": profile.system_2_strength,
            })

        # Cognitive load reduction interventions
        if profile.load_level in [CognitiveLoad.HIGH, CognitiveLoad.OVERWHELMING]:
            interventions.append({
                "type": "reduce_cognitive_load",
                "target": "Cognitive Load Reduction",
                "rationale": f"High cognitive load ({profile.load_level}) impairs deliberation",
                "strategies": [
                    "Simplify choices and reduce options",
                    "Break complex tasks into smaller steps",
                    "Provide clear, simple instructions",
                    "Reduce distractions and competing demands",
                    "Use checklists and external aids",
                    "Time interventions for low-load moments",
                ],
                "priority": profile.cognitive_load,
            })

        # Bias-specific interventions
        if diagnosis["relevant_biases"]:
            interventions.append({
                "type": "debiasing",
                "target": "Cognitive Bias Mitigation",
                "rationale": "Address identified cognitive biases",
                "biases": diagnosis["relevant_biases"],
                "strategies": self._get_debiasing_strategies(diagnosis["relevant_biases"]),
                "priority": 0.7,
            })

        # Sort by priority
        interventions.sort(key=lambda x: x["priority"], reverse=True)

        return interventions

    def _assess_system_1(self, behavior: Behavior) -> float:
        """Assess System 1 (automatic) processing strength."""
        return np.mean([
            behavior.motivation.automatic,
            behavior.motivation.automatic,  # Weight it more
            1.0 - behavior.capability.psychological,  # Less thinking = more automatic
        ])

    def _assess_system_2(self, behavior: Behavior) -> float:
        """Assess System 2 (deliberative) processing strength."""
        return np.mean([
            behavior.motivation.reflective,
            behavior.capability.psychological,
            behavior.capability.knowledge,
        ])

    def _assess_emotional_valence(self, behavior: Behavior) -> float:
        """Assess emotional association with behavior."""
        # Positive if high automatic motivation, negative if barriers
        base_valence = behavior.motivation.automatic * 2 - 1  # Convert 0-1 to -1 to 1

        # Reduce valence if many barriers
        barrier_penalty = len(behavior.barriers) * 0.1
        return max(-1.0, min(1.0, base_valence - barrier_penalty))

    def _identify_relevant_biases(
        self, behavior: Behavior, profile: DualProcessProfile
    ) -> List[Dict[str, str]]:
        """Identify cognitive biases likely affecting this behavior."""
        biases = []

        # Present bias (System 1)
        if profile.system_1_strength > 0.6:
            biases.append({
                "name": "Present Bias",
                "description": "Overvaluing immediate rewards over future benefits",
                "impact": "high",
            })

        # Status quo bias
        if behavior.current_frequency < behavior.desired_frequency:
            biases.append({
                "name": "Status Quo Bias",
                "description": "Preference for current state over change",
                "impact": "medium",
            })

        # Optimism bias
        if behavior.motivation.reflective > 0.7 and behavior.current_frequency < 1:
            biases.append({
                "name": "Optimism Bias",
                "description": "Overestimating likelihood of starting/maintaining behavior",
                "impact": "medium",
            })

        # Loss aversion
        if profile.emotional_valence < 0:
            biases.append({
                "name": "Loss Aversion",
                "description": "Losses loom larger than equivalent gains",
                "impact": "high",
            })

        # Availability heuristic (System 1)
        if profile.system_1_strength > 0.7:
            biases.append({
                "name": "Availability Heuristic",
                "description": "Judging by easily recalled examples",
                "impact": "medium",
            })

        return biases

    def _recommend_approach(self, profile: DualProcessProfile) -> str:
        """Recommend overall intervention approach based on profile."""
        if profile.load_level == CognitiveLoad.OVERWHELMING:
            return "Simplify drastically - reduce cognitive burden first"
        elif profile.dominant_system == CognitiveSystem.SYSTEM_1:
            return "Leverage automatic processes - habits, defaults, emotions"
        elif profile.system_2_strength > 0.7:
            return "Engage deliberative thinking - planning, analysis, reasoning"
        else:
            return "Hybrid approach - combine automatic and deliberative strategies"

    def _get_debiasing_strategies(self, biases: List[Dict[str, str]]) -> List[str]:
        """Get specific debiasing strategies."""
        strategies = []
        bias_names = [b["name"] for b in biases]

        if "Present Bias" in bias_names:
            strategies.extend([
                "Pre-commitment devices (lock in future behavior)",
                "Make future benefits more concrete and vivid",
                "Immediate small rewards for desired behavior",
            ])

        if "Status Quo Bias" in bias_names:
            strategies.extend([
                "Leverage defaults (make desired behavior default)",
                "Reduce switching costs",
                "Frame as improvement, not change",
            ])

        if "Optimism Bias" in bias_names:
            strategies.extend([
                "Implementation intentions (specific plans)",
                "Realistic goal-setting with contingencies",
                "Track actual vs. planned behavior",
            ])

        if "Loss Aversion" in bias_names:
            strategies.extend([
                "Frame in terms of losses avoided",
                "Loss-framed messaging for prevention behaviors",
                "Endowment effect (trial periods)",
            ])

        return strategies
