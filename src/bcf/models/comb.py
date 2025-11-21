"""
COM-B Model Implementation

The COM-B model (Michie et al., 2011) posits that behavior (B) is determined by
three components: Capability (C), Opportunity (O), and Motivation (M).

Reference:
Michie, S., van Stralen, M. M., & West, R. (2011). The behaviour change wheel:
A new method for characterising and designing behaviour change interventions.
Implementation Science, 6(1), 42.
"""

from typing import Any, Dict, List

import numpy as np

from bcf.models.base import Behavior, BehavioralModel, Capability, Motivation, Opportunity


class COMBModel(BehavioralModel):
    """
    Capability-Opportunity-Motivation-Behavior (COM-B) Model.

    The COM-B model is a comprehensive framework for understanding behavior.
    All three components (Capability, Opportunity, Motivation) must be present
    for behavior to occur.
    """

    def __init__(
        self,
        capability_weight: float = 0.33,
        opportunity_weight: float = 0.33,
        motivation_weight: float = 0.34,
    ):
        """
        Initialize COM-B model.

        Args:
            capability_weight: Weight for capability component (default: 0.33)
            opportunity_weight: Weight for opportunity component (default: 0.33)
            motivation_weight: Weight for motivation component (default: 0.34)
        """
        if not np.isclose(capability_weight + opportunity_weight + motivation_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")

        self.capability_weight = capability_weight
        self.opportunity_weight = opportunity_weight
        self.motivation_weight = motivation_weight

    def predict_behavior(
        self,
        capability: Capability,
        opportunity: Opportunity,
        motivation: Motivation,
        interaction_effect: bool = True,
    ) -> float:
        """
        Predict likelihood of behavior occurring based on COM-B components.

        Args:
            capability: Capability component
            opportunity: Opportunity component
            motivation: Motivation component
            interaction_effect: If True, uses multiplicative model (all needed);
                              If False, uses additive weighted model

        Returns:
            Probability of behavior (0-1)
        """
        if interaction_effect:
            # Multiplicative model: all components necessary
            return (
                capability.overall ** self.capability_weight
                * opportunity.overall ** self.opportunity_weight
                * motivation.overall ** self.motivation_weight
            )
        else:
            # Additive weighted model
            return (
                self.capability_weight * capability.overall
                + self.opportunity_weight * opportunity.overall
                + self.motivation_weight * motivation.overall
            )

    def diagnose(self, behavior: Behavior) -> Dict[str, Any]:
        """
        Diagnose which COM-B components are limiting behavior.

        Args:
            behavior: Behavior to analyze

        Returns:
            Diagnostic information including limiting factors and recommendations
        """
        components = {
            "capability": behavior.capability.overall,
            "opportunity": behavior.opportunity.overall,
            "motivation": behavior.motivation.overall,
        }

        # Identify limiting factors (below threshold)
        threshold = 0.6
        limiting_factors = {
            name: score for name, score in components.items() if score < threshold
        }

        # Find primary barrier (lowest score)
        primary_barrier = min(components.items(), key=lambda x: x[1])

        # Calculate detailed sub-component scores
        detailed_scores = {
            "capability": {
                "physical": behavior.capability.physical,
                "psychological": behavior.capability.psychological,
                "knowledge": behavior.capability.knowledge,
                "skills": behavior.capability.skills,
            },
            "opportunity": {
                "physical": behavior.opportunity.physical,
                "social": behavior.opportunity.social,
                "environmental": behavior.opportunity.environmental,
                "temporal": behavior.opportunity.temporal,
            },
            "motivation": {
                "reflective": behavior.motivation.reflective,
                "automatic": behavior.motivation.automatic,
                "intrinsic": behavior.motivation.intrinsic,
                "extrinsic": behavior.motivation.extrinsic,
            },
        }

        return {
            "component_scores": components,
            "limiting_factors": limiting_factors,
            "primary_barrier": {
                "component": primary_barrier[0],
                "score": primary_barrier[1],
            },
            "detailed_scores": detailed_scores,
            "behavior_likelihood": self.predict_behavior(
                behavior.capability, behavior.opportunity, behavior.motivation
            ),
            "intervention_priority": self._prioritize_interventions(components),
        }

    def recommend_interventions(self, behavior: Behavior) -> List[Dict[str, Any]]:
        """
        Recommend interventions based on COM-B diagnosis.

        Args:
            behavior: Target behavior

        Returns:
            List of intervention recommendations ordered by priority
        """
        diagnosis = self.diagnose(behavior)
        interventions = []

        # Capability interventions
        if diagnosis["component_scores"]["capability"] < 0.6:
            cap_details = diagnosis["detailed_scores"]["capability"]

            if cap_details["physical"] < 0.6:
                interventions.append({
                    "type": "capability_physical",
                    "target": "Physical capability",
                    "strategies": [
                        "Physical training and practice",
                        "Gradual progression (start small)",
                        "Adaptive equipment or tools",
                    ],
                    "priority": 1.0 - cap_details["physical"],
                })

            if cap_details["psychological"] < 0.6:
                interventions.append({
                    "type": "capability_psychological",
                    "target": "Psychological capability",
                    "strategies": [
                        "Cognitive behavioral techniques",
                        "Mental rehearsal and visualization",
                        "Stress management training",
                    ],
                    "priority": 1.0 - cap_details["psychological"],
                })

            if cap_details["knowledge"] < 0.6:
                interventions.append({
                    "type": "capability_knowledge",
                    "target": "Knowledge",
                    "strategies": [
                        "Educational materials and resources",
                        "Expert guidance and coaching",
                        "Peer learning opportunities",
                    ],
                    "priority": 1.0 - cap_details["knowledge"],
                })

            if cap_details["skills"] < 0.6:
                interventions.append({
                    "type": "capability_skills",
                    "target": "Skills",
                    "strategies": [
                        "Skill-building workshops",
                        "Deliberate practice with feedback",
                        "Mentorship programs",
                    ],
                    "priority": 1.0 - cap_details["skills"],
                })

        # Opportunity interventions
        if diagnosis["component_scores"]["opportunity"] < 0.6:
            opp_details = diagnosis["detailed_scores"]["opportunity"]

            if opp_details["physical"] < 0.6:
                interventions.append({
                    "type": "opportunity_physical",
                    "target": "Physical opportunity",
                    "strategies": [
                        "Improve access to resources/facilities",
                        "Reduce physical barriers",
                        "Create enabling environments",
                    ],
                    "priority": 1.0 - opp_details["physical"],
                })

            if opp_details["social"] < 0.6:
                interventions.append({
                    "type": "opportunity_social",
                    "target": "Social opportunity",
                    "strategies": [
                        "Build social support networks",
                        "Establish supportive social norms",
                        "Create accountability partnerships",
                    ],
                    "priority": 1.0 - opp_details["social"],
                })

            if opp_details["temporal"] < 0.6:
                interventions.append({
                    "type": "opportunity_temporal",
                    "target": "Time availability",
                    "strategies": [
                        "Time management strategies",
                        "Schedule optimization",
                        "Reduce competing time demands",
                    ],
                    "priority": 1.0 - opp_details["temporal"],
                })

        # Motivation interventions
        if diagnosis["component_scores"]["motivation"] < 0.6:
            mot_details = diagnosis["detailed_scores"]["motivation"]

            if mot_details["reflective"] < 0.6:
                interventions.append({
                    "type": "motivation_reflective",
                    "target": "Reflective motivation",
                    "strategies": [
                        "Goal-setting and planning",
                        "Clarify personal values alignment",
                        "Cost-benefit analysis exercises",
                        "Implementation intentions (if-then plans)",
                    ],
                    "priority": 1.0 - mot_details["reflective"],
                })

            if mot_details["automatic"] < 0.6:
                interventions.append({
                    "type": "motivation_automatic",
                    "target": "Automatic motivation",
                    "strategies": [
                        "Habit formation techniques",
                        "Emotional association building",
                        "Positive reinforcement",
                        "Make it enjoyable/rewarding",
                    ],
                    "priority": 1.0 - mot_details["automatic"],
                })

            if mot_details["intrinsic"] < 0.6:
                interventions.append({
                    "type": "motivation_intrinsic",
                    "target": "Intrinsic motivation",
                    "strategies": [
                        "Connect to personal values",
                        "Emphasize autonomy and choice",
                        "Build competence and mastery",
                        "Foster sense of purpose",
                    ],
                    "priority": 1.0 - mot_details["intrinsic"],
                })

        # Sort by priority
        interventions.sort(key=lambda x: x["priority"], reverse=True)

        return interventions

    def _prioritize_interventions(self, components: Dict[str, float]) -> str:
        """Determine intervention priority based on component scores."""
        min_component = min(components.items(), key=lambda x: x[1])

        if min_component[1] < 0.3:
            return f"Critical: Address {min_component[0]} immediately"
        elif min_component[1] < 0.6:
            return f"High priority: Improve {min_component[0]}"
        else:
            return "Moderate priority: Fine-tune all components"

    def behavior_change_potential(self, behavior: Behavior) -> Dict[str, float]:
        """
        Estimate potential for behavior change by improving each component.

        Args:
            behavior: Behavior to analyze

        Returns:
            Dict mapping component improvements to behavior likelihood gains
        """
        current_likelihood = self.predict_behavior(
            behavior.capability, behavior.opportunity, behavior.motivation
        )

        # Simulate improving each component to 0.9
        improved_cap = Capability(
            physical=0.9, psychological=0.9, knowledge=0.9, skills=0.9
        )
        improved_opp = Opportunity(
            physical=0.9, social=0.9, environmental=0.9, temporal=0.9
        )
        improved_mot = Motivation(
            reflective=0.9, automatic=0.9, intrinsic=0.9, extrinsic=0.9
        )

        cap_improvement = (
            self.predict_behavior(improved_cap, behavior.opportunity, behavior.motivation)
            - current_likelihood
        )
        opp_improvement = (
            self.predict_behavior(behavior.capability, improved_opp, behavior.motivation)
            - current_likelihood
        )
        mot_improvement = (
            self.predict_behavior(behavior.capability, behavior.opportunity, improved_mot)
            - current_likelihood
        )

        return {
            "current_likelihood": current_likelihood,
            "capability_improvement_potential": cap_improvement,
            "opportunity_improvement_potential": opp_improvement,
            "motivation_improvement_potential": mot_improvement,
            "max_potential_likelihood": self.predict_behavior(
                improved_cap, improved_opp, improved_mot
            ),
        }
