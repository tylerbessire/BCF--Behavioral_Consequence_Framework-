"""
EAST Framework Implementation

The EAST framework (UK Behavioural Insights Team) provides four principles
for designing effective behavioral interventions:
- Easy: Make it as simple as possible
- Attractive: Make it appealing and attention-grabbing
- Social: Leverage social influence and norms
- Timely: Deliver at the right moment

Reference:
Behavioural Insights Team (2014). EAST: Four simple ways to apply
behavioural insights.
"""

from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

from bcf.models.base import Behavior, BehavioralModel


class EASTComponents(BaseModel):
    """Components of the EAST framework."""

    easy: float = Field(
        ge=0.0, le=1.0, description="How easy/simple is the behavior (0-1)"
    )
    attractive: float = Field(
        ge=0.0, le=1.0, description="How attractive/appealing is the behavior (0-1)"
    )
    social: float = Field(
        ge=0.0, le=1.0, description="Social influence and norms support (0-1)"
    )
    timely: float = Field(
        ge=0.0, le=1.0, description="Timing and context appropriateness (0-1)"
    )

    @property
    def overall_score(self) -> float:
        """Calculate overall EAST score."""
        return np.mean([self.easy, self.attractive, self.social, self.timely])


class EASTFramework(BehavioralModel):
    """
    EAST Framework for Behavioral Intervention Design.

    Focuses on making behaviors Easy, Attractive, Social, and Timely.
    Particularly effective for nudge-based interventions.
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize EAST framework.

        Args:
            weights: Optional custom weights for each component
                   (default: equal weights of 0.25)
        """
        if weights is None:
            self.weights = {
                "easy": 0.25,
                "attractive": 0.25,
                "social": 0.25,
                "timely": 0.25,
            }
        else:
            if not np.isclose(sum(weights.values()), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights

    def predict_behavior(
        self, easy: float, attractive: float, social: float, timely: float
    ) -> float:
        """
        Predict likelihood of behavior based on EAST components.

        Args:
            easy: Ease/simplicity score (0-1)
            attractive: Attractiveness score (0-1)
            social: Social influence score (0-1)
            timely: Timing score (0-1)

        Returns:
            Predicted behavior likelihood (0-1)
        """
        return (
            self.weights["easy"] * easy
            + self.weights["attractive"] * attractive
            + self.weights["social"] * social
            + self.weights["timely"] * timely
        )

    def diagnose(self, behavior: Behavior) -> Dict[str, Any]:
        """
        Diagnose behavior using EAST framework.

        Args:
            behavior: Behavior to analyze

        Returns:
            EAST component scores and recommendations
        """
        # Map behavior attributes to EAST components
        easy_score = self._assess_ease(behavior)
        attractive_score = self._assess_attractiveness(behavior)
        social_score = self._assess_social(behavior)
        timely_score = self._assess_timeliness(behavior)

        overall = self.predict_behavior(
            easy_score, attractive_score, social_score, timely_score
        )

        components = {
            "easy": easy_score,
            "attractive": attractive_score,
            "social": social_score,
            "timely": timely_score,
        }

        # Identify weak components
        threshold = 0.6
        weak_components = {k: v for k, v in components.items() if v < threshold}

        return {
            "east_scores": components,
            "overall_score": overall,
            "weak_components": weak_components,
            "strongest_component": max(components.items(), key=lambda x: x[1])[0],
            "weakest_component": min(components.items(), key=lambda x: x[1])[0],
            "intervention_opportunities": self._identify_opportunities(components),
        }

    def recommend_interventions(self, behavior: Behavior) -> List[Dict[str, Any]]:
        """
        Recommend EAST-based interventions.

        Args:
            behavior: Target behavior

        Returns:
            List of EAST intervention recommendations
        """
        diagnosis = self.diagnose(behavior)
        interventions = []

        scores = diagnosis["east_scores"]

        # Easy interventions
        if scores["easy"] < 0.7:
            interventions.append({
                "type": "make_it_easy",
                "east_principle": "Easy",
                "strategies": [
                    "Harness power of defaults (opt-out vs opt-in)",
                    "Reduce friction and steps required",
                    "Simplify language and messages",
                    "Use pre-populated forms",
                    "Break down into smaller steps",
                    "Provide clear implementation guidance",
                ],
                "priority": 1.0 - scores["easy"],
                "examples": [
                    "Set healthy option as default in cafeteria",
                    "Auto-enroll in savings plan (with opt-out)",
                    "Pre-fill form with known information",
                ],
            })

        # Attractive interventions
        if scores["attractive"] < 0.7:
            interventions.append({
                "type": "make_it_attractive",
                "east_principle": "Attractive",
                "strategies": [
                    "Design for attention (visual salience)",
                    "Personalize messages and rewards",
                    "Use incentives and gamification",
                    "Frame positively (gains vs losses)",
                    "Make feedback tangible and visible",
                    "Create aesthetic appeal",
                ],
                "priority": 1.0 - scores["attractive"],
                "examples": [
                    "Use bright colors for healthy food options",
                    "Show personalized benefits (You will save $X)",
                    "Award badges/points for progress",
                    "Display real-time energy savings",
                ],
            })

        # Social interventions
        if scores["social"] < 0.7:
            interventions.append({
                "type": "make_it_social",
                "east_principle": "Social",
                "strategies": [
                    "Show what others are doing (social proof)",
                    "Use social norms messaging",
                    "Leverage peer networks and influencers",
                    "Create public commitments",
                    "Build community and belonging",
                    "Use messenger effects (who delivers message)",
                ],
                "priority": 1.0 - scores["social"],
                "examples": [
                    "9 out of 10 neighbors recycle regularly",
                    "Your friends are already using this",
                    "Join others in your community taking action",
                    "Publicly commit to goal on social media",
                ],
            })

        # Timely interventions
        if scores["timely"] < 0.7:
            interventions.append({
                "type": "make_it_timely",
                "east_principle": "Timely",
                "strategies": [
                    "Prompt at decision-making moments",
                    "Leverage key life transitions",
                    "Use just-in-time reminders",
                    "Consider present bias (immediate rewards)",
                    "Plan for future self (implementation intentions)",
                    "Match timing to routines",
                ],
                "priority": 1.0 - scores["timely"],
                "examples": [
                    "Prompt gym signup in January (New Year)",
                    "Offer retirement planning when changing jobs",
                    "Send medication reminder at usual time",
                    "Highlight immediate benefits over future ones",
                ],
            })

        # Sort by priority
        interventions.sort(key=lambda x: x["priority"], reverse=True)

        return interventions

    def _assess_ease(self, behavior: Behavior) -> float:
        """Assess how easy the behavior is."""
        # Combine capability and opportunity (physical/temporal)
        return np.mean([
            behavior.capability.overall,
            behavior.opportunity.physical,
            behavior.opportunity.temporal,
        ])

    def _assess_attractiveness(self, behavior: Behavior) -> float:
        """Assess how attractive the behavior is."""
        # Based on automatic motivation (emotional appeal)
        return behavior.motivation.automatic

    def _assess_social(self, behavior: Behavior) -> float:
        """Assess social component."""
        return behavior.opportunity.social

    def _assess_timeliness(self, behavior: Behavior) -> float:
        """Assess timing appropriateness."""
        # Combination of temporal opportunity and context
        temporal = behavior.opportunity.temporal

        # Bonus if in action or maintenance stage (good timing)
        from bcf.models.base import StageOfChange

        stage_bonus = 0.0
        if behavior.stage_of_change in [StageOfChange.ACTION, StageOfChange.MAINTENANCE]:
            stage_bonus = 0.2

        return min(1.0, temporal + stage_bonus)

    def _identify_opportunities(self, components: Dict[str, float]) -> List[str]:
        """Identify specific intervention opportunities."""
        opportunities = []

        if components["easy"] < 0.5:
            opportunities.append("High friction - simplify and reduce barriers")

        if components["attractive"] < 0.5:
            opportunities.append("Low appeal - enhance incentives and framing")

        if components["social"] < 0.5:
            opportunities.append("Weak social influence - leverage norms and proof")

        if components["timely"] < 0.5:
            opportunities.append("Poor timing - optimize prompts and context")

        if not opportunities:
            opportunities.append("Fine-tune existing strengths")

        return opportunities

    def design_nudge(
        self,
        behavior: Behavior,
        focus_components: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Design a nudge intervention using EAST principles.

        Args:
            behavior: Target behavior
            focus_components: Specific EAST components to focus on
                            (default: all weak components)

        Returns:
            Nudge design with specific tactics
        """
        diagnosis = self.diagnose(behavior)

        if focus_components is None:
            # Focus on weak components
            focus_components = list(diagnosis["weak_components"].keys())
            if not focus_components:
                # If nothing weak, improve weakest
                focus_components = [diagnosis["weakest_component"]]

        nudge_tactics = []

        for component in focus_components:
            interventions = [
                i for i in self.recommend_interventions(behavior)
                if i["east_principle"].lower() == component
            ]
            if interventions:
                nudge_tactics.extend(interventions[0]["strategies"][:2])  # Top 2

        return {
            "target_behavior": behavior.name,
            "focus_areas": focus_components,
            "current_east_scores": diagnosis["east_scores"],
            "nudge_tactics": nudge_tactics,
            "expected_improvement": self._estimate_improvement(
                diagnosis["east_scores"], focus_components
            ),
            "implementation_steps": self._generate_implementation_steps(
                behavior, focus_components
            ),
        }

    def _estimate_improvement(
        self, current_scores: Dict[str, float], focus_components: List[str]
    ) -> float:
        """Estimate expected improvement from intervention."""
        # Conservative estimate: 30-50% improvement in weak areas
        total_improvement = 0
        for component in focus_components:
            current = current_scores[component]
            potential_gain = (0.85 - current) * 0.4  # 40% of gap to 0.85
            total_improvement += potential_gain * self.weights[component]

        return total_improvement

    def _generate_implementation_steps(
        self, behavior: Behavior, focus_components: List[str]
    ) -> List[str]:
        """Generate concrete implementation steps."""
        steps = [
            "1. Conduct baseline measurement of current behavior rate",
            "2. Identify specific touchpoints for intervention",
        ]

        for i, component in enumerate(focus_components, start=3):
            steps.append(f"{i}. Implement {component.upper()} interventions")

        steps.extend([
            f"{len(steps) + 1}. Test and iterate nudge design",
            f"{len(steps) + 2}. Measure impact and refine",
        ])

        return steps
