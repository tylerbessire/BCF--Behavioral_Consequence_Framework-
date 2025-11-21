"""
Intervention Optimizer

Uses multi-objective optimization to design behavioral interventions
that maximize desired outcomes while respecting constraints.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from bcf.models.base import Behavior, BehavioralModel
from bcf.models.analyzer import BehavioralAnalyzer
from bcf.consequence.predictor import ConsequencePredictor, ConsequenceDimension


class InterventionStrategy(str, Enum):
    """Types of intervention strategies."""

    EDUCATION = "education"  # Information and knowledge
    PERSUASION = "persuasion"  # Motivation and attitude change
    INCENTIVIZATION = "incentivization"  # Rewards and penalties
    COERCION = "coercion"  # Rules and restrictions
    TRAINING = "training"  # Skill development
    ENABLEMENT = "enablement"  # Resources and support
    MODELING = "modeling"  # Social proof and examples
    ENVIRONMENTAL_RESTRUCTURING = "environmental_restructuring"  # Change context
    NUDGING = "nudging"  # Choice architecture


class InterventionComponent(BaseModel):
    """A component of an intervention."""

    strategy: InterventionStrategy
    description: str
    target_factor: str  # What it targets (capability, motivation, etc.)
    expected_effect_size: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)  # Relative cost (0-100 scale)
    time_required: float = Field(ge=0.0)  # Minutes per day
    difficulty: float = Field(ge=0.0, le=1.0)  # Implementation difficulty


@dataclass
class Intervention:
    """A complete behavioral intervention design."""

    name: str
    target_behavior: str
    components: List[InterventionComponent] = field(default_factory=list)
    expected_outcomes: Dict[str, float] = field(default_factory=dict)
    success_probability: float = 0.0
    total_cost: float = 0.0
    total_time: float = 0.0
    implementation_plan: List[str] = field(default_factory=list)
    monitoring_metrics: List[str] = field(default_factory=list)
    expected_consequences: Optional[Any] = None


class InterventionOptimizer:
    """
    Optimizes behavioral interventions using evidence-based strategies.

    Combines insights from multiple behavioral models to design
    interventions that maximize success while respecting constraints.
    """

    def __init__(self, analyzer: Optional[BehavioralAnalyzer] = None):
        """
        Initialize intervention optimizer.

        Args:
            analyzer: Behavioral analyzer (default: creates new one)
        """
        self.analyzer = analyzer or BehavioralAnalyzer()
        self.consequence_predictor = ConsequencePredictor()

    def design(
        self,
        target_behavior: str,
        current_state: Behavior,
        desired_outcomes: Dict[str, float],
        constraints: Optional[Dict[str, float]] = None,
        allowed_strategies: Optional[List[InterventionStrategy]] = None,
    ) -> Intervention:
        """
        Design an optimal behavioral intervention.

        Args:
            target_behavior: Name of target behavior
            current_state: Current behavioral state
            desired_outcomes: Desired outcomes (e.g., {"health": 0.9, "adherence": 0.8})
            constraints: Constraints (e.g., {"cost": 100, "time": 30})
            allowed_strategies: Allowed intervention strategies (default: all)

        Returns:
            Optimized intervention design
        """
        if constraints is None:
            constraints = {"cost": 100, "time": 60}  # Default constraints

        if allowed_strategies is None:
            allowed_strategies = list(InterventionStrategy)

        # Analyze current behavior
        analysis = self.analyzer.analyze(current_state)

        # Identify gaps and priorities
        gaps = self._identify_gaps(current_state, desired_outcomes, analysis)

        # Generate candidate intervention components
        candidates = self._generate_candidates(
            current_state, gaps, analysis, allowed_strategies
        )

        # Select optimal combination
        selected_components = self._optimize_selection(
            candidates, desired_outcomes, constraints
        )

        # Create intervention
        intervention = Intervention(
            name=f"Optimized intervention for {target_behavior}",
            target_behavior=target_behavior,
            components=selected_components,
        )

        # Calculate aggregates
        intervention.total_cost = sum(c.cost for c in selected_components)
        intervention.total_time = sum(c.time_required for c in selected_components)

        # Estimate success probability
        intervention.success_probability = self._estimate_success_probability(
            current_state, selected_components, analysis
        )

        # Generate implementation plan
        intervention.implementation_plan = self._generate_implementation_plan(
            selected_components, current_state
        )

        # Define monitoring metrics
        intervention.monitoring_metrics = self._define_monitoring_metrics(
            target_behavior, desired_outcomes
        )

        # Calculate expected outcomes
        intervention.expected_outcomes = self._calculate_expected_outcomes(
            current_state, selected_components, desired_outcomes
        )

        # Predict consequences
        intervention.expected_consequences = self._predict_intervention_consequences(
            current_state, selected_components
        )

        return intervention

    def _identify_gaps(
        self,
        current_state: Behavior,
        desired_outcomes: Dict[str, float],
        analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """Identify gaps between current and desired state."""
        gaps = {}

        # Capability gaps
        if "capability" in desired_outcomes:
            gaps["capability"] = desired_outcomes["capability"] - current_state.capability.overall

        # Motivation gaps
        if "motivation" in desired_outcomes:
            gaps["motivation"] = desired_outcomes["motivation"] - current_state.motivation.overall

        # Opportunity gaps
        if "opportunity" in desired_outcomes:
            gaps["opportunity"] = desired_outcomes["opportunity"] - current_state.opportunity.overall

        # Frequency gap
        gaps["frequency"] = current_state.desired_frequency - current_state.current_frequency

        # Adherence/consistency gap
        if "adherence" in desired_outcomes:
            gaps["adherence"] = desired_outcomes["adherence"] - (
                current_state.current_frequency / max(1, current_state.desired_frequency)
            )

        return gaps

    def _generate_candidates(
        self,
        behavior: Behavior,
        gaps: Dict[str, float],
        analysis: Dict[str, Any],
        allowed_strategies: List[InterventionStrategy],
    ) -> List[InterventionComponent]:
        """Generate candidate intervention components."""
        candidates = []

        # Education (targets capability - knowledge)
        if InterventionStrategy.EDUCATION in allowed_strategies:
            if behavior.capability.knowledge < 0.7:
                candidates.append(InterventionComponent(
                    strategy=InterventionStrategy.EDUCATION,
                    description="Educational materials and information sessions",
                    target_factor="capability_knowledge",
                    expected_effect_size=0.3,
                    cost=10.0,
                    time_required=15.0,
                    difficulty=0.2,
                ))

        # Training (targets capability - skills)
        if InterventionStrategy.TRAINING in allowed_strategies:
            if behavior.capability.skills < 0.7:
                candidates.append(InterventionComponent(
                    strategy=InterventionStrategy.TRAINING,
                    description="Skill-building workshops and practice sessions",
                    target_factor="capability_skills",
                    expected_effect_size=0.4,
                    cost=30.0,
                    time_required=45.0,
                    difficulty=0.5,
                ))

        # Incentivization (targets motivation)
        if InterventionStrategy.INCENTIVIZATION in allowed_strategies:
            if behavior.motivation.extrinsic < 0.6:
                candidates.append(InterventionComponent(
                    strategy=InterventionStrategy.INCENTIVIZATION,
                    description="Reward system for behavior completion",
                    target_factor="motivation_extrinsic",
                    expected_effect_size=0.4,
                    cost=50.0,
                    time_required=5.0,
                    difficulty=0.3,
                ))

        # Persuasion (targets motivation - reflective)
        if InterventionStrategy.PERSUASION in allowed_strategies:
            if behavior.motivation.reflective < 0.7:
                candidates.append(InterventionComponent(
                    strategy=InterventionStrategy.PERSUASION,
                    description="Motivational messaging and values clarification",
                    target_factor="motivation_reflective",
                    expected_effect_size=0.3,
                    cost=15.0,
                    time_required=10.0,
                    difficulty=0.2,
                ))

        # Enablement (targets opportunity)
        if InterventionStrategy.ENABLEMENT in allowed_strategies:
            if behavior.opportunity.physical < 0.7:
                candidates.append(InterventionComponent(
                    strategy=InterventionStrategy.ENABLEMENT,
                    description="Provide resources and remove barriers",
                    target_factor="opportunity_physical",
                    expected_effect_size=0.5,
                    cost=40.0,
                    time_required=0.0,
                    difficulty=0.4,
                ))

        # Modeling (targets motivation - automatic via social)
        if InterventionStrategy.MODELING in allowed_strategies:
            if behavior.opportunity.social < 0.6:
                candidates.append(InterventionComponent(
                    strategy=InterventionStrategy.MODELING,
                    description="Social proof and peer modeling",
                    target_factor="opportunity_social",
                    expected_effect_size=0.35,
                    cost=20.0,
                    time_required=10.0,
                    difficulty=0.3,
                ))

        # Environmental restructuring
        if InterventionStrategy.ENVIRONMENTAL_RESTRUCTURING in allowed_strategies:
            candidates.append(InterventionComponent(
                strategy=InterventionStrategy.ENVIRONMENTAL_RESTRUCTURING,
                description="Modify physical environment to support behavior",
                target_factor="opportunity_environmental",
                expected_effect_size=0.45,
                cost=35.0,
                time_required=0.0,
                difficulty=0.6,
            ))

        # Nudging
        if InterventionStrategy.NUDGING in allowed_strategies:
            candidates.append(InterventionComponent(
                strategy=InterventionStrategy.NUDGING,
                description="Choice architecture and default options",
                target_factor="automatic_behavior",
                expected_effect_size=0.3,
                cost=10.0,
                time_required=0.0,
                difficulty=0.4,
            ))

        return candidates

    def _optimize_selection(
        self,
        candidates: List[InterventionComponent],
        desired_outcomes: Dict[str, float],
        constraints: Dict[str, float],
    ) -> List[InterventionComponent]:
        """
        Select optimal combination of intervention components.

        Uses a greedy algorithm to maximize expected effect while respecting constraints.
        """
        selected = []
        remaining_cost = constraints.get("cost", float("inf"))
        remaining_time = constraints.get("time", float("inf"))

        # Sort by effectiveness per cost (greedy heuristic)
        candidates_sorted = sorted(
            candidates,
            key=lambda c: c.expected_effect_size / max(0.1, c.cost + c.time_required * 0.5),
            reverse=True,
        )

        for candidate in candidates_sorted:
            # Check constraints
            if candidate.cost <= remaining_cost and candidate.time_required <= remaining_time:
                selected.append(candidate)
                remaining_cost -= candidate.cost
                remaining_time -= candidate.time_required

        # Ensure at least one component selected
        if not selected and candidates:
            selected.append(min(candidates, key=lambda c: c.cost + c.time_required))

        return selected

    def _estimate_success_probability(
        self,
        behavior: Behavior,
        components: List[InterventionComponent],
        analysis: Dict[str, Any],
    ) -> float:
        """Estimate probability of intervention success."""
        # Base probability from current state
        base_prob = behavior.behavior_likelihood

        # Add expected effects (diminishing returns)
        cumulative_effect = sum(c.expected_effect_size for c in components)

        # Apply diminishing returns: 1 - (1 - base) * (1 - cumulative)^0.7
        improved_prob = 1 - (1 - base_prob) * ((1 - min(0.9, cumulative_effect)) ** 0.7)

        # Adjust for stage of change readiness
        readiness_multiplier = 0.5 + (behavior.readiness_score * 0.5)
        improved_prob *= readiness_multiplier

        return min(0.95, improved_prob)

    def _generate_implementation_plan(
        self, components: List[InterventionComponent], behavior: Behavior
    ) -> List[str]:
        """Generate step-by-step implementation plan."""
        plan = [
            "1. Conduct baseline assessment of current behavior",
            "2. Set specific, measurable goals",
        ]

        # Add component-specific steps
        for i, component in enumerate(components, start=3):
            plan.append(f"{i}. Implement {component.strategy.value}: {component.description}")

        plan.extend([
            f"{len(plan) + 1}. Monitor progress using defined metrics",
            f"{len(plan) + 2}. Adjust intervention based on feedback",
            f"{len(plan) + 3}. Celebrate successes and learn from setbacks",
        ])

        return plan

    def _define_monitoring_metrics(
        self, target_behavior: str, desired_outcomes: Dict[str, float]
    ) -> List[str]:
        """Define metrics to monitor intervention effectiveness."""
        metrics = [
            f"Frequency of {target_behavior} (times per week)",
            "Adherence rate (actual vs. planned)",
            "Self-reported motivation level (1-10)",
            "Perceived difficulty (1-10)",
        ]

        # Add outcome-specific metrics
        for outcome in desired_outcomes:
            if outcome not in ["adherence"]:
                metrics.append(f"{outcome.capitalize()} indicators")

        return metrics

    def _calculate_expected_outcomes(
        self,
        behavior: Behavior,
        components: List[InterventionComponent],
        desired_outcomes: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate expected outcomes from intervention."""
        outcomes = {}

        # Calculate improvement for each desired outcome
        for outcome, target in desired_outcomes.items():
            # Find relevant components
            relevant_components = [
                c for c in components
                if outcome.lower() in c.target_factor.lower()
                or outcome.lower() in c.description.lower()
            ]

            if relevant_components:
                # Aggregate effect
                total_effect = sum(c.expected_effect_size for c in relevant_components)

                # Current level (estimated)
                if outcome == "capability":
                    current = behavior.capability.overall
                elif outcome == "motivation":
                    current = behavior.motivation.overall
                elif outcome == "opportunity":
                    current = behavior.opportunity.overall
                else:
                    current = 0.5  # Default

                # Expected outcome with diminishing returns
                expected = current + total_effect * (1 - current)
                outcomes[outcome] = min(target, expected)
            else:
                # No direct intervention, expect modest improvement
                outcomes[outcome] = target * 0.7

        return outcomes

    def _predict_intervention_consequences(
        self, behavior: Behavior, components: List[InterventionComponent]
    ) -> Any:
        """Predict consequences of implementing the intervention."""
        # Create a hypothetical improved behavior state
        improved_behavior = Behavior(
            name=behavior.name,
            behavior_type=behavior.behavior_type,
            capability=behavior.capability,
            opportunity=behavior.opportunity,
            motivation=behavior.motivation,
            context=behavior.context,
            current_frequency=min(
                behavior.desired_frequency,
                behavior.current_frequency + sum(c.expected_effect_size for c in components) * 7,
            ),
            desired_frequency=behavior.desired_frequency,
            stage_of_change=behavior.stage_of_change,
            barriers=behavior.barriers,
            facilitators=behavior.facilitators,
        )

        # Predict consequences of improved state
        return self.consequence_predictor.predict(improved_behavior)
