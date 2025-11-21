"""
Consequence Prediction Engine

Predicts multi-dimensional consequences of behaviors and interventions
across temporal horizons and stakeholder groups.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from bcf.models.base import Behavior


class ConsequenceDimension(str, Enum):
    """Dimensions of consequences."""

    HEALTH = "health"
    PSYCHOLOGICAL = "psychological"
    SOCIAL = "social"
    FINANCIAL = "financial"
    ENVIRONMENTAL = "environmental"
    PROFESSIONAL = "professional"
    TIME = "time"


class TimeHorizon(str, Enum):
    """Time horizons for consequence analysis."""

    IMMEDIATE = "immediate"  # Seconds to minutes
    SHORT_TERM = "short_term"  # Days to weeks
    MEDIUM_TERM = "medium_term"  # Months
    LONG_TERM = "long_term"  # Years
    GENERATIONAL = "generational"  # Decades


class ConsequenceType(str, Enum):
    """Types of consequences."""

    DIRECT = "direct"  # Direct result of behavior
    INDIRECT = "indirect"  # Cascading or secondary effects
    OPPORTUNITY_COST = "opportunity_cost"  # What's foregone


class Stakeholder(str, Enum):
    """Stakeholders affected by consequences."""

    SELF = "self"
    FAMILY = "family"
    COMMUNITY = "community"
    ORGANIZATION = "organization"
    SOCIETY = "society"
    ENVIRONMENT = "environment"


class Consequence(BaseModel):
    """Represents a single consequence."""

    dimension: ConsequenceDimension
    time_horizon: TimeHorizon
    stakeholder: Stakeholder
    consequence_type: ConsequenceType
    description: str
    magnitude: float = Field(ge=-1.0, le=1.0, description="Impact magnitude (-1 to 1)")
    probability: float = Field(ge=0.0, le=1.0, description="Likelihood (0-1)")
    certainty: float = Field(ge=0.0, le=1.0, description="Confidence in prediction (0-1)")
    reversibility: float = Field(
        ge=0.0, le=1.0, description="How easily can this be reversed (0-1)"
    )

    @property
    def expected_value(self) -> float:
        """Calculate expected value (magnitude × probability)."""
        return self.magnitude * self.probability

    @property
    def risk_score(self) -> float:
        """Calculate risk score for negative consequences."""
        if self.magnitude < 0:
            return abs(self.magnitude) * self.probability * (1 - self.reversibility)
        return 0.0


@dataclass
class ConsequenceAnalysis:
    """Complete consequence analysis for a behavior."""

    behavior_name: str
    consequences: List[Consequence] = field(default_factory=list)
    net_expected_value: float = 0.0
    total_risk_score: float = 0.0
    dominant_dimension: Optional[ConsequenceDimension] = None
    time_profile: Dict[TimeHorizon, float] = field(default_factory=dict)
    stakeholder_impact: Dict[Stakeholder, float] = field(default_factory=dict)


class ConsequencePredictor:
    """
    Predicts multi-dimensional consequences of behaviors.

    Uses evidence-based models and heuristics to forecast outcomes
    across multiple dimensions, time horizons, and stakeholders.
    """

    def __init__(self, uncertainty_factor: float = 0.2):
        """
        Initialize consequence predictor.

        Args:
            uncertainty_factor: Factor to adjust certainty estimates (0-1)
                              Higher = more conservative certainty estimates
        """
        self.uncertainty_factor = uncertainty_factor

    def predict(
        self,
        behavior: Behavior,
        time_horizons: Optional[List[TimeHorizon]] = None,
        dimensions: Optional[List[ConsequenceDimension]] = None,
        stakeholders: Optional[List[Stakeholder]] = None,
    ) -> ConsequenceAnalysis:
        """
        Predict consequences of a behavior.

        Args:
            behavior: Behavior to analyze
            time_horizons: Time horizons to consider (default: all)
            dimensions: Dimensions to analyze (default: all relevant)
            stakeholders: Stakeholders to consider (default: all)

        Returns:
            Comprehensive consequence analysis
        """
        if time_horizons is None:
            time_horizons = list(TimeHorizon)

        if dimensions is None:
            dimensions = self._infer_relevant_dimensions(behavior)

        if stakeholders is None:
            stakeholders = [Stakeholder.SELF, Stakeholder.FAMILY, Stakeholder.COMMUNITY]

        consequences = []

        # Generate consequences for each combination
        for horizon in time_horizons:
            for dimension in dimensions:
                for stakeholder in stakeholders:
                    # Direct consequences
                    direct = self._predict_consequence(
                        behavior, horizon, dimension, stakeholder, ConsequenceType.DIRECT
                    )
                    if direct:
                        consequences.append(direct)

                    # Indirect consequences (for medium/long term)
                    if horizon in [TimeHorizon.MEDIUM_TERM, TimeHorizon.LONG_TERM]:
                        indirect = self._predict_consequence(
                            behavior,
                            horizon,
                            dimension,
                            stakeholder,
                            ConsequenceType.INDIRECT,
                        )
                        if indirect:
                            consequences.append(indirect)

        # Calculate aggregate metrics
        analysis = ConsequenceAnalysis(behavior_name=behavior.name, consequences=consequences)

        analysis.net_expected_value = sum(c.expected_value for c in consequences)
        analysis.total_risk_score = sum(c.risk_score for c in consequences)

        # Aggregate by time horizon
        for horizon in TimeHorizon:
            horizon_consequences = [c for c in consequences if c.time_horizon == horizon]
            if horizon_consequences:
                analysis.time_profile[horizon] = sum(
                    c.expected_value for c in horizon_consequences
                )

        # Aggregate by stakeholder
        for stakeholder in Stakeholder:
            stakeholder_consequences = [c for c in consequences if c.stakeholder == stakeholder]
            if stakeholder_consequences:
                analysis.stakeholder_impact[stakeholder] = sum(
                    c.expected_value for c in stakeholder_consequences
                )

        # Find dominant dimension
        dimension_impacts = {}
        for dim in ConsequenceDimension:
            dim_consequences = [c for c in consequences if c.dimension == dim]
            if dim_consequences:
                dimension_impacts[dim] = sum(abs(c.expected_value) for c in dim_consequences)

        if dimension_impacts:
            analysis.dominant_dimension = max(dimension_impacts.items(), key=lambda x: x[1])[0]

        return analysis

    def _infer_relevant_dimensions(self, behavior: Behavior) -> List[ConsequenceDimension]:
        """Infer which dimensions are most relevant for this behavior."""
        from bcf.models.base import BehaviorType

        # Map behavior types to relevant dimensions
        dimension_map = {
            BehaviorType.HEALTH: [
                ConsequenceDimension.HEALTH,
                ConsequenceDimension.PSYCHOLOGICAL,
                ConsequenceDimension.FINANCIAL,
            ],
            BehaviorType.FINANCIAL: [
                ConsequenceDimension.FINANCIAL,
                ConsequenceDimension.PSYCHOLOGICAL,
                ConsequenceDimension.TIME,
            ],
            BehaviorType.SOCIAL: [
                ConsequenceDimension.SOCIAL,
                ConsequenceDimension.PSYCHOLOGICAL,
            ],
            BehaviorType.ENVIRONMENTAL: [
                ConsequenceDimension.ENVIRONMENTAL,
                ConsequenceDimension.FINANCIAL,
                ConsequenceDimension.SOCIAL,
            ],
            BehaviorType.EDUCATIONAL: [
                ConsequenceDimension.PROFESSIONAL,
                ConsequenceDimension.FINANCIAL,
                ConsequenceDimension.TIME,
            ],
            BehaviorType.PROFESSIONAL: [
                ConsequenceDimension.PROFESSIONAL,
                ConsequenceDimension.FINANCIAL,
                ConsequenceDimension.PSYCHOLOGICAL,
            ],
        }

        return dimension_map.get(
            behavior.behavior_type,
            [ConsequenceDimension.PSYCHOLOGICAL, ConsequenceDimension.TIME],
        )

    def _predict_consequence(
        self,
        behavior: Behavior,
        horizon: TimeHorizon,
        dimension: ConsequenceDimension,
        stakeholder: Stakeholder,
        consequence_type: ConsequenceType,
    ) -> Optional[Consequence]:
        """Predict a specific consequence."""
        # Get base magnitude from behavior characteristics
        magnitude = self._estimate_magnitude(behavior, dimension, horizon, stakeholder)

        # Skip if magnitude too small
        if abs(magnitude) < 0.1:
            return None

        # Estimate probability
        probability = self._estimate_probability(
            behavior, dimension, horizon, consequence_type
        )

        # Estimate certainty (decreases with time horizon)
        certainty = self._estimate_certainty(horizon, dimension, consequence_type)

        # Estimate reversibility
        reversibility = self._estimate_reversibility(dimension, horizon, behavior)

        # Generate description
        description = self._generate_description(
            behavior, dimension, horizon, stakeholder, magnitude
        )

        return Consequence(
            dimension=dimension,
            time_horizon=horizon,
            stakeholder=stakeholder,
            consequence_type=consequence_type,
            description=description,
            magnitude=magnitude,
            probability=probability,
            certainty=certainty,
            reversibility=reversibility,
        )

    def _estimate_magnitude(
        self,
        behavior: Behavior,
        dimension: ConsequenceDimension,
        horizon: TimeHorizon,
        stakeholder: Stakeholder,
    ) -> float:
        """Estimate magnitude of consequence (-1 to 1)."""
        from bcf.models.base import BehaviorType

        # Base magnitude from behavior frequency
        freq_ratio = behavior.current_frequency / max(1, behavior.desired_frequency)

        # Positive if moving toward desired, negative if away
        base_magnitude = (freq_ratio - 0.5) * 2  # Map 0-1 to -1 to 1

        # Adjust by behavior type and dimension match
        if (behavior.behavior_type == BehaviorType.HEALTH and
                dimension == ConsequenceDimension.HEALTH):
            base_magnitude *= 1.5

        # Time decay/accumulation
        horizon_multiplier = {
            TimeHorizon.IMMEDIATE: 0.3,
            TimeHorizon.SHORT_TERM: 0.6,
            TimeHorizon.MEDIUM_TERM: 1.0,
            TimeHorizon.LONG_TERM: 1.3,
            TimeHorizon.GENERATIONAL: 0.8,  # Less certain but broad
        }
        base_magnitude *= horizon_multiplier[horizon]

        # Stakeholder distance (closer = larger impact)
        stakeholder_multiplier = {
            Stakeholder.SELF: 1.0,
            Stakeholder.FAMILY: 0.6,
            Stakeholder.COMMUNITY: 0.3,
            Stakeholder.ORGANIZATION: 0.4,
            Stakeholder.SOCIETY: 0.2,
            Stakeholder.ENVIRONMENT: 0.5,
        }
        base_magnitude *= stakeholder_multiplier[stakeholder]

        return np.clip(base_magnitude, -1.0, 1.0)

    def _estimate_probability(
        self,
        behavior: Behavior,
        dimension: ConsequenceDimension,
        horizon: TimeHorizon,
        consequence_type: ConsequenceType,
    ) -> float:
        """Estimate probability of consequence occurring."""
        # Base probability from behavior likelihood
        base_prob = behavior.behavior_likelihood

        # Direct consequences more likely than indirect
        if consequence_type == ConsequenceType.DIRECT:
            base_prob *= 0.9
        else:  # Indirect
            base_prob *= 0.5

        # Adjust for time horizon (longer = more uncertainty)
        horizon_adjustment = {
            TimeHorizon.IMMEDIATE: 0.95,
            TimeHorizon.SHORT_TERM: 0.85,
            TimeHorizon.MEDIUM_TERM: 0.70,
            TimeHorizon.LONG_TERM: 0.50,
            TimeHorizon.GENERATIONAL: 0.30,
        }
        base_prob *= horizon_adjustment[horizon]

        return np.clip(base_prob, 0.0, 1.0)

    def _estimate_certainty(
        self, horizon: TimeHorizon, dimension: ConsequenceDimension, consequence_type: ConsequenceType
    ) -> float:
        """Estimate certainty in prediction."""
        # Base certainty decreases with time
        base_certainty = {
            TimeHorizon.IMMEDIATE: 0.9,
            TimeHorizon.SHORT_TERM: 0.8,
            TimeHorizon.MEDIUM_TERM: 0.6,
            TimeHorizon.LONG_TERM: 0.4,
            TimeHorizon.GENERATIONAL: 0.2,
        }[horizon]

        # Direct consequences more certain
        if consequence_type == ConsequenceType.INDIRECT:
            base_certainty *= 0.7

        # Apply uncertainty factor
        base_certainty *= (1 - self.uncertainty_factor)

        return np.clip(base_certainty, 0.0, 1.0)

    def _estimate_reversibility(
        self, dimension: ConsequenceDimension, horizon: TimeHorizon, behavior: Behavior
    ) -> float:
        """Estimate how reversible the consequence is."""
        # Some dimensions are more reversible than others
        base_reversibility = {
            ConsequenceDimension.HEALTH: 0.4,  # Health can be hard to reverse
            ConsequenceDimension.PSYCHOLOGICAL: 0.6,
            ConsequenceDimension.SOCIAL: 0.7,
            ConsequenceDimension.FINANCIAL: 0.8,
            ConsequenceDimension.ENVIRONMENTAL: 0.3,  # Often hard to reverse
            ConsequenceDimension.PROFESSIONAL: 0.6,
            ConsequenceDimension.TIME: 0.0,  # Time cannot be reversed
        }[dimension]

        # Longer time horizons = less reversible
        horizon_penalty = {
            TimeHorizon.IMMEDIATE: 0.0,
            TimeHorizon.SHORT_TERM: 0.1,
            TimeHorizon.MEDIUM_TERM: 0.2,
            TimeHorizon.LONG_TERM: 0.3,
            TimeHorizon.GENERATIONAL: 0.5,
        }[horizon]

        reversibility = base_reversibility - horizon_penalty

        return np.clip(reversibility, 0.0, 1.0)

    def _generate_description(
        self,
        behavior: Behavior,
        dimension: ConsequenceDimension,
        horizon: TimeHorizon,
        stakeholder: Stakeholder,
        magnitude: float,
    ) -> str:
        """Generate human-readable description of consequence."""
        direction = "positive" if magnitude > 0 else "negative"
        intensity = "significant" if abs(magnitude) > 0.7 else "moderate"

        time_desc = {
            TimeHorizon.IMMEDIATE: "immediately",
            TimeHorizon.SHORT_TERM: "in the short term",
            TimeHorizon.MEDIUM_TERM: "over several months",
            TimeHorizon.LONG_TERM: "in the long run",
            TimeHorizon.GENERATIONAL: "over generations",
        }[horizon]

        return (
            f"{intensity.capitalize()} {direction} {dimension.value} impact for "
            f"{stakeholder.value} {time_desc} from {behavior.name}"
        )

    def compare_scenarios(
        self,
        scenarios: List[tuple[str, Behavior]],
        dimensions: Optional[List[ConsequenceDimension]] = None,
    ) -> Dict[str, Any]:
        """
        Compare consequences across multiple behavioral scenarios.

        Args:
            scenarios: List of (scenario_name, behavior) tuples
            dimensions: Dimensions to compare (default: all)

        Returns:
            Comparison analysis
        """
        scenario_analyses = {}

        for scenario_name, behavior in scenarios:
            analysis = self.predict(behavior, dimensions=dimensions)
            scenario_analyses[scenario_name] = analysis

        # Find best scenario overall
        best_scenario = max(
            scenario_analyses.items(),
            key=lambda x: x[1].net_expected_value
        )

        # Compare by dimension
        dimension_comparison = {}
        for dim in dimensions or list(ConsequenceDimension):
            dimension_comparison[dim] = {}
            for scenario_name, analysis in scenario_analyses.items():
                dim_consequences = [
                    c for c in analysis.consequences if c.dimension == dim
                ]
                dimension_comparison[dim][scenario_name] = sum(
                    c.expected_value for c in dim_consequences
                )

        return {
            "scenarios": scenario_analyses,
            "best_scenario": best_scenario[0],
            "best_scenario_value": best_scenario[1].net_expected_value,
            "dimension_comparison": dimension_comparison,
            "risk_comparison": {
                name: analysis.total_risk_score
                for name, analysis in scenario_analyses.items()
            },
        }
