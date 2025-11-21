"""Consequence evaluator - evaluates and ranks consequences based on values."""

from typing import Any, Dict, List

from bcf.consequence.predictor import Consequence, ConsequenceAnalysis, ConsequenceDimension


class ConsequenceEvaluator:
    """Evaluates consequences based on value frameworks."""

    def __init__(self, value_weights: Dict[ConsequenceDimension, float] = None):
        """
        Initialize evaluator.

        Args:
            value_weights: Custom weights for each dimension (default: equal weights)
        """
        if value_weights is None:
            # Equal weights by default
            dimensions = list(ConsequenceDimension)
            self.value_weights = {dim: 1.0 / len(dimensions) for dim in dimensions}
        else:
            # Normalize weights
            total = sum(value_weights.values())
            self.value_weights = {k: v / total for k, v in value_weights.items()}

    def evaluate(self, analysis: ConsequenceAnalysis) -> Dict[str, Any]:
        """
        Evaluate consequence analysis based on value framework.

        Args:
            analysis: Consequence analysis to evaluate

        Returns:
            Evaluation with value-weighted scores
        """
        # Calculate value-weighted score
        weighted_value = 0.0

        for consequence in analysis.consequences:
            weight = self.value_weights.get(consequence.dimension, 0.0)
            weighted_value += consequence.expected_value * weight

        # Identify high-value consequences
        high_value = [
            c for c in analysis.consequences
            if abs(c.expected_value) > 0.5
        ]

        # Identify high-risk consequences
        high_risk = [
            c for c in analysis.consequences
            if c.risk_score > 0.3
        ]

        return {
            "weighted_value": weighted_value,
            "high_value_consequences": high_value,
            "high_risk_consequences": high_risk,
            "value_alignment_score": self._calculate_alignment(analysis),
            "recommendation": self._generate_recommendation(weighted_value, high_risk),
        }

    def _calculate_alignment(self, analysis: ConsequenceAnalysis) -> float:
        """Calculate how well behavior aligns with values (0-1)."""
        positive_weighted = sum(
            c.expected_value * self.value_weights.get(c.dimension, 0)
            for c in analysis.consequences
            if c.expected_value > 0
        )

        negative_weighted = sum(
            abs(c.expected_value) * self.value_weights.get(c.dimension, 0)
            for c in analysis.consequences
            if c.expected_value < 0
        )

        total = positive_weighted + negative_weighted

        if total == 0:
            return 0.5

        return positive_weighted / total

    def _generate_recommendation(self, weighted_value: float, high_risk: List[Consequence]) -> str:
        """Generate recommendation based on evaluation."""
        if weighted_value > 0.3 and len(high_risk) == 0:
            return "Highly recommended - positive value with low risk"
        elif weighted_value > 0.1:
            return "Recommended with caution - manage identified risks"
        elif weighted_value > -0.1:
            return "Neutral - carefully weigh pros and cons"
        else:
            return "Not recommended - negative expected value"
