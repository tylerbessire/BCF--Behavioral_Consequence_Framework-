"""Tests for consequence prediction."""

import pytest
from bcf.models.base import Behavior, BehaviorType, Capability, Motivation, Opportunity, StageOfChange
from bcf.consequence.predictor import ConsequencePredictor, ConsequenceDimension, TimeHorizon


class TestConsequencePredictor:
    """Test consequence predictor."""

    def test_predict(self):
        """Test consequence prediction."""
        predictor = ConsequencePredictor()

        behavior = Behavior(
            name="daily exercise",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.7, psychological=0.6, knowledge=0.8, skills=0.7),
            opportunity=Opportunity(physical=0.6, social=0.5, environmental=0.6, temporal=0.5),
            motivation=Motivation(reflective=0.8, automatic=0.5, intrinsic=0.7, extrinsic=0.5),
            stage_of_change=StageOfChange.ACTION,
            current_frequency=3.0,
            desired_frequency=5.0,
        )

        analysis = predictor.predict(behavior)

        assert analysis.behavior_name == "daily exercise"
        assert len(analysis.consequences) > 0
        assert analysis.dominant_dimension is not None
        assert len(analysis.time_profile) > 0

    def test_consequence_properties(self):
        """Test consequence properties."""
        predictor = ConsequencePredictor()

        behavior = Behavior(
            name="test",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.5, psychological=0.5, knowledge=0.5, skills=0.5),
            opportunity=Opportunity(physical=0.5, social=0.5, environmental=0.5, temporal=0.5),
            motivation=Motivation(reflective=0.5, automatic=0.5, intrinsic=0.5, extrinsic=0.5),
            stage_of_change=StageOfChange.CONTEMPLATION,
        )

        analysis = predictor.predict(behavior)

        for consequence in analysis.consequences:
            assert -1 <= consequence.magnitude <= 1
            assert 0 <= consequence.probability <= 1
            assert 0 <= consequence.certainty <= 1
            assert 0 <= consequence.reversibility <= 1
            assert -1 <= consequence.expected_value <= 1

    def test_compare_scenarios(self):
        """Test scenario comparison."""
        predictor = ConsequencePredictor()

        behavior1 = Behavior(
            name="exercise 3x/week",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.7, psychological=0.6, knowledge=0.8, skills=0.7),
            opportunity=Opportunity(physical=0.6, social=0.5, environmental=0.6, temporal=0.5),
            motivation=Motivation(reflective=0.7, automatic=0.4, intrinsic=0.7, extrinsic=0.5),
            stage_of_change=StageOfChange.ACTION,
            current_frequency=3.0,
            desired_frequency=3.0,
        )

        behavior2 = Behavior(
            name="exercise 5x/week",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.7, psychological=0.6, knowledge=0.8, skills=0.7),
            opportunity=Opportunity(physical=0.6, social=0.5, environmental=0.6, temporal=0.5),
            motivation=Motivation(reflective=0.8, automatic=0.5, intrinsic=0.8, extrinsic=0.5),
            stage_of_change=StageOfChange.ACTION,
            current_frequency=5.0,
            desired_frequency=5.0,
        )

        comparison = predictor.compare_scenarios([
            ("3x per week", behavior1),
            ("5x per week", behavior2),
        ])

        assert "scenarios" in comparison
        assert "best_scenario" in comparison
        assert len(comparison["scenarios"]) == 2
