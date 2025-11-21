"""Tests for intervention optimizer."""

import pytest
from bcf.models.base import Behavior, BehaviorType, Capability, Motivation, Opportunity, StageOfChange
from bcf.intervention.optimizer import InterventionOptimizer


class TestInterventionOptimizer:
    """Test intervention optimizer."""

    def test_design(self):
        """Test intervention design."""
        optimizer = InterventionOptimizer()

        behavior = Behavior(
            name="daily exercise",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.6, psychological=0.5, knowledge=0.7, skills=0.6),
            opportunity=Opportunity(physical=0.4, social=0.3, environmental=0.5, temporal=0.3),
            motivation=Motivation(reflective=0.7, automatic=0.3, intrinsic=0.6, extrinsic=0.4),
            stage_of_change=StageOfChange.PREPARATION,
            current_frequency=1.0,
            desired_frequency=5.0,
        )

        intervention = optimizer.design(
            target_behavior="daily exercise",
            current_state=behavior,
            desired_outcomes={"capability": 0.8, "motivation": 0.8, "adherence": 0.85},
            constraints={"cost": 100, "time": 45},
        )

        assert intervention.target_behavior == "daily exercise"
        assert len(intervention.components) > 0
        assert 0 <= intervention.success_probability <= 1
        assert intervention.total_cost <= 100
        assert intervention.total_time <= 45
        assert len(intervention.implementation_plan) > 0
        assert len(intervention.monitoring_metrics) > 0

    def test_intervention_components(self):
        """Test that intervention has valid components."""
        optimizer = InterventionOptimizer()

        behavior = Behavior(
            name="test",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.3, psychological=0.4, knowledge=0.5, skills=0.4),
            opportunity=Opportunity(physical=0.3, social=0.3, environmental=0.4, temporal=0.3),
            motivation=Motivation(reflective=0.5, automatic=0.2, intrinsic=0.4, extrinsic=0.3),
            stage_of_change=StageOfChange.CONTEMPLATION,
        )

        intervention = optimizer.design(
            target_behavior="test",
            current_state=behavior,
            desired_outcomes={"capability": 0.7, "motivation": 0.7},
        )

        for component in intervention.components:
            assert component.expected_effect_size >= 0
            assert component.cost >= 0
            assert component.time_required >= 0
            assert 0 <= component.difficulty <= 1
