"""Tests for behavioral models."""

import pytest
from bcf.models.base import Behavior, BehaviorType, Capability, Motivation, Opportunity, StageOfChange
from bcf.models.comb import COMBModel
from bcf.models.fogg import FoggBehaviorModel
from bcf.models.east import EASTFramework
from bcf.models.analyzer import BehavioralAnalyzer


class TestCOMBModel:
    """Test COM-B model."""

    def test_predict_behavior(self):
        """Test behavior prediction."""
        model = COMBModel()

        cap = Capability(physical=0.8, psychological=0.7, knowledge=0.9, skills=0.8)
        opp = Opportunity(physical=0.6, social=0.7, environmental=0.6, temporal=0.5)
        mot = Motivation(reflective=0.9, automatic=0.5, intrinsic=0.8, extrinsic=0.6)

        likelihood = model.predict_behavior(cap, opp, mot)

        assert 0 <= likelihood <= 1
        assert likelihood > 0.5  # Should be reasonably high

    def test_diagnose(self):
        """Test behavioral diagnosis."""
        model = COMBModel()

        behavior = Behavior(
            name="test behavior",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.5, psychological=0.5, knowledge=0.5, skills=0.5),
            opportunity=Opportunity(physical=0.4, social=0.4, environmental=0.4, temporal=0.4),
            motivation=Motivation(reflective=0.6, automatic=0.6, intrinsic=0.6, extrinsic=0.6),
            stage_of_change=StageOfChange.CONTEMPLATION,
        )

        diagnosis = model.diagnose(behavior)

        assert "component_scores" in diagnosis
        assert "limiting_factors" in diagnosis
        assert "primary_barrier" in diagnosis
        assert len(diagnosis["limiting_factors"]) > 0  # Should have some limiting factors

    def test_recommend_interventions(self):
        """Test intervention recommendations."""
        model = COMBModel()

        behavior = Behavior(
            name="test behavior",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.3, psychological=0.7, knowledge=0.8, skills=0.6),
            opportunity=Opportunity(physical=0.8, social=0.5, environmental=0.7, temporal=0.6),
            motivation=Motivation(reflective=0.7, automatic=0.4, intrinsic=0.6, extrinsic=0.5),
            stage_of_change=StageOfChange.PREPARATION,
        )

        interventions = model.recommend_interventions(behavior)

        assert len(interventions) > 0
        assert all("type" in i for i in interventions)
        assert all("strategies" in i for i in interventions)
        # Should target physical capability (lowest score)
        assert any("physical" in i["type"] for i in interventions)


class TestFoggBehaviorModel:
    """Test Fogg Behavior Model."""

    def test_predict_behavior(self):
        """Test behavior prediction."""
        model = FoggBehaviorModel()

        likelihood = model.predict_behavior(
            motivation=0.8,
            ability=0.7,
            prompt_effectiveness=0.6
        )

        assert 0 <= likelihood <= 1

    def test_diagnose(self):
        """Test diagnosis."""
        model = FoggBehaviorModel()

        behavior = Behavior(
            name="test",
            behavior_type=BehaviorType.HEALTH,
            capability=Capability(physical=0.7, psychological=0.6, knowledge=0.8, skills=0.7),
            opportunity=Opportunity(physical=0.5, social=0.5, environmental=0.5, temporal=0.5),
            motivation=Motivation(reflective=0.8, automatic=0.4, intrinsic=0.7, extrinsic=0.5),
            stage_of_change=StageOfChange.ACTION,
        )

        diagnosis = model.diagnose(behavior)

        assert "motivation" in diagnosis
        assert "ability" in diagnosis
        assert "recommended_prompt_type" in diagnosis


class TestEASTFramework:
    """Test EAST framework."""

    def test_predict_behavior(self):
        """Test behavior prediction."""
        framework = EASTFramework()

        likelihood = framework.predict_behavior(
            easy=0.8,
            attractive=0.7,
            social=0.6,
            timely=0.7
        )

        assert 0 <= likelihood <= 1

    def test_diagnose(self):
        """Test EAST diagnosis."""
        framework = EASTFramework()

        behavior = Behavior(
            name="test",
            behavior_type=BehaviorType.SOCIAL,
            capability=Capability(physical=0.6, psychological=0.7, knowledge=0.7, skills=0.6),
            opportunity=Opportunity(physical=0.5, social=0.4, environmental=0.5, temporal=0.6),
            motivation=Motivation(reflective=0.7, automatic=0.5, intrinsic=0.6, extrinsic=0.5),
            stage_of_change=StageOfChange.CONTEMPLATION,
        )

        diagnosis = framework.diagnose(behavior)

        assert "east_scores" in diagnosis
        assert len(diagnosis["east_scores"]) == 4
        assert all(0 <= v <= 1 for v in diagnosis["east_scores"].values())


class TestBehavioralAnalyzer:
    """Test behavioral analyzer."""

    def test_diagnose(self):
        """Test creating a behavior through diagnosis."""
        analyzer = BehavioralAnalyzer()

        behavior = analyzer.diagnose(
            behavior="daily meditation",
            capability={"physical": 0.8, "psychological": 0.6, "knowledge": 0.7, "skills": 0.5},
            opportunity={"physical": 0.7, "social": 0.5, "environmental": 0.6, "temporal": 0.4},
            motivation={"reflective": 0.8, "automatic": 0.4, "intrinsic": 0.7, "extrinsic": 0.3},
            behavior_type=BehaviorType.HEALTH,
            stage_of_change=StageOfChange.PREPARATION,
            current_frequency=2.0,
            desired_frequency=7.0,
        )

        assert behavior.name == "daily meditation"
        assert behavior.behavior_type == BehaviorType.HEALTH
        assert behavior.current_frequency == 2.0
        assert behavior.desired_frequency == 7.0

    def test_analyze(self):
        """Test comprehensive analysis."""
        analyzer = BehavioralAnalyzer()

        behavior = analyzer.diagnose(
            behavior="daily exercise",
            behavior_type=BehaviorType.HEALTH,
            stage_of_change=StageOfChange.ACTION,
        )

        analysis = analyzer.analyze(behavior, include_all_models=True)

        assert "behavior_name" in analysis
        assert "comb" in analysis
        assert "fogg" in analysis
        assert "east" in analysis
        assert "dual_process" in analysis
        assert "transtheoretical" in analysis
        assert "synthesis" in analysis

    def test_synthesis(self):
        """Test insight synthesis."""
        analyzer = BehavioralAnalyzer()

        behavior = analyzer.diagnose(
            behavior="test",
            behavior_type=BehaviorType.HEALTH,
        )

        analysis = analyzer.analyze(behavior)

        synthesis = analysis["synthesis"]

        assert "overall_behavior_likelihood" in synthesis
        assert "top_priority_interventions" in synthesis
        assert "recommended_approach" in synthesis
        assert "quick_wins" in synthesis
        assert len(synthesis["quick_wins"]) > 0
