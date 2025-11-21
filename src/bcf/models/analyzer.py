"""
Behavioral Analyzer - Integrates multiple behavioral models for comprehensive analysis.
"""

from typing import Any, Dict, List, Optional

from bcf.models.base import Behavior, BehaviorContext, BehaviorType, Capability, Motivation, Opportunity, StageOfChange
from bcf.models.comb import COMBModel
from bcf.models.dual_process import DualProcessModel
from bcf.models.east import EASTFramework
from bcf.models.fogg import FoggBehaviorModel, Prompt
from bcf.models.transtheoretical import TranstheoreticalModel


class BehavioralAnalyzer:
    """
    Comprehensive behavioral analyzer integrating multiple evidence-based models.

    Combines:
    - COM-B Model
    - Fogg Behavior Model
    - EAST Framework
    - Dual-Process Theory
    - Transtheoretical Model
    """

    def __init__(self):
        """Initialize the behavioral analyzer with all models."""
        self.comb_model = COMBModel()
        self.fogg_model = FoggBehaviorModel()
        self.east_framework = EASTFramework()
        self.dual_process_model = DualProcessModel()
        self.transtheoretical_model = TranstheoreticalModel()

    def diagnose(
        self,
        behavior: str,
        capability: Optional[Dict[str, float]] = None,
        opportunity: Optional[Dict[str, float]] = None,
        motivation: Optional[Dict[str, float]] = None,
        behavior_type: BehaviorType = BehaviorType.OTHER,
        stage_of_change: StageOfChange = StageOfChange.CONTEMPLATION,
        current_frequency: float = 0.0,
        desired_frequency: float = 7.0,
        barriers: Optional[List[str]] = None,
        facilitators: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        prompt: Optional[Prompt] = None,
    ) -> Behavior:
        """
        Perform comprehensive behavioral diagnosis.

        Args:
            behavior: Name/description of behavior
            capability: Dict with 'physical', 'psychological', 'knowledge', 'skills' (0-1)
            opportunity: Dict with 'physical', 'social', 'environmental', 'temporal' (0-1)
            motivation: Dict with 'reflective', 'automatic', 'intrinsic', 'extrinsic' (0-1)
            behavior_type: Type of behavior
            stage_of_change: Current stage in change process
            current_frequency: Current frequency (times per week)
            desired_frequency: Desired frequency (times per week)
            barriers: List of barriers
            facilitators: List of facilitators
            context: Behavioral context information
            prompt: Optional prompt information

        Returns:
            Behavior object with full diagnosis
        """
        # Create behavior components with defaults
        cap = Capability(**(capability or {
            "physical": 0.5,
            "psychological": 0.5,
            "knowledge": 0.5,
            "skills": 0.5,
        }))

        opp = Opportunity(**(opportunity or {
            "physical": 0.5,
            "social": 0.5,
            "environmental": 0.5,
            "temporal": 0.5,
        }))

        mot = Motivation(**(motivation or {
            "reflective": 0.5,
            "automatic": 0.5,
            "intrinsic": 0.5,
            "extrinsic": 0.5,
        }))

        # Create context
        ctx = BehaviorContext(**(context or {}))

        # Create behavior object
        behavior_obj = Behavior(
            name=behavior,
            behavior_type=behavior_type,
            capability=cap,
            opportunity=opp,
            motivation=mot,
            context=ctx,
            current_frequency=current_frequency,
            desired_frequency=desired_frequency,
            stage_of_change=stage_of_change,
            barriers=barriers or [],
            facilitators=facilitators or [],
        )

        return behavior_obj

    def analyze(
        self,
        behavior: Behavior,
        prompt: Optional[Prompt] = None,
        include_all_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-model analysis.

        Args:
            behavior: Behavior object to analyze
            prompt: Optional prompt information for Fogg model
            include_all_models: If True, run all models; else just core models

        Returns:
            Comprehensive analysis from all models
        """
        analysis = {
            "behavior_name": behavior.name,
            "behavior_type": behavior.behavior_type,
            "current_frequency": behavior.current_frequency,
            "desired_frequency": behavior.desired_frequency,
            "frequency_gap": behavior.desired_frequency - behavior.current_frequency,
        }

        # COM-B analysis (always included)
        analysis["comb"] = self.comb_model.diagnose(behavior)
        analysis["comb"]["interventions"] = self.comb_model.recommend_interventions(behavior)
        analysis["comb"]["change_potential"] = self.comb_model.behavior_change_potential(behavior)

        # Fogg model analysis
        if include_all_models:
            analysis["fogg"] = self.fogg_model.diagnose(behavior, prompt)
            analysis["fogg"]["interventions"] = self.fogg_model.recommend_interventions(behavior)

        # EAST framework analysis
        if include_all_models:
            analysis["east"] = self.east_framework.diagnose(behavior)
            analysis["east"]["interventions"] = self.east_framework.recommend_interventions(behavior)
            analysis["east"]["nudge_design"] = self.east_framework.design_nudge(behavior)

        # Dual-process analysis
        if include_all_models:
            analysis["dual_process"] = self.dual_process_model.diagnose(behavior)
            analysis["dual_process"]["interventions"] = self.dual_process_model.recommend_interventions(behavior)

        # Transtheoretical model analysis
        analysis["transtheoretical"] = self.transtheoretical_model.diagnose(behavior)
        analysis["transtheoretical"]["interventions"] = self.transtheoretical_model.recommend_interventions(behavior)

        # Synthesize insights
        analysis["synthesis"] = self._synthesize_insights(behavior, analysis)

        return analysis

    def _synthesize_insights(self, behavior: Behavior, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize insights across all models.

        Args:
            behavior: Behavior being analyzed
            analysis: Results from all models

        Returns:
            Synthesized insights and recommendations
        """
        # Aggregate priority interventions from all models
        all_interventions = []

        if "comb" in analysis:
            all_interventions.extend([(i, "COM-B") for i in analysis["comb"]["interventions"]])

        if "fogg" in analysis:
            all_interventions.extend([(i, "Fogg") for i in analysis["fogg"]["interventions"]])

        if "east" in analysis:
            all_interventions.extend([(i, "EAST") for i in analysis["east"]["interventions"]])

        if "dual_process" in analysis:
            all_interventions.extend([(i, "Dual-Process") for i in analysis["dual_process"]["interventions"]])

        if "transtheoretical" in analysis:
            all_interventions.extend([(i, "TTM") for i in analysis["transtheoretical"]["interventions"]])

        # Sort by priority across all models
        all_interventions.sort(key=lambda x: x[0].get("priority", 0), reverse=True)

        # Get top interventions
        top_interventions = [
            {
                "intervention": interv,
                "source_model": model,
                "priority": interv.get("priority", 0),
            }
            for interv, model in all_interventions[:10]
        ]

        # Identify convergent recommendations (mentioned by multiple models)
        strategy_counts: Dict[str, int] = {}
        for interv, _ in all_interventions:
            if "strategies" in interv:
                for strategy in interv["strategies"]:
                    strategy_key = strategy[:50]  # First 50 chars
                    strategy_counts[strategy_key] = strategy_counts.get(strategy_key, 0) + 1

        convergent_strategies = [
            strategy for strategy, count in strategy_counts.items()
            if count >= 2
        ]

        # Overall behavior likelihood (average across models)
        likelihoods = []
        if "comb" in analysis:
            likelihoods.append(analysis["comb"]["behavior_likelihood"])
        if "fogg" in analysis:
            likelihoods.append(analysis["fogg"]["behavior_likelihood"])

        avg_likelihood = sum(likelihoods) / len(likelihoods) if likelihoods else 0.5

        # Critical barriers (mentioned across models)
        critical_barriers = set()
        if "comb" in analysis:
            critical_barriers.update(analysis["comb"]["limiting_factors"].keys())
        if "fogg" in analysis:
            critical_barriers.update(analysis["fogg"]["limiting_factors"])

        return {
            "overall_behavior_likelihood": avg_likelihood,
            "readiness_assessment": behavior.readiness_score,
            "top_priority_interventions": top_interventions,
            "convergent_strategies": convergent_strategies,
            "critical_barriers": list(critical_barriers),
            "recommended_approach": self._recommend_overall_approach(behavior, analysis),
            "success_factors": self._identify_success_factors(behavior, analysis),
            "quick_wins": self._identify_quick_wins(behavior, analysis),
        }

    def _recommend_overall_approach(self, behavior: Behavior, analysis: Dict[str, Any]) -> str:
        """Recommend overall intervention approach based on all models."""
        stage = behavior.stage_of_change

        # Early stages: focus on awareness and motivation
        if stage in [StageOfChange.PRECONTEMPLATION, StageOfChange.CONTEMPLATION]:
            return (
                "Focus on building awareness and motivation. Use emotional appeals, "
                "social proof, and consciousness-raising before pushing for action."
            )

        # Preparation: focus on planning and capability
        elif stage == StageOfChange.PREPARATION:
            return (
                "Focus on concrete planning and building capability. Create detailed "
                "implementation intentions and ensure resources are in place."
            )

        # Action: focus on support and simplification
        elif stage == StageOfChange.ACTION:
            return (
                "Focus on ongoing support and making it as easy as possible. Use "
                "environmental restructuring, prompts, and reinforcement."
            )

        # Maintenance: focus on habit formation and relapse prevention
        else:
            return (
                "Focus on habit formation and relapse prevention. Build automatic "
                "routines and prepare coping strategies for high-risk situations."
            )

    def _identify_success_factors(self, behavior: Behavior, analysis: Dict[str, Any]) -> List[str]:
        """Identify factors that support behavior change success."""
        factors = []

        # High capability
        if behavior.capability.overall > 0.6:
            factors.append("Strong capability - person has the skills and knowledge")

        # High motivation
        if behavior.motivation.overall > 0.6:
            factors.append("High motivation - person wants to change")

        # Good opportunity
        if behavior.opportunity.overall > 0.6:
            factors.append("Favorable opportunity - environment supports change")

        # Advanced stage
        if behavior.stage_of_change in [StageOfChange.ACTION, StageOfChange.MAINTENANCE]:
            factors.append("Advanced stage - already taking action")

        # Facilitators present
        if behavior.facilitators:
            factors.append(f"Facilitators present: {', '.join(behavior.facilitators[:3])}")

        if not factors:
            factors.append("Opportunities for improvement identified across all areas")

        return factors

    def _identify_quick_wins(self, behavior: Behavior, analysis: Dict[str, Any]) -> List[str]:
        """Identify quick win interventions that can be implemented immediately."""
        quick_wins = []

        # Easy EAST interventions
        if "east" in analysis and analysis["east"]["east_scores"]["easy"] < 0.6:
            quick_wins.append("Remove one major barrier to make behavior easier")

        # Prompt optimization
        if "fogg" in analysis and analysis["fogg"]["prompt_effectiveness"] < 0.5:
            quick_wins.append("Set up daily reminders/prompts at optimal times")

        # Social support
        if behavior.opportunity.social < 0.5:
            quick_wins.append("Find an accountability partner or join a group")

        # Implementation intentions
        if behavior.stage_of_change == StageOfChange.PREPARATION:
            quick_wins.append("Create specific if-then plan (e.g., 'If Monday morning, then exercise')")

        # Environmental cue
        quick_wins.append("Set up environmental cue (make it visible and easy to start)")

        return quick_wins[:5]  # Top 5 quick wins
