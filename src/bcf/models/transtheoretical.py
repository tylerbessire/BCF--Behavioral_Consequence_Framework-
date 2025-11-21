"""
Transtheoretical Model (Stages of Change) Implementation

The Transtheoretical Model describes behavior change as a process through stages:
1. Precontemplation: Not considering change
2. Contemplation: Thinking about change
3. Preparation: Planning to change soon
4. Action: Actively changing
5. Maintenance: Sustaining change
6. Termination: Change fully integrated (optional stage)

Reference:
Prochaska, J. O., & DiClemente, C. C. (1983). Stages and processes of
self-change of smoking: Toward an integrative model of change.
Journal of Consulting and Clinical Psychology, 51(3), 390-395.
"""

from typing import Any, Dict, List

from bcf.models.base import Behavior, BehavioralModel, StageOfChange


class TranstheoreticalModel(BehavioralModel):
    """
    Transtheoretical Model (Stages of Change).

    Recognizes that behavior change is a process, not an event.
    Interventions should be matched to the person's stage of change.
    """

    # Processes of change mapped to stages where they're most effective
    PROCESSES_BY_STAGE = {
        StageOfChange.PRECONTEMPLATION: [
            "Consciousness raising",
            "Dramatic relief",
            "Environmental reevaluation",
        ],
        StageOfChange.CONTEMPLATION: [
            "Self-reevaluation",
            "Environmental reevaluation",
        ],
        StageOfChange.PREPARATION: [
            "Self-liberation",
            "Helping relationships",
        ],
        StageOfChange.ACTION: [
            "Counterconditioning",
            "Stimulus control",
            "Reinforcement management",
            "Helping relationships",
        ],
        StageOfChange.MAINTENANCE: [
            "Stimulus control",
            "Reinforcement management",
            "Helping relationships",
        ],
    }

    def __init__(self):
        """Initialize Transtheoretical Model."""
        pass

    def predict_behavior(self, stage: StageOfChange, time_in_stage: int = 1) -> float:
        """
        Predict likelihood of behavior based on stage of change.

        Args:
            stage: Current stage of change
            time_in_stage: Months in current stage (affects regression risk)

        Returns:
            Likelihood of performing behavior (0-1)
        """
        base_probabilities = {
            StageOfChange.PRECONTEMPLATION: 0.05,
            StageOfChange.CONTEMPLATION: 0.20,
            StageOfChange.PREPARATION: 0.40,
            StageOfChange.ACTION: 0.70,
            StageOfChange.MAINTENANCE: 0.85,
            StageOfChange.TERMINATION: 0.95,
        }

        base_prob = base_probabilities[stage]

        # Adjust for time in stage (too long = possible regression)
        if stage in [StageOfChange.ACTION, StageOfChange.MAINTENANCE]:
            if time_in_stage > 6:  # More than 6 months
                # Slight decrease in probability (potential burnout)
                base_prob *= 0.95

        return base_prob

    def diagnose(self, behavior: Behavior) -> Dict[str, Any]:
        """
        Diagnose stage of change and readiness for progression.

        Args:
            behavior: Behavior to analyze

        Returns:
            Stage diagnosis and progression readiness
        """
        stage = behavior.stage_of_change

        # Assess readiness to progress to next stage
        readiness = self._assess_progression_readiness(behavior)

        # Identify appropriate processes of change
        recommended_processes = self.PROCESSES_BY_STAGE.get(stage, [])

        # Assess risk of regression
        regression_risk = self._assess_regression_risk(behavior)

        # Next stage
        next_stage = self._get_next_stage(stage)

        return {
            "current_stage": stage,
            "stage_description": self._get_stage_description(stage),
            "progression_readiness": readiness,
            "recommended_processes": recommended_processes,
            "regression_risk": regression_risk,
            "next_stage": next_stage,
            "barriers_to_progression": self._identify_barriers(behavior),
            "facilitators_to_progression": behavior.facilitators,
        }

    def recommend_interventions(self, behavior: Behavior) -> List[Dict[str, Any]]:
        """
        Recommend stage-appropriate interventions.

        Args:
            behavior: Target behavior

        Returns:
            List of stage-matched intervention recommendations
        """
        diagnosis = self.diagnose(behavior)
        stage = behavior.stage_of_change
        interventions = []

        # Precontemplation stage
        if stage == StageOfChange.PRECONTEMPLATION:
            interventions.append({
                "type": "precontemplation",
                "stage": "Precontemplation",
                "goal": "Raise awareness and build motivation to consider change",
                "strategies": [
                    "Provide personalized feedback and information",
                    "Increase awareness of risks of current behavior",
                    "Highlight discrepancy between values and behavior",
                    "Use motivational interviewing techniques",
                    "Avoid pressuring or pushing for immediate change",
                ],
                "processes": ["Consciousness raising", "Dramatic relief"],
                "priority": 0.9,
            })

        # Contemplation stage
        elif stage == StageOfChange.CONTEMPLATION:
            interventions.append({
                "type": "contemplation",
                "stage": "Contemplation",
                "goal": "Tip decisional balance toward change",
                "strategies": [
                    "Explore pros and cons of changing vs. not changing",
                    "Address ambivalence",
                    "Highlight benefits of change",
                    "Reduce perceived barriers",
                    "Build self-efficacy through small successes",
                ],
                "processes": ["Self-reevaluation", "Environmental reevaluation"],
                "priority": 0.85,
            })

        # Preparation stage
        elif stage == StageOfChange.PREPARATION:
            interventions.append({
                "type": "preparation",
                "stage": "Preparation",
                "goal": "Develop concrete action plan",
                "strategies": [
                    "Set specific, measurable goals",
                    "Create detailed implementation intentions",
                    "Identify and prepare resources needed",
                    "Establish social support",
                    "Make public commitment",
                    "Set start date",
                ],
                "processes": ["Self-liberation", "Helping relationships"],
                "priority": 0.8,
            })

        # Action stage
        elif stage == StageOfChange.ACTION:
            interventions.append({
                "type": "action",
                "stage": "Action",
                "goal": "Support initial behavior change and prevent relapse",
                "strategies": [
                    "Provide ongoing support and encouragement",
                    "Monitor progress closely",
                    "Reinforce successes",
                    "Modify environment to support new behavior",
                    "Develop coping strategies for high-risk situations",
                    "Problem-solve barriers as they arise",
                ],
                "processes": [
                    "Counterconditioning",
                    "Stimulus control",
                    "Reinforcement management",
                ],
                "priority": 0.75,
            })

        # Maintenance stage
        elif stage == StageOfChange.MAINTENANCE:
            interventions.append({
                "type": "maintenance",
                "stage": "Maintenance",
                "goal": "Prevent relapse and consolidate gains",
                "strategies": [
                    "Continue reinforcement of new behavior",
                    "Develop relapse prevention strategies",
                    "Maintain environmental supports",
                    "Build strong habit patterns",
                    "Prepare for high-risk situations",
                    "Celebrate long-term success",
                ],
                "processes": ["Stimulus control", "Reinforcement management"],
                "priority": 0.7,
            })

        # Add progression-specific interventions
        if diagnosis["progression_readiness"] > 0.6:
            interventions.append({
                "type": "stage_progression",
                "goal": f"Support progression to {diagnosis['next_stage']}",
                "strategies": self._get_progression_strategies(stage),
                "priority": 0.85,
            })

        # Add regression prevention if high risk
        if diagnosis["regression_risk"] > 0.5:
            interventions.append({
                "type": "regression_prevention",
                "goal": "Prevent regression to earlier stage",
                "strategies": [
                    "Identify and address warning signs",
                    "Strengthen coping mechanisms",
                    "Reinforce commitment and motivation",
                    "Enhance social support",
                    "Prepare relapse recovery plan",
                ],
                "priority": diagnosis["regression_risk"],
            })

        # Sort by priority
        interventions.sort(key=lambda x: x["priority"], reverse=True)

        return interventions

    def _assess_progression_readiness(self, behavior: Behavior) -> float:
        """Assess readiness to progress to next stage."""
        stage = behavior.stage_of_change

        if stage == StageOfChange.PRECONTEMPLATION:
            # Ready if starting to think about it (motivation increasing)
            return behavior.motivation.reflective * 0.7

        elif stage == StageOfChange.CONTEMPLATION:
            # Ready if motivation high and barriers addressed
            return (behavior.motivation.overall * 0.6 +
                    behavior.capability.overall * 0.4)

        elif stage == StageOfChange.PREPARATION:
            # Ready if plan is in place and resources available
            return (behavior.opportunity.overall * 0.5 +
                    behavior.capability.overall * 0.5)

        elif stage == StageOfChange.ACTION:
            # Ready to move to maintenance if sustained for period
            # (would need temporal data - using proxy)
            return min(1.0, behavior.current_frequency / max(1, behavior.desired_frequency))

        elif stage == StageOfChange.MAINTENANCE:
            # Ready for termination if fully integrated
            return behavior.motivation.automatic * behavior.readiness_score

        else:  # Termination
            return 1.0

    def _assess_regression_risk(self, behavior: Behavior) -> float:
        """Assess risk of regressing to earlier stage."""
        stage = behavior.stage_of_change

        # Higher stages have regression risk
        if stage in [StageOfChange.PRECONTEMPLATION, StageOfChange.CONTEMPLATION]:
            return 0.1  # Low risk (can't go much lower)

        # Risk factors
        risk = 0.0

        # Low motivation
        if behavior.motivation.overall < 0.5:
            risk += 0.3

        # Barriers present
        risk += len(behavior.barriers) * 0.1

        # Low opportunity
        if behavior.opportunity.overall < 0.4:
            risk += 0.2

        # Not meeting desired frequency (for action/maintenance)
        if stage in [StageOfChange.ACTION, StageOfChange.MAINTENANCE]:
            if behavior.current_frequency < behavior.desired_frequency * 0.7:
                risk += 0.3

        return min(1.0, risk)

    def _get_next_stage(self, current_stage: StageOfChange) -> StageOfChange:
        """Get next stage in progression."""
        stage_order = [
            StageOfChange.PRECONTEMPLATION,
            StageOfChange.CONTEMPLATION,
            StageOfChange.PREPARATION,
            StageOfChange.ACTION,
            StageOfChange.MAINTENANCE,
            StageOfChange.TERMINATION,
        ]

        current_index = stage_order.index(current_stage)
        if current_index < len(stage_order) - 1:
            return stage_order[current_index + 1]
        else:
            return current_stage  # Already at final stage

    def _get_stage_description(self, stage: StageOfChange) -> str:
        """Get description of stage."""
        descriptions = {
            StageOfChange.PRECONTEMPLATION: "Not considering change; unaware or unmotivated",
            StageOfChange.CONTEMPLATION: "Thinking about change; weighing pros and cons",
            StageOfChange.PREPARATION: "Planning to change soon; getting ready to act",
            StageOfChange.ACTION: "Actively making changes; new behavior implemented",
            StageOfChange.MAINTENANCE: "Sustaining change; preventing relapse",
            StageOfChange.TERMINATION: "Change fully integrated; no temptation to regress",
        }
        return descriptions[stage]

    def _identify_barriers(self, behavior: Behavior) -> List[str]:
        """Identify barriers to stage progression."""
        barriers = list(behavior.barriers)  # Start with explicit barriers

        # Add implicit barriers based on low scores
        if behavior.motivation.overall < 0.5:
            barriers.append("Low motivation")

        if behavior.capability.overall < 0.5:
            barriers.append("Limited capability/self-efficacy")

        if behavior.opportunity.overall < 0.5:
            barriers.append("Environmental/social barriers")

        return barriers

    def _get_progression_strategies(self, current_stage: StageOfChange) -> List[str]:
        """Get strategies for progressing to next stage."""
        strategies = {
            StageOfChange.PRECONTEMPLATION: [
                "Personalized risk assessment",
                "Consciousness-raising activities",
                "Reflective exercises on values alignment",
            ],
            StageOfChange.CONTEMPLATION: [
                "Decisional balance worksheet",
                "Small behavior experiments",
                "Connect with others who have changed",
            ],
            StageOfChange.PREPARATION: [
                "Set specific start date",
                "Gather necessary resources",
                "Create detailed action plan",
            ],
            StageOfChange.ACTION: [
                "Consistent daily practice",
                "Regular self-monitoring",
                "Celebrate small wins",
            ],
            StageOfChange.MAINTENANCE: [
                "Continue beyond 6 months",
                "Handle lapses without relapse",
                "Integrate into identity",
            ],
        }
        return strategies.get(current_stage, [])
