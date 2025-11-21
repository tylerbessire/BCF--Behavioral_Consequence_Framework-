"""
Example: Analyzing and Optimizing Exercise Behavior

This example demonstrates the full capabilities of the BCF framework
by analyzing a person's exercise behavior and designing an optimized intervention.
"""

from bcf import BehavioralAnalyzer, ConsequencePredictor, InterventionOptimizer
from bcf.models.base import BehaviorType, StageOfChange


def main():
    """Run comprehensive behavioral analysis and intervention design for exercise."""

    print("=" * 80)
    print("BEHAVIORAL CONSEQUENCE FRAMEWORK - EXERCISE BEHAVIOR EXAMPLE")
    print("=" * 80)
    print()

    # Step 1: Define the behavior and current state
    print("Step 1: Diagnosing Current Behavioral State")
    print("-" * 80)

    analyzer = BehavioralAnalyzer()

    behavior = analyzer.diagnose(
        behavior="daily exercise",
        behavior_type=BehaviorType.HEALTH,
        capability={
            "physical": 0.7,  # Physically able
            "psychological": 0.6,  # Some mental barriers
            "knowledge": 0.8,  # Knows how to exercise
            "skills": 0.7,  # Has basic fitness skills
        },
        opportunity={
            "physical": 0.5,  # Limited gym access
            "social": 0.4,  # Low social support
            "environmental": 0.6,  # Decent environment
            "temporal": 0.3,  # Time-constrained
        },
        motivation={
            "reflective": 0.8,  # Wants to exercise (conscious)
            "automatic": 0.3,  # But doesn't feel like it (emotional)
            "intrinsic": 0.7,  # Enjoys feeling healthy
            "extrinsic": 0.5,  # Some external motivation
        },
        stage_of_change=StageOfChange.PREPARATION,
        current_frequency=1.0,  # Once per week currently
        desired_frequency=5.0,  # Wants 5x per week
        barriers=["time", "tired after work", "no gym nearby"],
        facilitators=["knows it's important", "has exercise clothes"],
    )

    print(f"Behavior: {behavior.name}")
    print(f"Current frequency: {behavior.current_frequency}x/week")
    print(f"Desired frequency: {behavior.desired_frequency}x/week")
    print(f"Stage of change: {behavior.stage_of_change.value}")
    print(f"Overall behavior likelihood: {behavior.behavior_likelihood:.2%}")
    print()

    # Step 2: Multi-model behavioral analysis
    print("Step 2: Comprehensive Multi-Model Analysis")
    print("-" * 80)

    analysis = analyzer.analyze(behavior, include_all_models=True)

    # COM-B insights
    print("COM-B Model Analysis:")
    print(f"  Capability: {behavior.capability.overall:.2f}")
    print(f"  Opportunity: {behavior.opportunity.overall:.2f}")
    print(f"  Motivation: {behavior.motivation.overall:.2f}")
    print(f"  Primary barrier: {analysis['comb']['primary_barrier']['component']}")
    print()

    # EAST insights
    print("EAST Framework Analysis:")
    for component, score in analysis["east"]["east_scores"].items():
        print(f"  {component.capitalize()}: {score:.2f}")
    print(f"  Weakest component: {analysis['east']['weakest_component']}")
    print()

    # Synthesized insights
    print("Synthesized Insights:")
    print(f"  Overall behavior likelihood: {analysis['synthesis']['overall_behavior_likelihood']:.2%}")
    print(f"  Critical barriers: {', '.join(analysis['synthesis']['critical_barriers'])}")
    print(f"  Quick wins: {analysis['synthesis']['quick_wins'][0]}")
    print()

    # Step 3: Consequence prediction
    print("Step 3: Predicting Consequences")
    print("-" * 80)

    predictor = ConsequencePredictor()
    consequences = predictor.predict(behavior)

    print(f"Total consequences identified: {len(consequences.consequences)}")
    print(f"Net expected value: {consequences.net_expected_value:+.2f}")
    print(f"Total risk score: {consequences.total_risk_score:.2f}")
    print(f"Dominant dimension: {consequences.dominant_dimension.value if consequences.dominant_dimension else 'N/A'}")
    print()

    print("Time profile (expected value by time horizon):")
    for horizon, value in consequences.time_profile.items():
        print(f"  {horizon.value}: {value:+.2f}")
    print()

    # High-value consequences
    high_value_consequences = [
        c for c in consequences.consequences
        if abs(c.expected_value) > 0.3
    ][:5]

    if high_value_consequences:
        print("Top 5 High-Value Consequences:")
        for i, c in enumerate(high_value_consequences, 1):
            sign = "+" if c.magnitude > 0 else "-"
            print(f"  {i}. [{sign}] {c.dimension.value} - {c.time_horizon.value}")
            print(f"     Magnitude: {c.magnitude:+.2f}, Probability: {c.probability:.2%}")
    print()

    # Step 4: Intervention optimization
    print("Step 4: Designing Optimized Intervention")
    print("-" * 80)

    optimizer = InterventionOptimizer(analyzer=analyzer)

    intervention = optimizer.design(
        target_behavior="daily exercise",
        current_state=behavior,
        desired_outcomes={
            "capability": 0.8,
            "opportunity": 0.7,
            "motivation": 0.8,
            "adherence": 0.85,
        },
        constraints={
            "cost": 100,  # Max $100
            "time": 45,  # Max 45 min/day
        },
    )

    print(f"Intervention: {intervention.name}")
    print(f"Success probability: {intervention.success_probability:.2%}")
    print(f"Total cost: ${intervention.total_cost:.2f}")
    print(f"Time required: {intervention.total_time:.1f} min/day")
    print()

    print("Selected Intervention Components:")
    for i, component in enumerate(intervention.components, 1):
        print(f"  {i}. {component.strategy.value.upper()}")
        print(f"     {component.description}")
        print(f"     Effect size: {component.expected_effect_size:.2f}, Cost: ${component.cost:.2f}")
    print()

    print("Expected Outcomes:")
    for outcome, value in intervention.expected_outcomes.items():
        print(f"  {outcome.capitalize()}: {value:.2f}")
    print()

    print("Implementation Plan:")
    for step in intervention.implementation_plan[:5]:
        print(f"  {step}")
    print(f"  ... ({len(intervention.implementation_plan)} total steps)")
    print()

    print("Monitoring Metrics:")
    for metric in intervention.monitoring_metrics:
        print(f"  - {metric}")
    print()

    # Step 5: Top recommendations
    print("Step 5: Top Priority Recommendations")
    print("-" * 80)

    top_interventions = analysis["synthesis"]["top_priority_interventions"][:5]

    print("Top 5 Evidence-Based Strategies (across all models):")
    for i, item in enumerate(top_interventions, 1):
        interv = item["intervention"]
        print(f"\n{i}. {interv.get('type', interv.get('target', 'N/A'))} (from {item['source_model']})")
        print(f"   Priority: {item['priority']:.2f}")
        if "strategies" in interv:
            print(f"   Key strategies:")
            for strategy in interv["strategies"][:3]:
                print(f"   - {strategy}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
