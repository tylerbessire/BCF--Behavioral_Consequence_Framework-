# Behavioral Consequence Framework (BCF)

A comprehensive, state-of-the-art framework for analyzing, predicting, and optimizing behavioral interventions and their consequences using modern behavioral science, cognitive psychology, and machine learning.

## Overview

The Behavioral Consequence Framework integrates multiple evidence-based behavioral models to provide a holistic approach to understanding and influencing human behavior. It combines classical behavioral economics with modern computational methods to predict consequences and optimize interventions.

## Core Features

### 1. Multi-Model Behavioral Analysis
- **COM-B Model**: Capability, Opportunity, Motivation → Behavior
- **BJ Fogg Behavior Model**: B = MAP (Motivation × Ability × Prompt)
- **EAST Framework**: Easy, Attractive, Social, Timely
- **Dual-Process Theory**: System 1 (automatic) vs System 2 (deliberative)
- **Transtheoretical Model**: Stages of change (Precontemplation → Maintenance)

### 2. Advanced Consequence Prediction
- Multi-dimensional consequence analysis (temporal, social, individual, systemic)
- Probabilistic outcome modeling with uncertainty quantification
- Network effects and cascading consequences
- Temporal dynamics: immediate vs delayed consequences
- Value-aligned consequence evaluation

### 3. Behavioral Intervention Optimizer
- Data-driven intervention design
- A/B testing framework with causal inference
- Reinforcement learning for adaptive interventions
- Context-aware recommendation system
- Ethical constraint integration

### 4. Habit Formation & Change
- Habit loop analysis (Cue → Routine → Reward)
- Implementation intention support
- Behavioral momentum modeling
- Relapse prediction and prevention

### 5. Cognitive Biases & Heuristics
- Comprehensive bias detection (50+ documented biases)
- Debiasing strategies
- Choice architecture optimization
- Nudge design and evaluation

## Installation

```bash
pip install behavioral-consequence-framework
```

Or from source:

```bash
git clone https://github.com/yourusername/BCF--Behavioral_Consequence_Framework-.git
cd BCF--Behavioral_Consequence_Framework-
pip install -e .
```

## Quick Start

```python
from bcf import BehavioralAnalyzer, ConsequencePredictor, InterventionOptimizer

# Analyze a behavior
analyzer = BehavioralAnalyzer()
behavior = analyzer.diagnose(
    behavior="daily exercise",
    capability={"physical": 0.7, "psychological": 0.8},
    opportunity={"social": 0.6, "physical": 0.5},
    motivation={"reflective": 0.9, "automatic": 0.4}
)

# Predict consequences
predictor = ConsequencePredictor()
consequences = predictor.predict(
    behavior=behavior,
    time_horizons=["immediate", "short_term", "long_term"],
    dimensions=["health", "social", "financial", "psychological"]
)

# Optimize intervention
optimizer = InterventionOptimizer()
intervention = optimizer.design(
    target_behavior="daily exercise",
    current_state=behavior,
    desired_outcomes={"health": 0.9, "adherence": 0.8},
    constraints={"cost": 100, "time": 30}  # max cost $100, 30 min/day
)

print(f"Recommended intervention: {intervention.strategy}")
print(f"Predicted success rate: {intervention.success_probability:.2%}")
print(f"Expected consequences: {intervention.expected_consequences}")
```

## Architecture

```
bcf/
├── models/
│   ├── comb.py              # COM-B model implementation
│   ├── fogg.py              # BJ Fogg behavior model
│   ├── east.py              # EAST framework
│   ├── dual_process.py      # System 1/2 cognitive processing
│   └── transtheoretical.py  # Stages of change model
├── consequence/
│   ├── predictor.py         # Consequence prediction engine
│   ├── temporal.py          # Temporal consequence modeling
│   ├── network.py           # Social/network consequences
│   └── evaluator.py         # Value-based consequence evaluation
├── intervention/
│   ├── optimizer.py         # Intervention optimization
│   ├── nudge.py            # Nudge design tools
│   ├── ab_testing.py       # A/B testing framework
│   └── rl_agent.py         # Reinforcement learning agent
├── cognitive/
│   ├── biases.py           # Cognitive biases catalog
│   ├── heuristics.py       # Decision heuristics
│   └── debiasing.py        # Debiasing strategies
├── habits/
│   ├── formation.py        # Habit formation modeling
│   ├── loops.py            # Habit loop analysis
│   └── change.py           # Habit change strategies
└── ethics/
    ├── constraints.py      # Ethical constraints
    ├── fairness.py        # Fairness evaluation
    └── transparency.py    # Explainable interventions
```

## Key Concepts

### COM-B Model
The framework uses the COM-B model as a foundation:
- **Capability**: Physical and psychological ability
- **Opportunity**: Social and physical environment
- **Motivation**: Reflective (conscious) and automatic (emotional)

### Consequence Dimensions
Consequences are evaluated across multiple dimensions:
- **Temporal**: Immediate, short-term, long-term
- **Scope**: Individual, interpersonal, societal, global
- **Type**: Physical, psychological, social, economic, environmental
- **Certainty**: Deterministic vs probabilistic outcomes

### Intervention Principles
Evidence-based intervention design follows:
1. **EAST**: Make it Easy, Attractive, Social, Timely
2. **Behavioral Economics**: Leverage defaults, framing, anchoring
3. **Self-Determination Theory**: Support autonomy, competence, relatedness
4. **Implementation Intentions**: Specific if-then plans
5. **Feedback Loops**: Real-time progress tracking

## Research Foundation

This framework synthesizes research from:
- Behavioral Economics (Kahneman, Tversky, Thaler)
- Health Psychology (Michie et al. - COM-B)
- Persuasive Technology (BJ Fogg)
- Nudge Theory (Thaler & Sunstein)
- Machine Learning & AI (RL, causal inference)
- Cognitive Science (Dual-process theory)

## Use Cases

- **Health Behavior Change**: Exercise, diet, medication adherence
- **Product Design**: User engagement and retention
- **Public Policy**: Nudge design and evaluation
- **Education**: Learning behavior optimization
- **Sustainability**: Pro-environmental behavior promotion
- **Organizational**: Employee productivity and wellbeing

## Ethical Considerations

BCF includes built-in ethical safeguards:
- Transparency in behavioral influence
- User autonomy preservation
- Fairness across demographic groups
- Privacy protection
- Harm prevention
- Value alignment checks

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{bcf2025,
  title={Behavioral Consequence Framework},
  author={BCF Contributors},
  year={2025},
  url={https://github.com/yourusername/BCF--Behavioral_Consequence_Framework-}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

1. Michie, S., et al. (2011). The behaviour change wheel: A new method for characterising and designing behaviour change interventions. Implementation Science, 6(1), 42.
2. Fogg, B. J. (2009). A behavior model for persuasive design. Persuasive Technology, 40-47.
3. Kahneman, D. (2011). Thinking, Fast and Slow. Macmillan.
4. Thaler, R. H., & Sunstein, C. R. (2009). Nudge: Improving decisions about health, wealth, and happiness. Penguin.
5. Deci, E. L., & Ryan, R. M. (2000). The "what" and "why" of goal pursuits: Human needs and the self-determination of behavior. Psychological Inquiry, 11(4), 227-268.
6. Lally, P., et al. (2010). How are habits formed: Modelling habit formation in the real world. European Journal of Social Psychology, 40(6), 998-1009.
7. Prochaska, J. O., & Velicer, W. F. (1997). The transtheoretical model of health behavior change. American Journal of Health Promotion, 12(1), 38-48.

## Contact

For questions and support, please open an issue on GitHub.
