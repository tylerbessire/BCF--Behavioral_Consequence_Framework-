"""
Behavioral Consequence Framework (BCF)

A comprehensive framework for analyzing, predicting, and optimizing
behavioral interventions and their consequences.
"""

__version__ = "0.1.0"

from bcf.models.analyzer import BehavioralAnalyzer
from bcf.consequence.predictor import ConsequencePredictor
from bcf.intervention.optimizer import InterventionOptimizer

__all__ = [
    "BehavioralAnalyzer",
    "ConsequencePredictor",
    "InterventionOptimizer",
]
