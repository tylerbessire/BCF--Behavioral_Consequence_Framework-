"""Base classes and data structures for behavioral models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator


class BehaviorType(str, Enum):
    """Types of behaviors."""

    HEALTH = "health"
    FINANCIAL = "financial"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    OTHER = "other"


class StageOfChange(str, Enum):
    """Stages in the Transtheoretical Model."""

    PRECONTEMPLATION = "precontemplation"
    CONTEMPLATION = "contemplation"
    PREPARATION = "preparation"
    ACTION = "action"
    MAINTENANCE = "maintenance"
    TERMINATION = "termination"


class BehaviorContext(BaseModel):
    """Context in which a behavior occurs."""

    physical_environment: Dict[str, float] = Field(
        default_factory=dict, description="Physical environment factors (0-1 scale)"
    )
    social_environment: Dict[str, float] = Field(
        default_factory=dict, description="Social environment factors (0-1 scale)"
    )
    temporal_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Time-related factors"
    )
    cultural_factors: Dict[str, float] = Field(
        default_factory=dict, description="Cultural factors (0-1 scale)"
    )

    @field_validator("physical_environment", "social_environment", "cultural_factors")
    @classmethod
    def validate_scale(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that values are between 0 and 1."""
        for key, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Value for {key} must be between 0 and 1, got {value}")
        return v


class Capability(BaseModel):
    """Capability component of COM-B model."""

    physical: float = Field(ge=0.0, le=1.0, description="Physical capability (0-1)")
    psychological: float = Field(ge=0.0, le=1.0, description="Psychological capability (0-1)")
    knowledge: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Knowledge level (0-1)"
    )
    skills: float = Field(default=0.5, ge=0.0, le=1.0, description="Skill level (0-1)")

    @property
    def overall(self) -> float:
        """Calculate overall capability score."""
        return np.mean([self.physical, self.psychological, self.knowledge, self.skills])


class Opportunity(BaseModel):
    """Opportunity component of COM-B model."""

    physical: float = Field(
        ge=0.0, le=1.0, description="Physical opportunity (access, resources) (0-1)"
    )
    social: float = Field(
        ge=0.0, le=1.0, description="Social opportunity (norms, support) (0-1)"
    )
    environmental: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Environmental enablers (0-1)"
    )
    temporal: float = Field(default=0.5, ge=0.0, le=1.0, description="Time availability (0-1)")

    @property
    def overall(self) -> float:
        """Calculate overall opportunity score."""
        return np.mean([self.physical, self.social, self.environmental, self.temporal])


class Motivation(BaseModel):
    """Motivation component of COM-B model."""

    reflective: float = Field(
        ge=0.0, le=1.0, description="Reflective motivation (conscious, planned) (0-1)"
    )
    automatic: float = Field(
        ge=0.0, le=1.0, description="Automatic motivation (emotional, habitual) (0-1)"
    )
    intrinsic: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Intrinsic motivation (0-1)"
    )
    extrinsic: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Extrinsic motivation (0-1)"
    )

    @property
    def overall(self) -> float:
        """Calculate overall motivation score."""
        return np.mean([self.reflective, self.automatic, self.intrinsic, self.extrinsic])


@dataclass
class Behavior:
    """Represents a behavior and its characteristics."""

    name: str
    behavior_type: BehaviorType
    capability: Capability
    opportunity: Opportunity
    motivation: Motivation
    context: BehaviorContext = field(default_factory=BehaviorContext)
    current_frequency: float = 0.0  # times per week
    desired_frequency: float = 0.0  # times per week
    stage_of_change: StageOfChange = StageOfChange.PRECONTEMPLATION
    barriers: List[str] = field(default_factory=list)
    facilitators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def behavior_likelihood(self) -> float:
        """
        Calculate likelihood of behavior occurring based on COM-B.
        Uses a multiplicative model where all three components are necessary.
        """
        # Weighted combination favoring motivation slightly
        return (
            self.capability.overall ** 0.3
            * self.opportunity.overall ** 0.3
            * self.motivation.overall ** 0.4
        )

    @property
    def readiness_score(self) -> float:
        """Calculate readiness for behavior change based on stage."""
        stage_scores = {
            StageOfChange.PRECONTEMPLATION: 0.1,
            StageOfChange.CONTEMPLATION: 0.3,
            StageOfChange.PREPARATION: 0.5,
            StageOfChange.ACTION: 0.7,
            StageOfChange.MAINTENANCE: 0.9,
            StageOfChange.TERMINATION: 1.0,
        }
        return stage_scores[self.stage_of_change]


class BehavioralModel(ABC):
    """Abstract base class for behavioral models."""

    @abstractmethod
    def predict_behavior(self, **kwargs: Any) -> float:
        """
        Predict likelihood of behavior occurring.

        Returns:
            float: Probability between 0 and 1
        """
        pass

    @abstractmethod
    def diagnose(self, behavior: Behavior) -> Dict[str, Any]:
        """
        Diagnose behavioral determinants.

        Args:
            behavior: Behavior to analyze

        Returns:
            Dict with diagnostic information
        """
        pass

    @abstractmethod
    def recommend_interventions(self, behavior: Behavior) -> List[Dict[str, Any]]:
        """
        Recommend interventions to change behavior.

        Args:
            behavior: Target behavior

        Returns:
            List of intervention recommendations
        """
        pass
