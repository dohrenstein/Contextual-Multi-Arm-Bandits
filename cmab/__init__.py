"""
Contextual Multi-Arm Bandit (CMAB) Package

This package provides implementations of contextual multi-arm bandits using
different underlying models (XGBoost and Neural Networks).
"""

from .ground_truth import GroundTruthRewardFunction
from .student_generator import generate_student_class, CLASS_SIZE, TARGET_COUNTS
from .training import train_bandit
from .xgboost_bandit import ContextualBanditXGB
from .neural_net_bandit import ContextualBanditNN, RewardNetwork

__all__ = [
    "GroundTruthRewardFunction",
    "generate_student_class",
    "CLASS_SIZE",
    "TARGET_COUNTS",
    "train_bandit",
    "ContextualBanditXGB",
    "ContextualBanditNN",
    "RewardNetwork",
]
