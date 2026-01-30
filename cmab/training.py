"""
Training Functions for Contextual Multi-Arm Bandits.

This module provides functions to train and evaluate CMAB models.
"""

from typing import Any, Dict, Optional

from .ground_truth import GroundTruthRewardFunction
from .student_generator import generate_student_class, CLASS_SIZE, TARGET_PROPORTIONS


def train_bandit(
    bandit: Any,
    ground_truth: GroundTruthRewardFunction,
    class_size: int = CLASS_SIZE,
    target_proportions: Optional[Dict[int, float]] = None,
    n_epochs: int = 20,
    verbose: bool = True,
) -> Dict:
    """
    Train a contextual bandit over multiple epochs.

    For each epoch, we generate a new class of students and run one interaction
    with each student (one pass = one epoch).

    Args:
        bandit: Contextual bandit instance (must have select_arm, update, predict_best_arm methods)
        ground_truth: GroundTruthRewardFunction instance
        class_size: Students per epoch
        target_proportions: Desired optimal-arm distribution in each sampled class
        n_epochs: Number of epochs
        verbose: Whether to print progress

    Returns:
        history: Dictionary with training metrics
    """
    if target_proportions is None:
        target_proportions = TARGET_PROPORTIONS.copy()

    history = {
        "round": [],
        "epoch": [],
        "student_id": [],
        "reward": [],
        "optimal_arm": [],
        "selected_arm": [],
        "cumulative_reward": [],
        "cumulative_optimal_reward": [],
        "regret": [],
        "epoch_accuracy": [],
    }

    cumulative_reward = 0.0
    cumulative_optimal_reward = 0.0
    round_num = 0

    for epoch in range(n_epochs):
        student_class = generate_student_class(
            ground_truth,
            class_size=class_size,
            target_proportions=target_proportions,
        )

        epoch_correct = 0
        epoch_total = len(student_class)

        for student in student_class:
            context = student["context"]
            student_id = student["id"]

            selected_arm = bandit.select_arm(context)
            reward = float(ground_truth.get_reward(context, selected_arm))

            optimal_arm = int(student["optimal_arm"])
            optimal_reward = float(ground_truth.get_reward(context, optimal_arm))

            # Only collect the observation (no training yet)
            bandit.update(context, selected_arm, reward)

            if selected_arm == optimal_arm:
                epoch_correct += 1

            cumulative_reward += reward
            cumulative_optimal_reward += optimal_reward
            regret = cumulative_optimal_reward - cumulative_reward

            history["round"].append(round_num)
            history["epoch"].append(epoch)
            history["student_id"].append(student_id)
            history["reward"].append(reward)
            history["optimal_arm"].append(optimal_arm)
            history["selected_arm"].append(selected_arm)
            history["cumulative_reward"].append(cumulative_reward)
            history["cumulative_optimal_reward"].append(cumulative_optimal_reward)
            history["regret"].append(regret)
            history["epoch_accuracy"].append(None)

            round_num += 1

        # Train once per epoch after collecting all observations
        bandit.train_epoch()

        epoch_accuracy = epoch_correct / epoch_total * 100

        epoch_start = epoch * epoch_total
        for i in range(epoch_start, round_num):
            history["epoch_accuracy"][i] = epoch_accuracy

        if verbose:
            avg_reward = cumulative_reward / round_num
            print(
                f"Epoch {epoch+1:2d}/{n_epochs}: Accuracy = {epoch_accuracy:5.1f}%, "
                f"Avg Reward = {avg_reward:.2f}, Regret = {regret:.2f}"
            )

    return history
