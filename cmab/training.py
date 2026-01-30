"""
Training Functions for Contextual Multi-Arm Bandits.

This module provides functions to train and evaluate CMAB models.
"""

import time
import numpy as np
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

    Optimized version using numpy arrays and vectorized operations.

    Args:
        bandit: Contextual bandit instance (must have select_arm, update, train_epoch methods)
        ground_truth: GroundTruthRewardFunction instance
        class_size: Students per epoch
        target_proportions: Desired optimal-arm distribution in each sampled class
        n_epochs: Number of epochs
        verbose: Whether to print progress

    Returns:
        history: Dictionary with training metrics (numpy arrays)
    """
    if target_proportions is None:
        target_proportions = TARGET_PROPORTIONS.copy()

    total_rounds = n_epochs * class_size

    # Pre-allocate arrays for history
    history = {
        "round": np.arange(total_rounds, dtype=np.int32),
        "epoch": np.zeros(total_rounds, dtype=np.int32),
        "student_id": np.zeros(total_rounds, dtype=np.int32),
        "reward": np.zeros(total_rounds, dtype=np.float64),
        "optimal_arm": np.zeros(total_rounds, dtype=np.int32),
        "selected_arm": np.zeros(total_rounds, dtype=np.int32),
        "cumulative_reward": np.zeros(total_rounds, dtype=np.float64),
        "cumulative_optimal_reward": np.zeros(total_rounds, dtype=np.float64),
        "regret": np.zeros(total_rounds, dtype=np.float64),
        "epoch_accuracy": np.zeros(total_rounds, dtype=np.float64),
    }

    cumulative_reward = 0.0
    cumulative_optimal_reward = 0.0
    round_num = 0
    training_start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        # Generate student class (returns numpy arrays)
        contexts, optimal_arms = generate_student_class(
            ground_truth,
            class_size=class_size,
            target_proportions=target_proportions,
        )

        epoch_correct = 0
        epoch_start = round_num

        # Process each student
        for i in range(class_size):
            context = contexts[i]
            optimal_arm = optimal_arms[i]

            selected_arm = bandit.select_arm(context)
            reward = float(ground_truth.get_reward(context, selected_arm))
            optimal_reward = float(ground_truth.get_reward(context, optimal_arm))

            # Record observation for training
            bandit.update(context, selected_arm, reward)

            if selected_arm == optimal_arm:
                epoch_correct += 1

            cumulative_reward += reward
            cumulative_optimal_reward += optimal_reward

            # Store in pre-allocated arrays
            history["epoch"][round_num] = epoch
            history["student_id"][round_num] = i
            history["reward"][round_num] = reward
            history["optimal_arm"][round_num] = optimal_arm
            history["selected_arm"][round_num] = selected_arm
            history["cumulative_reward"][round_num] = cumulative_reward
            history["cumulative_optimal_reward"][round_num] = cumulative_optimal_reward
            history["regret"][round_num] = cumulative_optimal_reward - cumulative_reward

            round_num += 1

        # Train once per epoch after collecting all observations
        bandit.train_epoch()

        epoch_accuracy = epoch_correct / class_size * 100

        # Fill epoch_accuracy for all rounds in this epoch
        history["epoch_accuracy"][epoch_start:round_num] = epoch_accuracy

        if verbose:
            epoch_time = time.time() - epoch_start_time
            cumulative_time = time.time() - training_start_time
            avg_reward = cumulative_reward / round_num
            regret = history["regret"][round_num - 1]
            print(
                f"Epoch {epoch+1:2d}/{n_epochs}: Accuracy = {epoch_accuracy:5.1f}%, "
                f"Avg Reward = {avg_reward:.2f}, Regret = {regret:.2f}, "
                f"Time = {epoch_time:.2f}s (Total: {cumulative_time:.1f}s)"
            )

    return history
