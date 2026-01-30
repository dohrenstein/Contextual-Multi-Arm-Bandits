"""
Student Class Generator for the CMAB Example.

This module provides functions to generate fresh classes of students
for each training epoch.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .ground_truth import GroundTruthRewardFunction


# Default class configuration
CLASS_SIZE = 25  # Students per class
TARGET_PROPORTIONS = {
    0: 0.6,
    1: 0.3,
    2: 0.1,
}  # Visual / Auditory / Hands-on optimal proportions


def generate_student_class(
    ground_truth: GroundTruthRewardFunction,
    class_size: int = CLASS_SIZE,
    target_proportions: Optional[Dict[int, float]] = None,
    pool_multiplier: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a fresh class of students using vectorized operations.

    Returns numpy arrays instead of list of dicts for efficiency.

    Args:
        ground_truth: GroundTruthRewardFunction instance
        class_size: Number of students in the class
        target_proportions: Desired optimal-arm distribution {arm: proportion}
        pool_multiplier: Multiplier for candidate pool size

    Returns:
        contexts: numpy array of shape (class_size, 5)
        optimal_arms: numpy array of shape (class_size,) with dtype int
    """
    if target_proportions is None:
        target_proportions = TARGET_PROPORTIONS.copy()

    # Fast path: no stratification needed
    if not target_proportions:
        contexts = np.random.random((class_size, 5))
        optimal_arms = ground_truth.get_optimal_arms_batch(contexts)
        return contexts, optimal_arms

    # Convert proportions to integer counts that sum to class_size
    prop_sum = sum(float(v) for v in target_proportions.values())
    weights = {arm: float(v) / prop_sum for arm, v in target_proportions.items()}
    target_counts = {
        arm: int(round(weights.get(arm, 0.0) * class_size)) for arm in range(3)
    }
    # Fix rounding drift
    drift = class_size - sum(target_counts.values())
    if drift != 0:
        best_arm = max(target_counts.items(), key=lambda x: x[1])[0]
        target_counts[best_arm] += drift

    pool_size = max(class_size * pool_multiplier, class_size)

    # Build a candidate pool from the base distribution (vectorized)
    contexts = np.random.random((pool_size, 5))
    optimal_arms = ground_truth.get_optimal_arms_batch(contexts)

    selected_contexts_list = []
    selected_optimal_arms_list = []

    for arm in range(3):
        need = int(target_counts.get(arm, 0))
        if need <= 0:
            continue

        idxs = np.where(optimal_arms == arm)[0]

        # If the pool doesn't contain enough, generate more samples
        if len(idxs) < need:
            extra_pool_size = max((need - len(idxs)) * 50, class_size * 10)
            extra_contexts = np.random.random((extra_pool_size, 5))
            extra_optimal_arms = ground_truth.get_optimal_arms_batch(extra_contexts)

            contexts = np.vstack([contexts, extra_contexts])
            optimal_arms = np.concatenate([optimal_arms, extra_optimal_arms])
            idxs = np.where(optimal_arms == arm)[0]

        if len(idxs) == 0:
            raise RuntimeError(
                f"Could not sample any students with optimal arm={arm}. "
                "Try increasing pool_multiplier or adjusting the ground truth."
            )

        replace = len(idxs) < need
        chosen = np.random.choice(idxs, size=need, replace=replace)
        selected_contexts_list.append(contexts[chosen])
        selected_optimal_arms_list.append(np.full(need, arm, dtype=int))

    contexts_final = np.vstack(selected_contexts_list)
    optimal_arms_final = np.concatenate(selected_optimal_arms_list)

    # Shuffle so the class isn't ordered by arm
    perm = np.random.permutation(len(contexts_final))
    contexts_final = contexts_final[perm]
    optimal_arms_final = optimal_arms_final[perm]

    return contexts_final, optimal_arms_final
