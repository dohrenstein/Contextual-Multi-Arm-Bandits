"""
Student Class Generator for the CMAB Example.

This module provides functions to generate fresh classes of students
for each training epoch.
"""

import numpy as np
from typing import Dict, List, Optional

from .ground_truth import GroundTruthRewardFunction


# Default class configuration
CLASS_SIZE = 100
TARGET_COUNTS = {0: 60, 1: 30, 2: 10}  # Visual / Auditory / Hands-on optimal counts


def generate_student_class(
    ground_truth: GroundTruthRewardFunction,
    class_size: int = CLASS_SIZE,
    target_counts: Optional[Dict[int, int]] = None,
    pool_multiplier: int = 30,
) -> List[Dict]:
    """
    Generate a fresh class of students.

    By default, contexts are sampled from a smooth base distribution
    (Uniform[0,1]^3). If `target_counts` is provided, we then stratify by the
    *true* optimal arm (oracle) and sample a class with that mix.

    Args:
        ground_truth: GroundTruthRewardFunction instance
        class_size: Number of students in the class
        target_counts: Desired optimal-arm distribution {arm: count}
        pool_multiplier: Multiplier for candidate pool size

    Returns:
        List of dicts: {id, context, optimal_arm}
    """
    if target_counts is None:
        target_counts = TARGET_COUNTS.copy()

    if not target_counts:
        contexts = np.random.random((class_size, 3))
        student_class = []
        for student_id, context in enumerate(contexts):
            optimal_arm = int(ground_truth.get_optimal_arm(context))
            student_class.append(
                {"id": student_id, "context": context, "optimal_arm": optimal_arm}
            )
        return student_class

    # If target_counts doesn't sum to class_size, treat it as proportions.
    target_sum = sum(int(v) for v in target_counts.values())
    if target_sum != class_size:
        # Convert to proportions then to integer counts that sum to class_size
        weights = {
            arm: float(v) / float(target_sum) for arm, v in target_counts.items()
        }
        counts = {
            arm: int(round(weights.get(arm, 0.0) * class_size)) for arm in range(3)
        }
        # Fix rounding drift
        drift = class_size - sum(counts.values())
        if drift != 0:
            # Add/subtract to the most common arm
            best_arm = max(counts.items(), key=lambda x: x[1])[0]
            counts[best_arm] += drift
        target_counts = counts

    pool_size = max(class_size * pool_multiplier, class_size)

    # Build a candidate pool from the base distribution
    contexts = np.random.random((pool_size, 3))
    optimal_arms = np.array(
        [int(ground_truth.get_optimal_arm(c)) for c in contexts], dtype=int
    )

    selected_contexts = []
    selected_optimal_arms = []

    for arm in range(3):
        need = int(target_counts.get(arm, 0))
        if need <= 0:
            continue

        idxs = np.where(optimal_arms == arm)[0]

        # If the pool doesn't contain enough of a given arm, resample a bigger pool.
        # (This keeps the feature distribution smooth without hand-crafting.)
        if len(idxs) < need:
            # Increase pool and try once more
            extra_pool_size = max((need - len(idxs)) * 50, class_size * 10)
            extra_contexts = np.random.random((extra_pool_size, 3))
            extra_optimal_arms = np.array(
                [int(ground_truth.get_optimal_arm(c)) for c in extra_contexts],
                dtype=int,
            )

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
        selected_contexts.append(contexts[chosen])
        selected_optimal_arms.append(np.full(need, arm, dtype=int))

    contexts_final = np.vstack(selected_contexts)
    optimal_arms_final = np.concatenate(selected_optimal_arms)

    # Shuffle so the class isn't ordered by arm
    perm = np.random.permutation(len(contexts_final))
    contexts_final = contexts_final[perm]
    optimal_arms_final = optimal_arms_final[perm]

    student_class = []
    for student_id, (context, optimal_arm) in enumerate(
        zip(contexts_final, optimal_arms_final)
    ):
        student_class.append(
            {"id": student_id, "context": context, "optimal_arm": int(optimal_arm)}
        )

    return student_class
