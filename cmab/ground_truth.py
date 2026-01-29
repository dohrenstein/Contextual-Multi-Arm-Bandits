"""
Ground Truth Reward Function for the Student Learning CMAB Example.

This module contains the hidden ground truth function that generates student
test scores based on their features and the lesson type they receive.
"""

import numpy as np


class GroundTruthRewardFunction:
    """
    Hidden ground truth function that generates student test scores.

    Student context has 3 features:
    - Feature 0: Prior knowledge (0-1)
    - Feature 1: Study hours per week (0-1, normalized)
    - Feature 2: Mathematical aptitude (0-1)

    Lesson types (arms):
    - 0: Visual lessons (diagrams, videos)
    - 1: Auditory lessons (lectures, discussions)
    - 2: Hands-on lessons (projects, experiments)
    """

    def __init__(self, noise_std: float = 5.0):
        """
        Initialize the ground truth reward function.

        Args:
            noise_std: Standard deviation of noise added to test scores
        """
        # True coefficients for each lesson type (unknown to the bandit)
        # Each row represents how features affect test scores for that lesson type
        # More exaggerated weights to create clearer differentiation
        self.true_weights = {
            0: np.array(
                [0.05, 0.05, 1.0]
            ),  # Visual: VERY strongly benefits high aptitude students
            1: np.array(
                [1.0, 0.05, 0.05]
            ),  # Auditory: VERY strongly benefits those with prior knowledge
            2: np.array(
                [0.05, 1.0, 0.05]
            ),  # Hands-on: VERY strongly benefits those who study more
        }

        # Base scores for each lesson type (equal baseline to ensure fairness)
        self.base_scores = {
            0: 60,  # Visual baseline
            1: 60,  # Auditory baseline
            2: 60,  # Hands-on baseline
        }

        # Noise level in test scores
        self.noise_std = noise_std

    def get_expected_reward(self, context: np.ndarray, arm: int) -> float:
        """
        Get the expected test score without noise (ground truth).

        This version intentionally exaggerates non-linearities so the *optimal policy*
        is easy to see in the decision boundary plots.

        Args:
            context: numpy array of shape (3,) with student features
            arm: integer 0, 1, or 2 representing lesson type

        Returns:
            expected_score: float representing the true expected score (no noise)
        """
        prior, study, aptitude = context

        # Linear component (kept from the original ground truth)
        linear_score = np.dot(self.true_weights[arm], context) * 30

        # Strong non-linear component (arm-specific)
        # Using smooth functions (tanh/sin) to create obvious curved boundaries.
        if arm == 0:  # Visual
            # Visual has a strong "aptitude threshold" effect + interaction wiggles
            nonlinear = (
                28.0 * np.tanh(6.0 * (aptitude - 0.55))
                + 18.0 * np.sin(2.0 * np.pi * study * aptitude)
                - 14.0 * (aptitude - 0.85) ** 2
            )
        elif arm == 1:  # Auditory
            # Auditory has a strong "prior threshold" effect + prior×aptitude interaction
            nonlinear = (
                28.0 * np.tanh(6.0 * (prior - 0.55))
                + 16.0 * np.sin(2.0 * np.pi * prior * aptitude)
                - 14.0 * (prior - 0.80) ** 2
            )
        else:  # Hands-on (arm == 2)
            # Hands-on has a strong "study threshold" effect + study×aptitude interaction
            nonlinear = (
                30.0 * np.tanh(6.0 * (study - 0.50))
                + 20.0 * np.sin(2.0 * np.pi * study)
                + 18.0 * (study * aptitude)
                - 10.0 * (1.0 - study) ** 2
            )

        return self.base_scores[arm] + linear_score + nonlinear

    def get_reward(self, context: np.ndarray, arm: int) -> float:
        """
        Generate a test score for a student given their context and lesson type.

        Args:
            context: numpy array of shape (3,) with student features
            arm: integer 0, 1, or 2 representing lesson type

        Returns:
            test_score: float
        """
        # Calculate expected score based on true relationship
        expected_score = self.get_expected_reward(context, arm)

        # Add random noise to simulate real-world variability
        actual_score = expected_score + np.random.normal(0, self.noise_std)

        return actual_score

    def get_optimal_arm(self, context: np.ndarray) -> int:
        """
        Returns the best lesson type for a given student (oracle knowledge).

        Args:
            context: numpy array of shape (3,) with student features

        Returns:
            optimal_arm: integer representing best lesson type
        """
        expected_rewards = [self.get_expected_reward(context, arm) for arm in range(3)]
        return int(np.argmax(expected_rewards))
