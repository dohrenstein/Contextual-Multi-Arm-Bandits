"""
Ground Truth Reward Function for the Student Learning CMAB Example.

This module contains the hidden ground truth function that generates student
test scores based on their features and the lesson type they receive.
"""

import numpy as np


class GroundTruthRewardFunction:
    """
    Hidden ground truth function that generates student test scores.

    Student context has 5 features:
    - Feature 0: Prior knowledge (0-1)
    - Feature 1: Study hours per week (0-1, normalized)
    - Feature 2: Mathematical aptitude (0-1)
    - Feature 3: Attention span (0-1)
    - Feature 4: Creativity (0-1)

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
        # Features: [prior, study, aptitude, attention, creativity]
        self.true_weights = {
            0: np.array(
                [0.05, 0.05, 0.8, 0.6, 0.1]
            ),  # Visual: benefits high aptitude + attention span
            1: np.array(
                [0.8, 0.1, 0.05, 0.5, 0.05]
            ),  # Auditory: benefits prior knowledge + attention
            2: np.array(
                [0.05, 0.7, 0.1, 0.1, 0.8]
            ),  # Hands-on: benefits study hours + creativity
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
            context: numpy array of shape (5,) with student features
            arm: integer 0, 1, or 2 representing lesson type

        Returns:
            expected_score: float representing the true expected score (no noise)
        """
        prior, study, aptitude, attention, creativity = context

        # Linear component (kept from the original ground truth)
        linear_score = np.dot(self.true_weights[arm], context) * 25

        # Strong non-linear component (arm-specific)
        # Using smooth functions (tanh/sin) to create obvious curved boundaries.
        if arm == 0:  # Visual
            # Visual benefits high aptitude + attention, with interaction effects
            nonlinear = (
                25.0 * np.tanh(5.0 * (aptitude - 0.55))
                + 15.0 * np.tanh(4.0 * (attention - 0.5))
                + 12.0 * np.sin(2.0 * np.pi * aptitude * attention)
                - 10.0 * (aptitude - 0.85) ** 2
            )
        elif arm == 1:  # Auditory
            # Auditory benefits prior knowledge + attention span
            nonlinear = (
                25.0 * np.tanh(5.0 * (prior - 0.55))
                + 15.0 * np.tanh(4.0 * (attention - 0.5))
                + 10.0 * np.sin(2.0 * np.pi * prior * attention)
                - 10.0 * (prior - 0.80) ** 2
            )
        else:  # Hands-on (arm == 2)
            # Hands-on benefits study hours + creativity
            nonlinear = (
                25.0 * np.tanh(5.0 * (study - 0.50))
                + 20.0 * np.tanh(4.0 * (creativity - 0.5))
                + 15.0 * (study * creativity)
                + 10.0 * np.sin(2.0 * np.pi * creativity)
                - 8.0 * (1.0 - study) ** 2
            )

        return self.base_scores[arm] + linear_score + nonlinear

    def get_reward(self, context: np.ndarray, arm: int) -> float:
        """
        Generate a test score for a student given their context and lesson type.

        Args:
            context: numpy array of shape (5,) with student features
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
            context: numpy array of shape (5,) with student features

        Returns:
            optimal_arm: integer representing best lesson type
        """
        expected_rewards = [self.get_expected_reward(context, arm) for arm in range(3)]
        return int(np.argmax(expected_rewards))

    def get_expected_rewards_batch(self, contexts: np.ndarray) -> np.ndarray:
        """
        Vectorized: compute expected rewards for all contexts and all arms.

        Args:
            contexts: numpy array of shape (n, 5)

        Returns:
            expected_rewards: numpy array of shape (n, 3)
        """
        n = contexts.shape[0]
        prior = contexts[:, 0]
        study = contexts[:, 1]
        aptitude = contexts[:, 2]
        attention = contexts[:, 3]
        creativity = contexts[:, 4]

        rewards = np.zeros((n, 3))

        for arm in range(3):
            # Linear component
            linear_score = contexts @ self.true_weights[arm] * 25

            # Non-linear component
            if arm == 0:  # Visual
                nonlinear = (
                    25.0 * np.tanh(5.0 * (aptitude - 0.55))
                    + 15.0 * np.tanh(4.0 * (attention - 0.5))
                    + 12.0 * np.sin(2.0 * np.pi * aptitude * attention)
                    - 10.0 * (aptitude - 0.85) ** 2
                )
            elif arm == 1:  # Auditory
                nonlinear = (
                    25.0 * np.tanh(5.0 * (prior - 0.55))
                    + 15.0 * np.tanh(4.0 * (attention - 0.5))
                    + 10.0 * np.sin(2.0 * np.pi * prior * attention)
                    - 10.0 * (prior - 0.80) ** 2
                )
            else:  # Hands-on (arm == 2)
                nonlinear = (
                    25.0 * np.tanh(5.0 * (study - 0.50))
                    + 20.0 * np.tanh(4.0 * (creativity - 0.5))
                    + 15.0 * (study * creativity)
                    + 10.0 * np.sin(2.0 * np.pi * creativity)
                    - 8.0 * (1.0 - study) ** 2
                )

            rewards[:, arm] = self.base_scores[arm] + linear_score + nonlinear

        return rewards

    def get_optimal_arms_batch(self, contexts: np.ndarray) -> np.ndarray:
        """
        Vectorized: compute optimal arm for each context.

        Args:
            contexts: numpy array of shape (n, 5)

        Returns:
            optimal_arms: numpy array of shape (n,) with dtype int
        """
        rewards = self.get_expected_rewards_batch(contexts)
        return np.argmax(rewards, axis=1).astype(int)

    def get_rewards_batch(self, contexts: np.ndarray, arms: np.ndarray) -> np.ndarray:
        """
        Vectorized: compute noisy rewards for given contexts and selected arms.

        Args:
            contexts: numpy array of shape (n, 5)
            arms: numpy array of shape (n,) with arm indices

        Returns:
            rewards: numpy array of shape (n,)
        """
        expected = self.get_expected_rewards_batch(contexts)
        # Select the reward for each context's chosen arm
        n = contexts.shape[0]
        selected_rewards = expected[np.arange(n), arms]
        noise = np.random.normal(0, self.noise_std, size=n)
        return selected_rewards + noise
