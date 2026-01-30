"""
Neural Network-based Contextual Multi-Arm Bandit with Online Learning.

This module implements a CMAB using a neural network for reward prediction.
The model is trained incrementally (online learning) with each new observation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardNetwork(nn.Module):
    """
    Neural network that predicts rewards given context and arm.

    Architecture: MLP with configurable hidden layers and ReLU activations.
    Input: [context (n_features) + arm one-hot (n_arms)]
    Output: predicted reward (1)
    """

    def __init__(
        self, n_features: int = 3, n_arms: int = 3, hidden_sizes: List[int] = None
    ):
        super(RewardNetwork, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        input_size = n_features + n_arms  # Context + one-hot encoded arm

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ContextualBanditNN:
    """
    Contextual Multi-Arm Bandit using a Neural Network for reward prediction.
    Uses epsilon-greedy strategy for exploration vs exploitation.

    KEY FEATURE: Online learning - the model is updated incrementally with each
    new observation using gradient descent, rather than retraining on all data.

    For stability, we use:
    1. Experience replay: store recent experiences and sample mini-batches
    2. Running normalization: track mean/std of inputs for normalization
    3. Gradient clipping: prevent exploding gradients
    """

    def __init__(
        self,
        n_arms: int = 3,
        n_features: int = 3,
        epsilon: float = 0.1,
        learning_rate: float = 0.01,
        hidden_sizes: List[int] = None,
        replay_buffer_size: int = 500,
        batch_size: int = 32,
        updates_per_step: int = 1,
    ):
        """
        Initialize the bandit.

        Args:
            n_arms: Number of lesson types (arms)
            n_features: Number of student features (context dimensions)
            epsilon: Exploration rate (0 = always exploit, 1 = always explore)
            learning_rate: Learning rate for neural network optimizer
            hidden_sizes: List of hidden layer sizes
            replay_buffer_size: Size of experience replay buffer
            batch_size: Mini-batch size for training
            updates_per_step: Number of gradient updates per new observation
        """
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.n_arms = n_arms
        self.n_features = n_features
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step

        # Neural network model
        self.model = RewardNetwork(n_features, n_arms, hidden_sizes).to(device)

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Experience replay buffer (circular buffer)
        # Store raw (context, arm, reward) tuples - normalize at training time
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer: List[Tuple[np.ndarray, int, float]] = []
        self.buffer_idx = 0

        # Running statistics for input normalization
        self.n_samples = 0
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)

        # Track training progress
        self.total_updates = 0
        self.is_trained = False

    def _normalize_context(self, context: np.ndarray) -> np.ndarray:
        """Normalize context using running statistics."""
        if self.n_samples < 2:
            return context
        return (context - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)

    def _update_running_stats(self, context: np.ndarray) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self.n_samples += 1
        if self.n_samples == 1:
            self.running_mean = context.copy()
            self.running_var = np.zeros(self.n_features)
        else:
            delta = context - self.running_mean
            self.running_mean += delta / self.n_samples
            delta2 = context - self.running_mean
            self.running_var += (delta * delta2 - self.running_var) / self.n_samples

    def _make_input(self, context: np.ndarray, arm: int) -> np.ndarray:
        """Create neural network input from context and arm."""
        # Normalize context
        normalized_context = self._normalize_context(context)

        # One-hot encode the arm
        arm_one_hot = np.zeros(self.n_arms)
        arm_one_hot[arm] = 1.0

        # Concatenate
        return np.concatenate([normalized_context, arm_one_hot])

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.FloatTensor(x).to(device)

    def select_arm(self, context: np.ndarray) -> int:
        """Select which lesson type to use (epsilon-greedy strategy)."""
        # Exploration: choose random arm with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        # Exploitation: choose arm with highest predicted reward
        if not self.is_trained:
            return np.random.randint(self.n_arms)

        self.model.eval()
        with torch.no_grad():
            predicted_rewards = []
            for arm in range(self.n_arms):
                x = self._make_input(context, arm)
                x_tensor = self._to_tensor(x).unsqueeze(0)
                pred = self.model(x_tensor).item()
                predicted_rewards.append(pred)

        return int(np.argmax(predicted_rewards))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """
        Record a new (context, arm, reward) observation.

        This only stores the data. Call train_epoch() to actually train the model.
        """
        # Update running statistics
        self._update_running_stats(context)

        # Store raw (context, arm, reward) - normalize at training time for consistency
        experience = (context.copy(), int(arm), float(reward))

        if len(self.replay_buffer) < self.replay_buffer_size:
            self.replay_buffer.append(experience)
        else:
            # Circular buffer: overwrite oldest experience
            self.replay_buffer[self.buffer_idx] = experience
            self.buffer_idx = (self.buffer_idx + 1) % self.replay_buffer_size

    def train_epoch(self, n_updates: int = None) -> None:
        """
        Train the neural network on accumulated experience.

        This should be called once per epoch after all observations are collected.
        Performs multiple gradient updates by sampling from the replay buffer.

        Args:
            n_updates: Number of gradient update steps to perform.
                       If None, defaults to enough updates to see each sample ~once on average.
        """
        if len(self.replay_buffer) < 2:
            return

        # Default: enough updates to cover the buffer approximately once
        if n_updates is None:
            n_updates = max(10, len(self.replay_buffer) // self.batch_size)

        self.model.train()

        for _ in range(n_updates):
            # Sample a mini-batch from replay buffer
            batch_size = min(self.batch_size, len(self.replay_buffer))
            indices = np.random.choice(
                len(self.replay_buffer), batch_size, replace=False
            )

            # Build batch with current normalization (consistent across all samples)
            batch_x = []
            batch_y = []
            for i in indices:
                context, arm, reward = self.replay_buffer[i]
                x = self._make_input(context, arm)
                batch_x.append(x)
                batch_y.append(reward)

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)

            # Convert to tensors
            x_tensor = self._to_tensor(batch_x)
            y_tensor = self._to_tensor(batch_y)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x_tensor)
            loss = self.criterion(predictions, y_tensor)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.total_updates += 1

        self.is_trained = True

    def predict_best_arm(self, context: np.ndarray) -> int:
        """Predict the best arm for a given context (pure exploitation)."""
        if not self.is_trained:
            return 0  # Default to first arm if nothing trained

        self.model.eval()
        with torch.no_grad():
            predicted_rewards = []
            for arm in range(self.n_arms):
                x = self._make_input(context, arm)
                x_tensor = self._to_tensor(x).unsqueeze(0)
                pred = self.model(x_tensor).item()
                predicted_rewards.append(pred)

        return int(np.argmax(predicted_rewards))
