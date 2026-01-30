"""
XGBoost-based Contextual Multi-Arm Bandit.

This module implements a CMAB using XGBoost for reward prediction.
The model is retrained on all historical data after each observation.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


class ContextualBanditXGB:
    """
    Contextual Multi-Arm Bandit using XGBoost for reward prediction.
    Uses epsilon-greedy strategy for exploration vs exploitation.

    We model reward as a function of:
      - student context features (prior knowledge, study hours, aptitude, attention, creativity)
      - lesson type / arm (categorical)

    Using a single categorical arm feature avoids one-hot expansion and lets
    XGBoost learn arm-specific effects and interactions via categorical splits.

    NOTE: This implementation uses BATCH LEARNING - it retrains on all
    historical data after each new observation.
    """

    def __init__(
        self,
        n_arms: int = 3,
        n_features: int = 3,
        epsilon: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ):
        """
        Initialize the bandit.

        Args:
            n_arms: Number of lesson types (arms)
            n_features: Number of student features (context dimensions)
            epsilon: Exploration rate (0 = always exploit, 1 = always explore)
            n_estimators: Number of boosting rounds for XGBoost
            max_depth: Maximum tree depth for XGBoost
            learning_rate: Learning rate for XGBoost
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.epsilon = epsilon

        # XGBoost regressor model
        # Input: [context (5) + arm (1 categorical)] = 6 features
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0,
            tree_method="hist",
            enable_categorical=True,
        )

        # Store all training data
        # Each row is a dict with keys: prior_knowledge, study_hours, aptitude, arm
        self.X_train = []
        self.y_train = []

        # Track if model has been trained
        self.is_trained = False

    def _make_row(self, context: np.ndarray, arm: int) -> dict:
        """Create a single training row as a python dict."""
        prior, study, aptitude, attention, creativity = context
        return {
            "prior_knowledge": float(prior),
            "study_hours": float(study),
            "aptitude": float(aptitude),
            "attention": float(attention),
            "creativity": float(creativity),
            "arm": int(arm),
        }

    def _make_frame(self, rows: list) -> pd.DataFrame:
        """Convert one or many rows to a pandas DataFrame with categorical arm."""
        X = pd.DataFrame(rows)
        X["arm"] = pd.Categorical(X["arm"], categories=list(range(self.n_arms)))
        return X

    def select_arm(self, context: np.ndarray) -> int:
        """Select which lesson type to use (epsilon-greedy strategy)."""
        # Exploration: choose random arm with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        # Exploitation: choose arm with highest predicted reward
        # If model not trained yet, explore randomly
        if not self.is_trained:
            return np.random.randint(self.n_arms)

        predicted_rewards = []
        for arm in range(self.n_arms):
            X_row = self._make_frame([self._make_row(context, arm)])
            pred = float(self.model.predict(X_row)[0])
            predicted_rewards.append(pred)

        return int(np.argmax(predicted_rewards))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """
        Record a new (context, arm, reward) observation.

        This only stores the data. Call train_epoch() to actually train the model.
        """
        self.X_train.append(self._make_row(context, arm))
        self.y_train.append(float(reward))

    def train_epoch(self) -> None:
        """
        Train the XGBoost model on all accumulated data.

        This should be called once per epoch after all observations are collected.
        """
        if len(self.X_train) >= 2:  # Need at least 2 samples
            X = self._make_frame(self.X_train)
            y = np.array(self.y_train)
            self.model.fit(X, y)
            self.is_trained = True

    def predict_best_arm(self, context: np.ndarray) -> int:
        """Predict the best arm for a given context (pure exploitation)."""
        if not self.is_trained:
            return 0  # Default to first arm if nothing trained

        predicted_rewards = []
        for arm in range(self.n_arms):
            X_row = self._make_frame([self._make_row(context, arm)])
            pred = float(self.model.predict(X_row)[0])
            predicted_rewards.append(pred)

        return int(np.argmax(predicted_rewards))

    def predict_best_arms_batch(self, contexts: np.ndarray) -> np.ndarray:
        """Predict the best arm for a batch of contexts (vectorized).
        
        Args:
            contexts: Array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) with best arm indices
        """
        if not self.is_trained:
            return np.zeros(len(contexts), dtype=int)
        
        n_samples = len(contexts)
        all_predictions = np.zeros((n_samples, self.n_arms))
        
        for arm in range(self.n_arms):
            # Create rows for all contexts with this arm
            rows = [self._make_row(ctx, arm) for ctx in contexts]
            X = self._make_frame(rows)
            all_predictions[:, arm] = self.model.predict(X)
        
        return np.argmax(all_predictions, axis=1)
