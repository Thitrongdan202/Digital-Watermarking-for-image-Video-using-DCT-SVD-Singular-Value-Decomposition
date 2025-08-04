import numpy as np
from sklearn.linear_model import LogisticRegression


class WatermarkDetector:
    """Simple AI-based detector using logistic regression on singular values."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=200)
        self.trained = False

    def _ensure_trained(self, feature_length: int) -> None:
        if not self.trained:
            # generate synthetic training data
            X0 = np.random.normal(0, 1, (50, feature_length))
            X1 = np.random.normal(1, 1, (50, feature_length))
            X = np.vstack([X0, X1])
            y = np.array([0] * 50 + [1] * 50)
            self.model.fit(X, y)
            self.trained = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict whether watermark is present based on feature matrix."""
        self._ensure_trained(features.shape[1])
        return self.model.predict(features)
