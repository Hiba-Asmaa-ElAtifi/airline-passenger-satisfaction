
"""
Logistic Regression implemented from scratch.

Binary classification using sigmoid, binary cross-entropy,
and batch gradient descent.
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Binary Logistic Regression trained with gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size used to update the model parameters.
    epochs : int
        Number of training iterations.
    """

    def __init__(self, learning_rate=0.01, epochs=100):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _initialize_parameters(self, n_features):
        """Initialize weights and bias to zero."""
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

    def _sigmoid(self, z):
        """Convert linear scores into probabilities."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _hypothesis(self, X):
        """Compute predicted probabilities."""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def _compute_cost(self, y, y_hat):
        """Compute mean binary cross-entropy loss."""
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        loss = -(
            y * np.log(y_hat)
            + (1 - y) * np.log(1 - y_hat)
        )

        return float(np.mean(loss))

    def _compute_gradients(self, X, y, y_hat):
        """Compute gradients of weights and bias."""
        n_samples = X.shape[0]
        error = y_hat - y

        dw = np.dot(X.T, error) / n_samples
        db = np.mean(error)

        return dw, db

    def fit(self, X, y):
        """
        Train the model using batch gradient descent.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Preprocessed training features.
        y : array-like, shape (n_samples,)
            Binary labels (0 or 1).

        Returns
        -------
        self
            Trained model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if len(y) != X.shape[0]:
            raise ValueError("X and y must have matching rows.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1.")

        self._initialize_parameters(X.shape[1])
        self.loss_history = []

        for epoch in range(self.epochs):
            y_hat = self._hypothesis(X)

            dw, db = self._compute_gradients(X, y, y_hat)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            updated_predictions = self._hypothesis(X)
            loss = self._compute_cost(y, updated_predictions)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """Return probability of class 1."""
        if self.weights is None:
            raise RuntimeError("The model must be fitted first.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Invalid feature dimensions.")

        return self._hypothesis(X)

    def predict(self, X):
        """Predict binary class labels using threshold 0.5."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
