
"""
Linear Discriminant Analysis (LDA) implemented from scratch.

Project: Airline Passenger Satisfaction

Based on the statistical principles studied in TP5:
    1. Compute the mean of each class.
    2. Estimate a shared within-class covariance matrix.
    3. Estimate class prior probabilities.
    4. Compute a discriminant score for each class.
    5. Predict the class with the highest score.

This implementation supports multiple numerical features.
"""

import numpy as np


class LDAScratch:
    """Binary Linear Discriminant Analysis classifier."""

    def __init__(self, regularization=1e-4):
        """
        Parameters
        ----------
        regularization : float
            Non-negative value added to the covariance diagonal
            to improve numerical stability.
        """
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError(
                "regularization must be a finite, non-negative value."
            )

        self.regularization = regularization

        # Parameters learned during training
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_ = None
        self.weights_ = None
        self.intercepts_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """
        Estimate LDA parameters using training data only.

        X : array-like of shape (n_samples, n_features)
            Preprocessed passenger characteristics.
        y : array-like of shape (n_samples,)
            Binary satisfaction labels (0 or 1).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        # --------------------------------------------------
        # Step 1: Validate the input data
        # --------------------------------------------------
        if X.ndim != 2 or X.shape[0] < 3:
            raise ValueError(
                "X must be a 2D array with at least 3 samples."
            )

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if len(y) != len(X):
            raise ValueError("X and y must have matching rows.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1.")

        self.classes_, counts = np.unique(
            y.astype(int),
            return_counts=True
        )

        if len(self.classes_) != 2:
            raise ValueError(
                "LDA requires both binary classes in training."
            )

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # --------------------------------------------------
        # Step 2: Compute class prior probabilities
        # --------------------------------------------------
        # Example: proportion of satisfied passengers.
        self.priors_ = counts / n_samples

        # --------------------------------------------------
        # Step 3: Compute the mean vector of each class
        # --------------------------------------------------
        self.means_ = np.array([
            X[y == label].mean(axis=0)
            for label in self.classes_
        ])

        # --------------------------------------------------
        # Step 4: Estimate the pooled covariance matrix
        # --------------------------------------------------
        # Each observation is centered using its own
        # class mean, following the idea of TP5.
        pooled_scatter = np.zeros(
            (n_features, n_features),
            dtype=float
        )

        for index, label in enumerate(self.classes_):
            X_class = X[y == label]
            centered = X_class - self.means_[index]

            pooled_scatter += centered.T @ centered

        # Two classes => denominator n_samples - 2.
        covariance = pooled_scatter / (n_samples - 2)

        # --------------------------------------------------
        # Step 5: Apply covariance regularization
        # --------------------------------------------------
        # One-hot encoded features may create linear
        # dependencies. Regularization helps stabilize
        # the discriminant calculation.
        covariance += (
            self.regularization * np.eye(n_features)
        )

        self.covariance_ = covariance

        # --------------------------------------------------
        # Step 6: Calculate discriminant coefficients
        # --------------------------------------------------
        # Solve covariance * W = means.T instead of
        # explicitly calculating the matrix inverse.
        try:
            self.weights_ = np.linalg.solve(
                covariance,
                self.means_.T
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Covariance matrix is singular. "
                "Try a positive regularization value."
            ) from exc

        # One intercept for each satisfaction class.
        self.intercepts_ = (
            -0.5 * np.sum(
                self.means_ * self.weights_.T,
                axis=1
            )
            + np.log(self.priors_)
        )

        return self

    def decision_function(self, X):
        """
        Calculate the LDA discriminant scores.

        Returns an array of shape (n_samples, 2).
        Column 0: score for neutral/dissatisfied.
        Column 1: score for satisfied.
        """
        if self.weights_ is None:
            raise RuntimeError(
                "The model must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Invalid feature dimensions.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        return X @ self.weights_ + self.intercepts_

    def predict(self, X):
        """
        Predict passenger satisfaction.

        Returns
        -------
        numpy.ndarray
            Binary predictions: 0 or 1.
        """
        scores = self.decision_function(X)

        best_class_indices = np.argmax(scores, axis=1)

        return self.classes_[best_class_indices]
