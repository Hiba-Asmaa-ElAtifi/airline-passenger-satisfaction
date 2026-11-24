
"""
K-Nearest Neighbors (KNN) implemented from scratch.

Project: Airline Passenger Satisfaction

Based on LAB2:
    1. Calculate Euclidean distances.
    2. Find the k nearest training observations.
    3. Retrieve their satisfaction labels.
    4. Count votes for each class.
    5. Predict the majority class.

NumPy batch processing is used for efficiency.
No scikit-learn KNN classifier is used.
"""

import numpy as np


class KNNScratch:
    """Binary K-Nearest Neighbors classifier."""

    def __init__(self, k=5, batch_size=32):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors.
        batch_size : int
            Number of query passengers processed per batch.
        """

        if (
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or k < 1
        ):
            raise ValueError("k must be a positive integer.")

        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError(
                "batch_size must be a positive integer."
            )

        self.k = int(k)
        self.batch_size = int(batch_size)

        # Attributes initialized before training.
        self.X_train_ = None
        self.y_train_ = None
        self.train_squared_norms_ = None
        self.n_features_ = None

    # ========================================================
    # 1. FIT - STORE TRAINING DATA
    # ========================================================

    def fit(self, X, y):
        """
        Store preprocessed training passengers and labels.

        Unlike Logistic Regression or LDA, KNN does not
        learn coefficients during training.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        # Validate training features.
        if (
            X.ndim != 2
            or X.shape[0] == 0
            or X.shape[1] == 0
        ):
            raise ValueError(
                "X must be a non-empty 2D array."
            )

        if len(y) != len(X):
            raise ValueError(
                "X and y must have matching rows."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError(
                "Labels must be binary: 0 or 1."
            )

        if self.k > len(X):
            raise ValueError(
                "k cannot exceed the number of "
                "training samples."
            )

        # Store training observations.
        self.X_train_ = np.ascontiguousarray(
            X.copy()
        )

        self.y_train_ = y.astype(
            np.int64,
            copy=True
        )

        self.n_features_ = X.shape[1]

        # Precompute squared norms to speed up distances.
        self.train_squared_norms_ = np.sum(
            self.X_train_ ** 2,
            axis=1
        )

        return self

    # ========================================================
    # 2. VALIDATE PREDICTION DATA
    # ========================================================

    def _validate_query(self, X):
        """
        Check that prediction data matches the
        training feature dimensions.
        """

        if self.X_train_ is None:
            raise RuntimeError(
                "Fit the model before prediction."
            )

        X = np.asarray(X, dtype=np.float64)

        if (
            X.ndim != 2
            or X.shape[1] != self.n_features_
        ):
            raise ValueError(
                "Invalid input feature dimensions."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    # ========================================================
    # 3. FIND NEAREST NEIGHBORS
    # ========================================================

    def _find_neighbors(self, X_batch):
        """
        Return the k nearest neighbors, ordered by distance.

        Squared Euclidean distance:
            ||a-b||² = ||a||² + ||b||² - 2(a.b)

        Square roots are unnecessary because they
        do not change the distance ordering.
        """

        # Squared norms of query passengers.
        query_squared_norms = np.sum(
            X_batch ** 2,
            axis=1
        )[:, None]

        # Compute distances to training passengers.
        distances_squared = (
            query_squared_norms
            + self.train_squared_norms_[None, :]
            - 2.0 * (
                X_batch @ self.X_train_.T
            )
        )

        # Avoid tiny negative values from rounding errors.
        np.maximum(
            distances_squared,
            0.0,
            out=distances_squared
        )

        # Select the k nearest candidates.
        nearest_indices = np.argpartition(
            distances_squared,
            kth=self.k - 1,
            axis=1
        )[:, :self.k]

        # Retrieve distances for the selected candidates.
        nearest_distances = np.take_along_axis(
            distances_squared,
            nearest_indices,
            axis=1
        )

        # Sort selected neighbors by increasing distance.
        sorted_positions = np.argsort(
            nearest_distances,
            axis=1,
            kind="stable"
        )

        nearest_indices = np.take_along_axis(
            nearest_indices,
            sorted_positions,
            axis=1
        )

        return nearest_indices

    # ========================================================
    # 4. PREDICT PASSENGER SATISFACTION
    # ========================================================

    def predict(self, X):
        """
        Predict passenger satisfaction.

        Class 0: neutral or dissatisfied
        Class 1: satisfied

        Uses majority voting among the k nearest neighbors.
        """

        X = self._validate_query(X)

        predictions = np.empty(
            len(X),
            dtype=np.int64
        )

        # Process passengers in batches.
        for start in range(
            0,
            len(X),
            self.batch_size
        ):

            end = min(
                start + self.batch_size,
                len(X)
            )

            X_batch = X[start:end]

            # Find nearest training passengers.
            nearest_indices = self._find_neighbors(
                X_batch
            )

            # Retrieve their satisfaction labels.
            neighbor_labels = self.y_train_[
                nearest_indices
            ]

            # Count votes for class 1.
            votes_for_one = np.sum(
                neighbor_labels == 1,
                axis=1
            )

            # Count votes for class 0.
            votes_for_zero = (
                self.k - votes_for_one
            )

            # Majority voting.
            # In a tie, choose class 0.
            predictions[start:end] = (
                votes_for_one > votes_for_zero
            ).astype(np.int64)

        return predictions
