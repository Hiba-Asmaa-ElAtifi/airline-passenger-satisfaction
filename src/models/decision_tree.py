
"""Decision Tree classifier implemented from scratch using NumPy.

Based on the Gini splitting algorithm studied in TP6.
No scikit-learn classifier is used.
"""

import numpy as np


class DecisionTreeScratch:
    """Binary classification tree using Gini impurity."""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.tree_ = None
        self.n_features_ = None

    @staticmethod
    def _gini_from_counts(positive, total):
        """Compute Gini impurity using class counts."""
        negative = total - positive
        p0 = negative / total
        p1 = positive / total
        return 1.0 - p0**2 - p1**2

    def _best_split(self, X, y):
        """Find the split minimizing weighted Gini impurity.

        Sort each feature once and evaluate candidate splits
        using cumulative positive-class counts.
        """
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_score = float("inf")

        for feature in range(n_features):
            order = np.argsort(X[:, feature], kind="mergesort")
            values = X[order, feature]
            labels = y[order]

            # A valid threshold must separate distinct values.
            distinct = values[:-1] < values[1:]

            left_sizes = np.arange(1, n_samples)
            right_sizes = n_samples - left_sizes

            valid = (
                distinct
                & (left_sizes >= self.min_samples_leaf)
                & (right_sizes >= self.min_samples_leaf)
            )

            if not np.any(valid):
                continue

            # Number of positive labels in the left partition.
            left_positive = np.cumsum(labels)[:-1]
            right_positive = np.sum(labels) - left_positive

            left_gini = self._gini_from_counts(
                left_positive[valid],
                left_sizes[valid],
            )
            right_gini = self._gini_from_counts(
                right_positive[valid],
                right_sizes[valid],
            )

            scores = (
                left_sizes[valid] / n_samples * left_gini
                + right_sizes[valid] / n_samples * right_gini
            )

            candidate_index = np.argmin(scores)
            candidate_score = float(scores[candidate_index])

            if candidate_score < best_score:
                split_positions = np.flatnonzero(valid)
                position = split_positions[candidate_index]

                # Stable midpoint between consecutive values.
                a = values[position]
                b = values[position + 1]
                threshold = a + (b - a) / 2.0

                best_score = candidate_score
                best_feature = feature
                best_threshold = float(threshold)

        return best_feature, best_threshold, best_score

    def _build_tree(self, X, y, depth):
        """Recursively construct the decision tree."""
        n_samples = len(y)
        positive_count = int(np.sum(y))
        majority_class = int(positive_count > n_samples / 2)

        # Store a leaf when no further splitting is allowed.
        stop = (
            positive_count == 0
            or positive_count == n_samples
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
            or (
                self.max_depth is not None
                and depth >= self.max_depth
            )
        )

        if stop:
            return {"values": majority_class}

        feature, threshold, split_score = self._best_split(X, y)

        if feature is None:
            return {"values": majority_class}

        parent_gini = float(
            self._gini_from_counts(positive_count, n_samples)
        )

        # Do not split when impurity cannot be reduced.
        if split_score >= parent_gini - 1e-12:
            return {"values": majority_class}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(
                X[left_mask], y[left_mask], depth + 1
            ),
            "right": self._build_tree(
                X[right_mask], y[right_mask], depth + 1
            ),
        }

    def fit(self, X, y):
        """Train the decision tree on binary labeled data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")
        if len(y) != len(X):
            raise ValueError("X and y must have matching rows.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1.")

        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(
            X, y.astype(int), depth=0
        )
        return self

    def _predict_one(self, x):
        """Traverse the tree for one observation."""
        node = self.tree_

        while "values" not in node:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        return node["values"]

    def predict(self, X):
        """Predict binary labels for multiple observations."""
        if self.tree_ is None:
            raise RuntimeError("The model must be fitted first.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Invalid feature dimensions.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        return np.array(
            [self._predict_one(row) for row in X],
            dtype=int,
        )
