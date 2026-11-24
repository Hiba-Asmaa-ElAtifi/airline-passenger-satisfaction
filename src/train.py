
"""
Training and hyperparameter tuning module.

Project: Airline Passenger Satisfaction

Responsibilities:
    - Train machine learning models from scratch.
    - Compare hyperparameter configurations.
    - Select the best model using validation accuracy.

Data loading, preprocessing, detailed evaluation, and model
serialization are handled by other project modules.
"""

import numpy as np

from src.models.logistic_regression import LogisticRegressionScratch
from src.models.decision_tree import DecisionTreeScratch
from src.models.lda import LDAScratch
from src.models.knn import KNNScratch
# ============================================================
# 1. COMMON TRAINING UTILITIES
# ============================================================

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    float
        Proportion of correctly classified observations.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Invalid prediction or target dimensions.")

    return float(np.mean(y_true == y_pred))


# ============================================================
# 2. LOGISTIC REGRESSION - TRAINING
# ============================================================

def train_logistic_regression(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=100
):
    """
    Train Logistic Regression from scratch.

    Returns
    -------
    LogisticRegressionScratch
        Fitted Logistic Regression model.
    """
    model = LogisticRegressionScratch(
        learning_rate=learning_rate,
        epochs=epochs
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# 3. LOGISTIC REGRESSION - HYPERPARAMETER TUNING
# ============================================================

def tune_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rates=(0.01, 0.1),
    epochs_options=(100, 500, 1000)
):
    """
    Select Logistic Regression hyperparameters using
    validation accuracy.

    Returns
    -------
    tuple
        best_model, best_params, results
    """
    best_model = None
    best_params = None
    best_score = -1.0
    results = []

    for learning_rate in learning_rates:
        for epochs in epochs_options:

            model = train_logistic_regression(
                X_train,
                y_train,
                learning_rate=learning_rate,
                epochs=epochs
            )

            predictions = model.predict(X_val)
            accuracy = compute_accuracy(y_val, predictions)

            result = {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "validation_accuracy": accuracy
            }
            results.append(result)

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_params = {
                    "learning_rate": learning_rate,
                    "epochs": epochs
                }

    return best_model, best_params, results


# ============================================================
# 4. LOGISTIC REGRESSION - SINGLE EXPERIMENT
# ============================================================

def run_logistic_regression_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=0.1,
    epochs=500
):
    """
    Train one Logistic Regression configuration.

    Returns
    -------
    tuple
        Trained model and experiment results.
    """
    model = train_logistic_regression(
        X_train,
        y_train,
        learning_rate=learning_rate,
        epochs=epochs
    )

    predictions = model.predict(X_val)

    results = {
        "validation_accuracy": compute_accuracy(
            y_val, predictions
        ),
        "initial_training_loss": model.loss_history[0],
        "final_training_loss": model.loss_history[-1]
    }

    return model, results


# ============================================================
# 5. DECISION TREE - TRAINING
# ============================================================

def train_decision_tree(
    X_train,
    y_train,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
):
    """
    Train a Decision Tree classifier from scratch.

    Parameters
    ----------
    X_train : array-like
        Preprocessed training features.
    y_train : array-like
        Binary target labels.
    max_depth : int
        Maximum tree depth.
    min_samples_split : int
        Minimum observations required to split a node.
    min_samples_leaf : int
        Minimum observations required in each leaf.

    Returns
    -------
    DecisionTreeScratch
        Fitted Decision Tree model.
    """
    model = DecisionTreeScratch(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# 6. DECISION TREE - HYPERPARAMETER TUNING
# ============================================================

def tune_decision_tree(
    X_train,
    y_train,
    X_val,
    y_val,
    max_depth_options=(3, 5, 8, 10, 12),
    min_samples_split=10,
    min_samples_leaf=5
):
    """
    Compare Decision Tree depths on validation data.

    The other stopping parameters remain fixed so that
    the effect of maximum depth can be studied separately.

    Returns
    -------
    tuple
        best_model : DecisionTreeScratch
            Best fitted model.
        best_params : dict
            Hyperparameters of the best model.
        results : list of dict
            Training and validation accuracy for each depth.
    """
    best_model = None
    best_params = None
    best_score = -1.0
    results = []

    for max_depth in max_depth_options:

        model = train_decision_tree(
            X_train,
            y_train,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_accuracy = compute_accuracy(
            y_train, train_predictions
        )
        val_accuracy = compute_accuracy(
            y_val, val_predictions
        )

        result = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy
        }
        results.append(result)

        if val_accuracy > best_score:
            best_score = val_accuracy
            best_model = model
            best_params = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf
            }

    return best_model, best_params, results


# ============================================================
# 7. DECISION TREE - SINGLE EXPERIMENT
# ============================================================

def run_decision_tree_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5
):
    """
    Train and evaluate one Decision Tree configuration.

    Returns
    -------
    tuple
        Trained model and experiment results.
    """
    model = train_decision_tree(
        X_train,
        y_train,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    results = {
        "train_accuracy": compute_accuracy(
            y_train, train_predictions
        ),
        "validation_accuracy": compute_accuracy(
            y_val, val_predictions
        )
    }

    return model, results

# ============================================================
# 8. LDA - TRAINING
# ============================================================

def train_lda(
    X_train,
    y_train,
    regularization=1e-4
):
    """
    Train Linear Discriminant Analysis from scratch.

    Parameters
    ----------
    X_train : array-like
        Preprocessed training features.
    y_train : array-like
        Binary target labels (0 or 1).
    regularization : float
        Value added to the covariance diagonal to
        improve numerical stability.

    Returns
    -------
    LDAScratch
        Fitted LDA classifier.
    """
    model = LDAScratch(
        regularization=regularization
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# 9. LDA - HYPERPARAMETER TUNING
# ============================================================

def tune_lda(
    X_train,
    y_train,
    X_val,
    y_val,
    regularization_options=(1e-4, 1e-3, 1e-2, 1e-1)
):
    """
    Compare LDA regularization values using validation accuracy.

    Returns
    -------
    tuple
        best_model : LDAScratch
            Best fitted LDA model.
        best_params : dict
            Best regularization parameter.
        results : list of dict
            Training and validation accuracy for each configuration.
    """
    best_model = None
    best_params = None
    best_score = -1.0
    results = []

    for regularization in regularization_options:

        model = train_lda(
            X_train,
            y_train,
            regularization=regularization
        )

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_accuracy = compute_accuracy(
            y_train,
            train_predictions
        )

        val_accuracy = compute_accuracy(
            y_val,
            val_predictions
        )

        result = {
            "regularization": regularization,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy
        }

        results.append(result)

        if val_accuracy > best_score:
            best_score = val_accuracy
            best_model = model
            best_params = {
                "regularization": regularization
            }

    return best_model, best_params, results


# ============================================================
# 10. LDA - SINGLE EXPERIMENT
# ============================================================

def run_lda_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    regularization=1e-4
):
    """
    Train and evaluate one LDA configuration.

    Returns
    -------
    tuple
        Fitted LDA model and experiment results.
    """
    model = train_lda(
        X_train,
        y_train,
        regularization=regularization
    )

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    results = {
        "train_accuracy": compute_accuracy(
            y_train,
            train_predictions
        ),
        "validation_accuracy": compute_accuracy(
            y_val,
            val_predictions
        )
    }

    return model, results

# ============================================================
# KNN - TRAINING
# ============================================================

def train_knn(X_train, y_train, k=5, batch_size=32):
    """
    Fit KNN from scratch.

    KNN stores training observations rather than
    learning model coefficients.
    """
    model = KNNScratch(k=k, batch_size=batch_size)
    model.fit(X_train, y_train)
    return model


# ============================================================
# KNN - HYPERPARAMETER TUNING
# ============================================================

def tune_knn(
    X_train,
    y_train,
    X_val,
    y_val,
    k_options=(3, 5, 7, 11),
    batch_size=32
):
    """
    Select the best k using validation accuracy.

    Distances are calculated once for the largest k,
    then reused for all smaller k values.

    Returns
    -------
    best_model : KNNScratch
    best_params : dict
    results : list of dict
    """
    import numpy as np

    k_values = sorted(set(k_options))

    if not k_values or any(
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
        for k in k_values
    ):
        raise ValueError("k_options must contain positive integers.")

    max_k = max(k_values)

    # Fit once with the largest neighborhood size.
    search_model = train_knn(
        X_train,
        y_train,
        k=max_k,
        batch_size=batch_size
    )

    X_val = search_model._validate_query(X_val)
    y_val = np.asarray(y_val).ravel()

    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val must have matching rows.")

    if not np.all(np.isin(y_val, [0, 1])):
        raise ValueError("Validation labels must be 0 or 1.")

    correct_counts = {k: 0 for k in k_values}

    # Process validation passengers in small batches.
    for start in range(0, len(X_val), batch_size):
        end = min(start + batch_size, len(X_val))

        X_batch = X_val[start:end]
        y_batch = y_val[start:end]

        # Find the max_k nearest neighbors once.
        neighbor_indices = search_model._find_neighbors(
            X_batch
        )

        neighbor_labels = search_model.y_train_[
            neighbor_indices
        ]

        # Cumulative class-1 votes for k=1,...,max_k.
        cumulative_votes = np.cumsum(
            neighbor_labels == 1,
            axis=1
        )

        for k in k_values:
            votes_for_one = cumulative_votes[:, k - 1]

            # Tie => class 0.
            predictions = (votes_for_one > k / 2).astype(int)

            correct_counts[k] += int(
                np.sum(predictions == y_batch)
            )

    results = []
    best_k = None
    best_score = -1.0

    for k in k_values:
        validation_accuracy = correct_counts[k] / len(y_val)

        results.append({
            "k": k,
            "validation_accuracy": validation_accuracy
        })

        if validation_accuracy > best_score:
            best_score = validation_accuracy
            best_k = k

    # KNN fitting is inexpensive: reuse the stored data.
    best_model = train_knn(
        X_train,
        y_train,
        k=best_k,
        batch_size=batch_size
    )

    best_params = {
        "k": best_k,
        "batch_size": batch_size
    }

    return best_model, best_params, results


# ============================================================
# KNN - SINGLE EXPERIMENT
# ============================================================

def run_knn_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    k=5,
    batch_size=32
):
    """
    Run one KNN experiment on validation data.
    """
    import numpy as np

    model = train_knn(
        X_train,
        y_train,
        k=k,
        batch_size=batch_size
    )

    predictions = model.predict(X_val)

    validation_accuracy = compute_accuracy(
        y_val,
        predictions
    )

    results = {
        "k": k,
        "validation_accuracy": validation_accuracy
    }

    return model, results
