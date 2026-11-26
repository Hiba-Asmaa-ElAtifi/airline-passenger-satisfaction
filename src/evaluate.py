
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }

    return metrics


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )


def get_model_scores(model, X):
    """
    Retrieve continuous prediction scores.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores[:, 1] - scores[:, 0]

    else:
        raise AttributeError(
            "This model does not provide prediction scores."
        )


def calculate_roc_auc(y_true, y_scores):
    """
    Calculate ROC curve and AUC score.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_scores
    )

    auc = roc_auc_score(
        y_true,
        y_scores
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc
    }


def calculate_precision_recall(y_true, y_scores):
    """
    Calculate Precision-Recall curve and Average Precision.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_scores
    )

    average_precision = average_precision_score(
        y_true,
        y_scores
    )

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": average_precision
    }


def evaluate_model(model, X, y_true):
    """
    Evaluate a trained model on a dataset.
    """

    y_pred = model.predict(X)

    y_scores = get_model_scores(
        model,
        X
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    roc_results = calculate_roc_auc(
        y_true,
        y_scores
    )

    pr_results = calculate_precision_recall(
        y_true,
        y_scores
    )

    metrics["roc_auc"] = roc_results["auc"]

    metrics["average_precision"] = (
        pr_results["average_precision"]
    )

    cm = calculate_confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "predictions": y_pred,
        "scores": y_scores,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": roc_results,
        "precision_recall_curve": pr_results
    }


def compare_models(models, X, y_true):
    """
    Evaluate and compare multiple trained models.
    """

    results = {}

    for model_name, model in models.items():

        results[model_name] = evaluate_model(
            model,
            X,
            y_true
        )

    return results


def calculate_feature_importance(model, X, y):
    """
    Calculate feature importance using weighted Gini impurity reduction.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    importances = np.zeros(model.n_features_)
    total_samples = len(y)

    def calculate_gini(labels):

        if len(labels) == 0:
            return 0

        _, counts = np.unique(
            labels,
            return_counts=True
        )

        probabilities = counts / len(labels)

        return 1 - np.sum(probabilities ** 2)

    def traverse(node, X_node, y_node):

        if "values" in node:
            return

        feature = node["feature"]
        threshold = node["threshold"]

        left_mask = X_node[:, feature] <= threshold
        right_mask = ~left_mask

        X_left = X_node[left_mask]
        y_left = y_node[left_mask]

        X_right = X_node[right_mask]
        y_right = y_node[right_mask]

        parent_gini = calculate_gini(y_node)

        weighted_children_gini = (
            len(y_left) / len(y_node) * calculate_gini(y_left)
            + len(y_right) / len(y_node) * calculate_gini(y_right)
        )

        improvement = parent_gini - weighted_children_gini

        importances[feature] += (
            len(y_node) / total_samples
        ) * improvement

        traverse(node["left"], X_left, y_left)
        traverse(node["right"], X_right, y_right)

    if model.tree_ is None:
        raise RuntimeError(
            "The model must be fitted first."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    traverse(model.tree_, X, y)

    total_importance = np.sum(importances)

    if total_importance > 0:
        importances /= total_importance

    return importances
