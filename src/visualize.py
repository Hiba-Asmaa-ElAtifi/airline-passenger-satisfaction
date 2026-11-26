
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, model_name):
    """
    Plot and save the confusion matrix.
    """

    plt.figure(figsize=(6, 5))

    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    classes = ["Neutral/Dissatisfied", "Satisfied"]

    plt.xticks(
        np.arange(len(classes)),
        classes,
        rotation=20
    )

    plt.yticks(
        np.arange(len(classes)),
        classes
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_roc_curve(roc_results, model_name):
    """
    Plot and save the ROC curve.
    """

    fpr = roc_results["fpr"]
    tpr = roc_results["tpr"]
    auc = roc_results["auc"]

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_precision_recall_curve(pr_results, model_name):
    """
    Plot and save the Precision-Recall curve.
    """

    precision = pr_results["precision"]
    recall = pr_results["recall"]
    average_precision = pr_results["average_precision"]

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AP = {average_precision:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    filename = f"precision_recall_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_model_comparison(results):
    """
    Compare and save the performance metrics of multiple models.
    """

    models = list(results.keys())

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    data = {}

    for metric in metrics:
        data[metric] = [
            results[model]["metrics"][metric]
            for model in models
        ]

    df = pd.DataFrame(data, index=models)

    ax = df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=20)
    plt.legend(title="Metrics")
    plt.tight_layout()

    plt.savefig(
        "model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_feature_importance(feature_importance, output_dir="results/figures"):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    top_features = feature_importance.head(15).sort_values(
        by="Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Decision Tree Feature Importances")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "feature_importance.png"),
        dpi=300
    )

    plt.close()
