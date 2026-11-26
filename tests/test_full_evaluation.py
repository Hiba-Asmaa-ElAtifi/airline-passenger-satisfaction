
import pandas as pd

from src.data import load_data
from src.preprocessing import build_preprocessor

from src.train import (
    tune_logistic_regression,
    tune_decision_tree,
    tune_lda,
    tune_knn
)

from src.evaluate import (
    compare_models,
    calculate_feature_importance
)

from src.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_comparison,
    plot_feature_importance
)


# 1. Load dataset
X_train, X_val, X_test, y_train, y_val, y_test = load_data()


# 2. Encode target labels
label_mapping = {
    "neutral or dissatisfied": 0,
    "satisfied": 1
}

y_train = y_train.map(label_mapping)
y_val = y_val.map(label_mapping)
y_test = y_test.map(label_mapping)


# 3. Build preprocessing pipeline
preprocessor = build_preprocessor(scale_numeric=True)


# 4. Fit on training data
X_train_processed = preprocessor.fit_transform(X_train)


# 5. Transform validation data
X_val_processed = preprocessor.transform(X_val)


# 6. Transform test data
X_test_processed = preprocessor.transform(X_test)


# 7. Train Logistic Regression
logistic_model, logistic_params, logistic_results = tune_logistic_regression(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val
)

print("\n=== LOGISTIC REGRESSION ===")
print("Best parameters:", logistic_params)


# 8. Train Decision Tree
decision_tree_model, decision_tree_params, decision_tree_results = tune_decision_tree(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val
)

print("\n=== DECISION TREE ===")
print("Best parameters:", decision_tree_params)


# 9. Train LDA
lda_model, lda_params, lda_results = tune_lda(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val
)

print("\n=== LDA ===")
print("Best parameters:", lda_params)


# 10. Train KNN
knn_model, knn_params, knn_results = tune_knn(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val
)

print("\n=== KNN ===")
print("Best parameters:", knn_params)


# 11. Collect trained models
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "LDA": lda_model,
    "KNN": knn_model
}

print("\n=== TRAINED MODELS ===")

for model_name in models:
    print(model_name)


# 12. Evaluate all models on validation data
validation_results = compare_models(
    models,
    X_val_processed,
    y_val
)

print("\n=== VALIDATION EVALUATION ===")

for model_name, result in validation_results.items():
    print(f"\n{model_name}")

    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.4f}")


# 13. Visualize validation model comparison
plot_model_comparison(validation_results)


# 14. Visualize confusion matrices for all models
for model_name, result in validation_results.items():

    plot_confusion_matrix(
        result["confusion_matrix"],
        f"{model_name} - Validation"
    )


# 15. Visualize Decision Tree ROC curve
plot_roc_curve(
    validation_results["Decision Tree"]["roc_curve"],
    "Decision Tree - Validation"
)


# 16. Visualize Decision Tree Precision-Recall curve
plot_precision_recall_curve(
    validation_results["Decision Tree"]["precision_recall_curve"],
    "Decision Tree - Validation"
)


# 17. Final evaluation on test data
test_models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "LDA": lda_model,
    "KNN": knn_model
}

print("\nEvaluating all four models on the test dataset...")

test_results = compare_models(
    test_models,
    X_test_processed,
    y_test
)


# 18. Display final test evaluation
print("\n=== FINAL TEST EVALUATION ===")

for model_name, result in test_results.items():

    print(f"\n{model_name}")

    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.4f}")


# 19. Final Decision Tree confusion matrix
plot_confusion_matrix(
    test_results["Decision Tree"]["confusion_matrix"],
    "Decision Tree - Test"
)


# 20. Final Decision Tree ROC curve
plot_roc_curve(
    test_results["Decision Tree"]["roc_curve"],
    "Decision Tree - Test"
)


# 21. Final Decision Tree Precision-Recall curve
plot_precision_recall_curve(
    test_results["Decision Tree"]["precision_recall_curve"],
    "Decision Tree - Test"
)


# 22. Analyze Decision Tree feature importance
feature_importances = calculate_feature_importance(
    decision_tree_model,
    X_train_processed,
    y_train
)

feature_importance = pd.DataFrame({
    "Feature": preprocessor.get_feature_names_out(),
    "Importance": feature_importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n=== TOP 15 IMPORTANT FEATURES ===")

print(
    feature_importance.head(15).to_string(index=False)
)


# 23. Visualize feature importance
plot_feature_importance(feature_importance)

print("\nFeature importance chart saved successfully.")


# 24. Verify model lists before CSV export
print(
    "\nModels in validation_results:",
    list(validation_results.keys())
)

print(
    "Models in test_results:",
    list(test_results.keys())
)


# 25. Export evaluation results to CSV
evaluation_rows = []

for model_name, result in validation_results.items():
    row = {
        "Model": model_name,
        "Dataset": "Validation",
        **result["metrics"]
    }
    evaluation_rows.append(row)

for model_name, result in test_results.items():
    row = {
        "Model": model_name,
        "Dataset": "Test",
        **result["metrics"]
    }
    evaluation_rows.append(row)

evaluation_df = pd.DataFrame(evaluation_rows)

evaluation_df.to_csv(
    "results/evaluation_results.csv",
    index=False
)

print("\n=== EVALUATION RESULTS EXPORTED ===")
print(evaluation_df.to_string(index=False))
