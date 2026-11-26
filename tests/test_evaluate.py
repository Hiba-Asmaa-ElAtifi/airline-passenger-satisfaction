import numpy as np

from src.evaluate import (
    evaluate_model,
    compare_models
)

from src.models.logistic_regression import LogisticRegressionScratch
from src.models.knn import KNNScratch

def test_evaluate_model():

    # Create a small dataset
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3]
    ])

    y = np.array([0, 0, 1, 1])

    # Train a Logistic Regression model
    model = LogisticRegressionScratch(
        learning_rate=0.1,
        epochs=1000
    )

    model.fit(X, y)

    # Evaluate the model
    results = evaluate_model(model, X, y)

    # Verify the results
    assert "metrics" in results
    assert "confusion_matrix" in results
    assert "predictions" in results

    assert len(results["predictions"]) == len(y)
    assert results["confusion_matrix"].shape == (2, 2)

    print("Evaluate model test passed!")

def test_compare_models():

    # Create a small dataset
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3]
    ])

    y = np.array([0, 0, 1, 1])

    # Train Logistic Regression
    logistic_model = LogisticRegressionScratch(
        learning_rate=0.1,
        epochs=1000
    )
    logistic_model.fit(X, y)

    # Train KNN
    knn_model = KNNScratch(k=3)
    knn_model.fit(X, y)

    # Store trained models
    models = {
        "Logistic Regression": logistic_model,
        "KNN": knn_model
    }

    # Compare models
    results = compare_models(models, X, y)

    # Verify the results
    assert "Logistic Regression" in results
    assert "KNN" in results

    for model_name, result in results.items():
        assert "metrics" in result
        assert "confusion_matrix" in result
        assert len(result["predictions"]) == len(y)

    print("Compare models test passed!")

if __name__ == "__main__":
    test_evaluate_model()
    test_compare_models()