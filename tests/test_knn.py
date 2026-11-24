
"""Integration test for KNN training and hyperparameter tuning.

Project: Airline Passenger Satisfaction
The official test set remains untouched.
"""

import time
import numpy as np

from src.data import load_data
from src.preprocessing import build_preprocessor
from src.models.knn import KNNScratch
from src.train import tune_knn


TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}


def test_knn_synthetic():
    """Verify KNN majority voting on a simple dataset."""

    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [0.8, 0.9],
        [4.0, 4.0],
        [4.2, 4.1],
        [3.8, 3.9],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = KNNScratch(k=3, batch_size=2)
    model.fit(X, y)

    predictions = model.predict(
        np.array([[1.1, 1.0], [4.1, 4.0]])
    )

    assert np.array_equal(predictions, [0, 1])

    print("KNN synthetic test passed!")


def main():
    # 1. Verify the mathematical implementation.
    test_knn_synthetic()

    # 2. Load training and validation datasets.
    X_train, X_val, _, y_train, y_val, _ = load_data()

    # 3. Scale numeric features for distance calculations.
    preprocessor = build_preprocessor(scale_numeric=True)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    # 4. Encode satisfaction labels.
    y_train = y_train.map(TARGET_MAPPING).to_numpy()
    y_val = y_val.map(TARGET_MAPPING).to_numpy()

    assert not np.isnan(y_train.astype(float)).any()
    assert not np.isnan(y_val.astype(float)).any()

    print("=" * 60)
    print("KNN - TRAINING MODULE INTEGRATION TEST")
    print("=" * 60)
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Number of features:", X_train.shape[1])

    # 5. Tune k using validation data.
    print("\nStarting KNN hyperparameter tuning...")

    start_time = time.perf_counter()

    best_model, best_params, results = tune_knn(
        X_train,
        y_train,
        X_val,
        y_val,
        k_options=(3, 5, 7, 11),
        batch_size=32,
    )

    elapsed_time = time.perf_counter() - start_time

    # 6. Display results.
    print("\nHYPERPARAMETER TUNING RESULTS")

    for result in results:
        print(
            f"k={result['k']} | "
            f"Validation: {result['validation_accuracy']:.4f}"
        )

    print("\nBest parameters:", best_params)
    print(
        "Best validation accuracy:",
        round(
            max(r["validation_accuracy"] for r in results),
            4
        )
    )
    print(f"Tuning time: {elapsed_time:.2f} seconds")

    # 7. Verify integration without repeating the full
    # validation prediction, which is computationally expensive.
    assert best_model is not None
    assert len(results) == 4
    assert best_params["k"] in (3, 5, 7, 11)
    assert all(
        0.0 <= result["validation_accuracy"] <= 1.0
        for result in results
    )

    # Check prediction shape on a small validation subset.
    sample_predictions = best_model.predict(X_val[:10])

    assert len(sample_predictions) == 10
    assert np.all(np.isin(sample_predictions, [0, 1]))

    print("\nKNN integration test passed!")


if __name__ == "__main__":
    main()
