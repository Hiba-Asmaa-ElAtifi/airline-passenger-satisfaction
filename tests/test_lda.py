
"""Integration test for LDA training and hyperparameter tuning.

Uses the real Airline Passenger Satisfaction dataset.
The official test set remains untouched.
"""

import numpy as np

from src.data import load_data
from src.preprocessing import build_preprocessor
from src.train import tune_lda
from src.models.lda import LDAScratch


TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}


def test_lda_synthetic():
    """Verify the LDA implementation on a small artificial dataset."""

    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [0.8, 0.9],
        [4.0, 4.0],
        [4.2, 4.1],
        [3.8, 3.9],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = LDAScratch(regularization=1e-4)
    model.fit(X, y)

    predictions = model.predict(X)

    assert np.array_equal(predictions, y)
    assert model.means_.shape == (2, 2)
    assert model.covariance_.shape == (2, 2)

    print("LDA synthetic test passed!")


def main():
    # 1. Verify the mathematical implementation.
    test_lda_synthetic()

    # 2. Load training and validation datasets.
    X_train, X_val, _, y_train, y_val, _ = load_data()

    # 3. Apply preprocessing.
    # LDA uses scaled numerical features.
    preprocessor = build_preprocessor(scale_numeric=True)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    # 4. Encode the binary target.
    y_train = y_train.map(TARGET_MAPPING).to_numpy()
    y_val = y_val.map(TARGET_MAPPING).to_numpy()

    assert not np.isnan(y_train.astype(float)).any()
    assert not np.isnan(y_val.astype(float)).any()

    print("=" * 60)
    print("LDA - TRAINING MODULE INTEGRATION TEST")
    print("=" * 60)
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Number of features:", X_train.shape[1])

    # 5. Run hyperparameter tuning through src/train.py.
    best_model, best_params, results = tune_lda(
        X_train,
        y_train,
        X_val,
        y_val,
        regularization_options=(1e-4, 1e-3, 1e-2, 1e-1),
    )

    # 6. Display experiment results.
    print("\nHYPERPARAMETER TUNING RESULTS")

    for result in results:
        print(
            f"Regularization: {result['regularization']} | "
            f"Train: {result['train_accuracy']:.4f} | "
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

    # 7. Verify the integration.
    assert best_model is not None
    assert len(results) == 4
    assert len(best_model.predict(X_val)) == len(y_val)

    print("\nLDA integration test passed!")


if __name__ == "__main__":
    main()
