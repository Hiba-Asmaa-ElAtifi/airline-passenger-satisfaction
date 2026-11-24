
"""Integration test for Decision Tree training and tuning.

Uses the real Airline Passenger Satisfaction dataset.
The official test set remains untouched.
"""

import numpy as np

from src.data import load_data
from src.preprocessing import build_preprocessor
from src.train import tune_decision_tree


TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}


def main():
    # 1. Load training and validation datasets.
    X_train, X_val, _, y_train, y_val, _ = load_data()

    # 2. Apply the same preprocessing to both datasets.
    preprocessor = build_preprocessor(scale_numeric=False)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    # 3. Encode the binary target.
    y_train = y_train.map(TARGET_MAPPING).to_numpy()
    y_val = y_val.map(TARGET_MAPPING).to_numpy()

    assert not np.isnan(y_train.astype(float)).any()
    assert not np.isnan(y_val.astype(float)).any()

    print("=" * 60)
    print("DECISION TREE - TRAINING MODULE INTEGRATION TEST")
    print("=" * 60)
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # 4. Run hyperparameter tuning through src/train.py.
    best_model, best_params, results = tune_decision_tree(
        X_train,
        y_train,
        X_val,
        y_val,
        max_depth_options=(3, 5, 8, 10, 12),
        min_samples_split=10,
        min_samples_leaf=5,
    )

    # 5. Display the experiment results.
    print("\nHYPERPARAMETER TUNING RESULTS")

    for result in results:
        print(
            f"Depth: {result['max_depth']} | "
            f"Train: {result['train_accuracy']:.4f} | "
            f"Validation: {result['validation_accuracy']:.4f}"
        )

    print("\nBest parameters:", best_params)
    print(
        "Best validation accuracy:",
        round(max(r["validation_accuracy"] for r in results), 4)
    )

    # 6. Verify the integration.
    assert best_model is not None
    assert len(results) == 5
    assert len(best_model.predict(X_val)) == len(y_val)

    print("\nDecision Tree integration test passed!")


if __name__ == "__main__":
    main()
