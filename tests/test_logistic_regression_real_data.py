
"""
Integration test: Logistic Regression with real passenger data.

Uses Salma's data and preprocessing modules,
and Hayam's Logistic Regression training module.
"""

import numpy as np
from src.train import tune_logistic_regression

from src.data import load_data
from src.preprocessing import build_preprocessor
from src.train import run_logistic_regression_experiment


def main():

    # STEP 1: Load data using Salma's module
    X_train, X_val, _, y_train, y_val, _ = load_data()

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # STEP 2: Preprocess using Salma's module
    preprocessor = build_preprocessor(scale_numeric=True)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # STEP 3: Encode the target labels
    target_mapping = {
        "neutral or dissatisfied": 0,
        "satisfied": 1
    }

    y_train_encoded = y_train.map(target_mapping).to_numpy()
    y_val_encoded = y_val.map(target_mapping).to_numpy()

    assert not np.isnan(y_train_encoded.astype(float)).any()
    assert not np.isnan(y_val_encoded.astype(float)).any()

    # STEP 4: Train and validate Hayam's model
    print("\nTraining Logistic Regression...")

    model, results = run_logistic_regression_experiment(
        X_train_processed,
        y_train_encoded,
        X_val_processed,
        y_val_encoded,
        learning_rate=0.1,
        epochs=500
    )

    # STEP 5: Display experimental results
    print("\nTraining completed!")
    print("Validation accuracy:",
          round(results["validation_accuracy"], 4))
    print("Initial training loss:",
          round(results["initial_training_loss"], 4))
    print("Final training loss:",
          round(results["final_training_loss"], 4))

    assert results["final_training_loss"] < results["initial_training_loss"]
    assert 0 <= results["validation_accuracy"] <= 1

    print("\nIntegration test passed!")


    # STEP 6: Hyperparameter tuning
    print("\nStarting hyperparameter tuning...")

    best_model, best_params, tuning_results = tune_logistic_regression(
        X_train_processed,
        y_train_encoded,
        X_val_processed,
        y_val_encoded,
        learning_rates=(0.01, 0.1),
        epochs_options=(100, 500, 1000)
    )

    # Display all configurations
    print("\nHyperparameter tuning results:")

    for result in tuning_results:
        print(
            f"Learning rate: {result['learning_rate']} | "
            f"Epochs: {result['epochs']} | "
            f"Validation accuracy: "
            f"{result['validation_accuracy']:.4f}"
        )

    # Display the best configuration
    print("\nBest hyperparameters:", best_params)
    print(
        "Best validation accuracy:",
        max(result["validation_accuracy"] for result in tuning_results)
    )
    print("Best model initial loss:", round(best_model.loss_history[0], 4))
    print("Best model final loss:", round(best_model.loss_history[-1], 4))

if __name__ == "__main__":
    main()
