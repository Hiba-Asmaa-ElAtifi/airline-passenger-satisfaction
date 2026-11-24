
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Name of the column that the model must predict
TARGET_COLUMN = "satisfaction"

# Identifier columns that must not be used for prediction
DROP_COLUMNS = ["Unnamed: 0", "id"]


def load_dataset(csv_path):
    # Read the CSV file and return a pandas DataFrame
    return pd.read_csv(csv_path)


def split_features_target(df):
    # Check that all required columns exist
    required_columns = DROP_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    # Stop if the dataset does not contain the expected columns
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # X contains the 22 input features
    X = df.drop(columns=required_columns).copy()

    # y contains the target: passenger satisfaction
    y = df[TARGET_COLUMN].copy()

    return X, y


def load_data(data_dir="Dataset", validation_size=0.2,
              random_state=42):

    # Define the dataset folder
    data_dir = Path(data_dir)

    # Load the original training and test datasets
    train_df = load_dataset(data_dir / "train.csv")
    test_df = load_dataset(data_dir / "test.csv")

    # Separate features and target
    X, y = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    # Ensure train and test contain identical feature columns
    if list(X.columns) != list(X_test.columns):
        raise ValueError(
            "Train and test feature columns do not match."
        )

    # Create a validation set from the training dataset only
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y
    )

    # Return the three datasets and their target labels
    return X_train, X_val, X_test, y_train, y_val, y_test
