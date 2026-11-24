
# ============================================================
# Airline Passenger Satisfaction
# Data Preprocessing Module
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Define categorical features
CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class"
]


# Define numerical features
NUMERICAL_COLUMNS = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]


def build_preprocessor(scale_numeric=True):
    """
    Build a reusable preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Optional standardization

    Categorical features:
        - Most frequent imputation
        - One-Hot Encoding
    """

    # Create numerical preprocessing steps
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    # Add scaling when requested
    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    # Create numerical pipeline
    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    # Create categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine numerical and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
        ],
        sparse_threshold=0
    )

    return preprocessor
