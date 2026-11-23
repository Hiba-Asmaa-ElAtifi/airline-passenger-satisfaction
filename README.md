# Airline Passenger Satisfaction

**Integrated Machine Learning Project — Academic Year 2026–2027**

Université Internationale de Rabat (UIR) — Group 17

---

## Overview

Airline passenger satisfaction depends on several aspects of the travel experience: the passenger's profile, the characteristics of the flight and the quality of onboard and airport services.

This project studies how these factors relate to overall satisfaction and uses them to predict whether a passenger is **satisfied** or **neutral/dissatisfied**. The task is treated as a supervised binary classification problem using the classical algorithms covered in the Machine Learning course.

Rather than focusing only on prediction accuracy, we pay particular attention to data quality, the justification of preprocessing choices, the comparison of models and the interpretation of results. In particular, we aim to identify which services and passenger characteristics are most strongly associated with satisfaction.

## Dataset

The project uses the [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset, available on Kaggle.

| Property | Value |
|---|---|
| Total observations | 129,880 |
| Training set (`Dataset/train.csv`) | 103,904 |
| Test set (`Dataset/test.csv`) | 25,976 |
| Input features | 22 |
| Target variable | `satisfaction` |
| Classes | neutral or dissatisfied (56.7%), satisfied (43.3%) |

The input features describe three aspects of the passenger experience:

| Group | Features |
|---|---|
| Passenger characteristics | Gender, Customer Type, Age |
| Flight information | Type of Travel, Class, Flight Distance, Departure Delay, Arrival Delay |
| Service ratings | 14 services rated from 1 to 5 (0 = not applicable), such as Online Boarding, Seat Comfort, Inflight Entertainment and Inflight Wi-Fi |

The columns `id` and `Unnamed: 0` are identifiers and are not used for prediction. The original training and test files are kept separate: the test set is only used for the final evaluation.

## Methodology

**1. Exploratory Data Analysis**
Analysis of the dataset structure, variable distributions, missing values, duplicates, class balance and the relationships between service ratings and satisfaction. Particular attention is given to zero-valued service ratings, which represent non-applicable evaluations rather than negative ratings.

**2. Data Preprocessing**
Handling of missing values, encoding of categorical variables, feature scaling and representation of zero ratings. All transformations are fitted on the training data only and then applied to the validation and test data, in order to avoid data leakage. Alternative choices are compared to measure their effect on classification performance.

**3. Model Development**
The classification models covered in the course are trained and compared. Hyperparameters are selected on a validation set created from the training data.

| Model | Hyperparameters explored |
|---|---|
| K-Nearest Neighbors | Number of neighbors |
| Logistic Regression | Regularisation strength `C` |
| Linear Discriminant Analysis (LDA) | — |
| Quadratic Discriminant Analysis (QDA) | — |
| Decision Tree | Maximum depth |
| Random Forest | Number of trees |

**4. Evaluation and Interpretation**
Models are evaluated with accuracy, precision, recall, F1-score and confusion matrices. The selected model is then evaluated once on the test set. Finally, we analyse which passenger characteristics and services contribute most to the predictions.

## Main Findings from the Exploratory Analysis

- Travel class and type of travel show the largest differences in satisfaction: 69% of Business class passengers are satisfied, compared with 19% in Economy.
- Online boarding, inflight entertainment and seat comfort show the largest rating differences between satisfied and dissatisfied passengers.
- Zero ratings carry information: almost all passengers who rated Inflight Wi-Fi as 0 (not applicable) are satisfied. Zeros should therefore not simply be replaced by an average value.
- Departure and arrival delays are strongly correlated (r = 0.965). The arrival delay contains 310 missing values in the training set.

The complete analysis is available in [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb).

## Project Structure

```text
airline-passenger-satisfaction/
│
├── configs/
│   └── config.yaml           # File paths, validation settings and hyperparameters
├── Dataset/
│   ├── train.csv             # Original training data
│   └── test.csv              # Original test data (final evaluation only)
├── models/                   # Saved trained model
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── src/
│   ├── data.py               # Data loading and train/validation split
│   ├── preprocessing.py      # Imputation, encoding, scaling and zero-rating indicators
│   ├── model.py              # Model definitions
│   ├── train.py              # Training and hyperparameter selection
│   ├── evaluate.py           # Metrics and confusion matrix
│   └── utils.py              # Configuration loading, model saving and loading
├── main.py                   # Training entry point
├── test.py                   # Independent inference on new samples
├── requirements.txt          # Python dependencies
└── README.md
```

Some modules and scripts are still under development and will be added as the pipeline is implemented.

## Getting Started

### Prerequisites

Python 3.10 or later and Git.

### 1. Clone the repository

```bash
git clone https://github.com/Hiba-Asmaa-ElAtifi/airline-passenger-satisfaction.git
cd airline-passenger-satisfaction
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model

```bash
python main.py
```

Runs the complete pipeline: data loading, preprocessing, training, model selection on the validation set and saving of the best model in `models/`.

### 5. Run inference on new samples

```bash
python test.py
```

Loads the saved model and predicts the satisfaction of new passengers, without retraining.

All settings (file paths, validation split, random seed and hyperparameters) are defined in `configs/config.yaml`.

> **Current implementation:** `main.py` loads and displays the project configuration. The training pipeline and the inference script `test.py` will be added in the next development stages.

## Current Progress

| Stage | Status |
|---|---|
| Dataset selection | Completed |
| Exploratory data analysis | Completed |
| Project structure and configuration | Completed |
| Data preprocessing | In progress |
| Model training and comparison | Planned |
| Evaluation and interpretation | Planned |
| Training and inference scripts | Planned |
| Final report and presentation | Planned |

## Team

**Group 17 — Université Internationale de Rabat**

- Hiba Asmaa El Atifi
- Salma Alami
- Hayam Mimouni
- Khaoula Marhrani

## Academic Context

This repository was developed as part of the Integrated Machine Learning Project (2026–2027) at Université Internationale de Rabat, under the supervision of Prof. Youness Moukafih.

The project emphasises reproducibility, justified preprocessing choices, understandable models and an evidence-based interpretation of the results.