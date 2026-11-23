<a id="top"></a>

<div align="center">

<img src="assets/banner.svg" alt="Airline Passenger Satisfaction" width="100%">

<br>

![Python](https://img.shields.io/badge/Python-3.10+-344035?style=flat-square&logo=python&logoColor=CFD5BB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-80999A?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-binary%20classification-B0CEB1?style=flat-square&labelColor=344035)
![Course](https://img.shields.io/badge/UIR-2026–2027-926749?style=flat-square&labelColor=344035)

**Predicting whether an airline passenger is satisfied, and understanding which factors matter most.**

[Installation](#6-installation) · [Usage](#7-usage) · [Configuration](#8-configuration) · [Exploratory analysis](notebooks/01_dataset_exploration.ipynb)

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Key Findings from the EDA](#4-key-findings-from-the-eda)
5. [Repository Structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Usage](#7-usage)
8. [Configuration](#8-configuration)
9. [Results](#9-results)
10. [Project Status](#10-project-status)
11. [Team](#11-team)

---

## 1. Project Overview

Airline passenger satisfaction depends on several aspects of the travel experience, from flight characteristics and passenger profiles to the quality of onboard and airport services.

As part of an academic Machine Learning project, we investigate how these factors relate to overall passenger satisfaction and how they can be used to predict whether a passenger is **satisfied** or **neutral/dissatisfied**. 

Using supervised binary classification, we focus not only on making predictions but also on understanding the data, exploring preprocessing methods and analysing how different choices affect the results.
<div align="right"><a href="#top">Back to top ↑</a></div>

## 2. Dataset

Source: [Airline Passenger Satisfaction — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

| File | Passengers | Used for |
|:---|---:|:---|
| `Dataset/train.csv` | 103,904 | Training and validation |
| `Dataset/test.csv` | 25,976 | Final evaluation only |
| **Total** | **129,880** | |

Each row is one passenger, described by **22 input features** and the target column `satisfaction`.

| Passenger profile | Flight information | Service ratings (0–5) |
|:---|:---|:---|
| Gender | Type of Travel | Online Boarding |
| Customer Type | Class | Seat Comfort |
| Age | Flight Distance | Inflight Entertainment |
| | Departure Delay | Inflight Wi-Fi |
| | Arrival Delay | + 10 other services |

- **Target classes:** neutral or dissatisfied (56.7%) and satisfied (43.3%).
- **Rating of 0:** means *not applicable*, not a bad rating.
- **`id` and `Unnamed: 0`:** identifiers, removed before training.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 3. Methodology

The project follows four steps.

| Step | What is done | Where |
|:---|:---|:---|
| **1. Exploratory analysis** | Data quality, distributions, class balance, link between each feature and satisfaction, study of zero ratings | `notebooks/` |
| **2. Preprocessing** | Missing-value imputation, encoding of categorical features, scaling, representation of zero ratings | `src/preprocessing.py` |
| **3. Training** | Training of the course models, hyperparameter selection on a validation set | `src/model.py`, `src/train.py` |
| **4. Evaluation** | Accuracy, precision, recall, F1-score, confusion matrix, interpretation of the final model | `src/evaluate.py` |

Two rules are applied throughout:

- All preprocessing parameters (median, mean, standard deviation) are computed on the **training data only**, then applied to the validation and test data. This prevents data leakage.
- The test set is used **once**, at the very end, to evaluate the selected model.

### Models

| Model | Hyperparameters tested |
|:---|:---|
| K-Nearest Neighbors | number of neighbors `k` |
| Logistic Regression | regularisation strength `C` |
| Linear Discriminant Analysis (LDA) | none |
| Quadratic Discriminant Analysis (QDA) | none |
| Decision Tree | maximum depth |
| Random Forest | number of trees |

### Preprocessing comparisons

To measure the effect of preprocessing choices, the same model is trained twice with a single change:

| Experiment | Model | Compared settings |
|:---|:---|:---|
| Feature scaling | KNN | without vs. with standardisation |
| Zero ratings | Logistic Regression | zeros kept as values vs. zeros + indicator columns |

<div align="right"><a href="#top">Back to top ↑</a></div>

## 4. Key Findings from the EDA

| Finding | Consequence for the project |
|:---|:---|
| 69% of Business class passengers are satisfied, vs. 19% in Economy | Class and type of travel are strong candidate predictors |
| Online boarding, entertainment and seat comfort show the largest rating gaps between the two groups | Services to check in the model interpretation |
| Almost all passengers who rated Wi-Fi as 0 (*not applicable*) are satisfied | Zeros must not be replaced by an average; an indicator column is tested |
| Departure and arrival delays are strongly correlated (r = 0.965); 310 arrival delays are missing | Simple median imputation is enough |
| No duplicated rows | No row removal needed |

Full analysis: [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb)

<div align="right"><a href="#top">Back to top ↑</a></div>

## 5. Repository Structure

```text
airline-passenger-satisfaction/
├── assets/
│   └── banner.svg
├── configs/
│   └── config.yaml
├── Dataset/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── main.py
├── test.py
├── requirements.txt
└── README.md
```

| File | Role |
|:---|:---|
| `main.py` | Training entry point: runs the whole pipeline and saves the best model |
| `test.py` | Loads the saved model and predicts on new passengers, without retraining |
| `configs/config.yaml` | All paths and settings used by the code |
| `src/data.py` | Loads the CSV files and creates the train / validation split |
| `src/preprocessing.py` | Imputation, encoding, scaling and zero-rating indicators |
| `src/model.py` | Definition of the classification models |
| `src/train.py` | Training and selection of the best hyperparameters |
| `src/evaluate.py` | Metrics and confusion matrix |
| `src/utils.py` | Reading the configuration, saving and loading the model |

> Files in `src/`, `test.py` and `models/best_model.pkl` are being developed and will be added progressively.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 6. Installation

**Prerequisites:** Python 3.10 or later, and Git.

**Step 1 — Clone the repository**

```bash
git clone https://github.com/Hiba-Asmaa-ElAtifi/airline-passenger-satisfaction.git
cd airline-passenger-satisfaction
```

**Step 2 — Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Step 3 — Install the dependencies**

```bash
python -m pip install -r requirements.txt
```

The dataset is already included in `Dataset/`, so no extra download is needed.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 7. Usage

### Train the model

```bash
python main.py
```

This command loads the data, applies the preprocessing, trains the models, selects the best one on the validation set and saves it in `models/best_model.pkl`.

### Predict on new passengers

```bash
python test.py
```

This command loads the saved model and predicts the satisfaction of new passengers. It does not retrain anything, so `main.py` must have been run at least once before.

### Explore the analysis

Open `notebooks/01_dataset_exploration.ipynb` in Jupyter or VS Code to see the exploratory analysis with its code and figures.

> **Current state:** `main.py` currently loads and displays the configuration only. Training and `test.py` will be connected once the `src/` modules are complete.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 8. Configuration

All settings are stored in `configs/config.yaml`, so no path or parameter is written directly in the code.

| Key | Meaning | Current value |
|:---|:---|:---|
| `data.train_path` | Training file | `Dataset/train.csv` |
| `data.test_path` | Test file | `Dataset/test.csv` |
| `data.target` | Column to predict | `satisfaction` |
| `validation.test_size` | Share of the training data used for validation | `0.2` |
| `validation.random_state` | Seed that makes the split reproducible | `42` |
| `model.save_path` | Where the trained model is saved | `models/best_model.pkl` |

To change a setting, edit this file and run `python main.py` again.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 9. Results

Results will be added after the evaluation step: validation scores of each model, effect of the preprocessing choices, final test performance and the most important features.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 10. Project Status

| Component | Status |
|:---|:---|
| Dataset selection and inspection | ![Done](https://img.shields.io/badge/Done-B0CEB1?style=flat-square) |
| Exploratory data analysis | ![Done](https://img.shields.io/badge/Done-B0CEB1?style=flat-square) |
| Repository structure and configuration | ![Done](https://img.shields.io/badge/Done-B0CEB1?style=flat-square) |
| Data loading and preprocessing | ![In progress](https://img.shields.io/badge/In%20progress-926749?style=flat-square) |
| Model training and comparison | ![Planned](https://img.shields.io/badge/Planned-80999A?style=flat-square) |
| Evaluation and interpretation | ![Planned](https://img.shields.io/badge/Planned-80999A?style=flat-square) |
| Training and inference scripts | ![Planned](https://img.shields.io/badge/Planned-80999A?style=flat-square) |
| Final report and presentation | ![Planned](https://img.shields.io/badge/Planned-80999A?style=flat-square) |

<div align="right"><a href="#top">Back to top ↑</a></div>

## 11. Team

**Group 17 — Université Internationale de Rabat**

| Member | Main contribution |
|:---|:---|
| **Hiba Asmaa El Atifi** | Exploratory analysis, project integration and documentation |
| **Salma Alami** | Data loading and preprocessing |
| **Hayam Mimouni** | Model training and hyperparameter selection |
| **Khaoula Marhrani** | Evaluation and interpretation |

All members contribute to the final report and presentation.

---

<div align="center">
<sub>Integrated Machine Learning Project · Université Internationale de Rabat · 2026–2027 · Prof. Youness Moukafih</sub>
</div>
