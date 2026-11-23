<a id="top"></a>

<div align="center">

<img src="assets/banner.svg" alt="Airline Passenger Satisfaction" width="100%">

<br>

![Python](https://img.shields.io/badge/Python-3.10+-344035?style=flat-square&logo=python&logoColor=CFD5BB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-80999A?style=flat-square&logo=scikitlearn&logoColor=white)
![Task](https://img.shields.io/badge/Task-binary%20classification-B0CEB1?style=flat-square&labelColor=344035)
![Course](https://img.shields.io/badge/UIR-2026–2027-926749?style=flat-square&labelColor=344035)

**Predicting whether an airline passenger is satisfied, and understanding which factors matter most.**


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

We use the [Airline Passenger Satisfaction dataset — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction), which contains survey responses from 129,880 passengers.

| File | Passengers | Used for |
|:---|---:|:---|
| `Dataset/train.csv` | 103,904 | Training and validation |
| `Dataset/test.csv` | 25,976 | Final evaluation only |
| **Total** | **129,880** | |

Each row represents a passenger record, with **22 input features** and one target variable, `satisfaction`.

| Passenger profile | Flight information | Service ratings (0–5) |
|:---|:---|:---|
| Gender | Type of Travel | Online Boarding |
| Customer Type | Class | Seat Comfort |
| Age | Flight Distance | Inflight Entertainment |
| | Departure Delay | Inflight Wi-Fi |
| | Arrival Delay | 10 additional services |

**A few important details about the dataset:**

- **Target classes:** neutral or dissatisfied (56.7%) and satisfied (43.3%) in the training set.
- **Zero ratings:** a value of 0 may indicate *Not Applicable* rather than a negative evaluation.
- **Identifiers:** `id` and `Unnamed: 0` are excluded from the model inputs.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 3. Methodology

Our approach follows four main steps, from understanding the dataset to evaluating the final predictions.

| Step | What we do | Where |
|:---|:---|:---|
| **01. Exploratory Data Analysis** | Explore data quality, distributions, class balance, service ratings and their relationship with satisfaction. | `notebooks/` |
| **02. Data Preprocessing** | Handle missing values, encode categorical features, scale numerical variables and investigate zero ratings. | `src/preprocessing.py` |
| **03. Model Training** | Train classification models and compare hyperparameters using validation data. | `src/model.py`, `src/train.py` |
| **04. Evaluation & Interpretation** | Compare model performance, examine prediction errors and identify influential features. | `src/evaluate.py` |

One important part of our approach is keeping the **training, validation and test data separate**. Preprocessing parameters are learned from the training data, while the validation set helps us compare different approaches. The test set is kept aside for the final evaluation.

### Which models will we use?

We plan to work with the classical classification algorithms studied in our Machine Learning course.

| Model | What we will explore |
|:---|:---|
| K-Nearest Neighbours (KNN) | Number of neighbours (K) |
| Logistic Regression | Regularisation strength (C) |
| Linear Discriminant Analysis (LDA) | Standard model |
| Quadratic Discriminant Analysis (QDA) | Standard model |
| Decision Tree | Maximum tree depth |
| Random Forest | Number of trees |

We will compare their performance using the same validation approach and document the settings used for each model.

### Does preprocessing make a difference?

This is one of the main questions behind our project.

Instead of simply preparing the data and moving on, we want to understand whether some preprocessing decisions actually improve the predictions.

We plan to investigate two comparisons:

| Experiment | Model | What we compare |
|:---|:---|:---|
| **Feature scaling** | KNN | Predictions with and without standardisation |
| **Zero ratings** | Logistic Regression | Keeping zeros as values versus adding zero-indicator features |

In each experiment, we will keep the other settings unchanged so that we can better understand the effect of the preprocessing choice.

These comparisons will help us justify our final pipeline rather than selecting a method without testing it.

<div align="right"><a href="#top">Back to top ↑</a></div>

## 4. Key Findings from the EDA

Before training our models, we explored the dataset to better understand the passenger experience and identify the main points to consider during preprocessing.

A few observations stood out:

| What we found | Why it matters |
|:---|:---|
| **Travel class:** around 69% of Business class passengers are satisfied, compared with 19% in Economy. | Travel class may be an important feature for predicting satisfaction. |
| **Service experience:** online boarding, inflight entertainment and seat comfort show some of the largest rating differences between the two satisfaction groups. | These services are worth examining when interpreting our models. |
| **Zero ratings:** some services contain zero-valued ratings, including inflight Wi-Fi. | Rather than automatically replacing zeros with an average, we will investigate whether keeping them or adding indicator features produces better results. |
| **Flight delays:** departure and arrival delays are strongly correlated (r = 0.965), and 310 arrival-delay values are missing from the training set. | We need to handle missing values and consider the relationship between these two features during preprocessing. |
| **Duplicate records:** no duplicate rows were found. | No duplicate removal is needed based on our initial checks. |

These findings give us a starting point for the next stages of the project. They help us decide which preprocessing methods to investigate and which features deserve closer attention when evaluating the models.

The complete analysis, including visualisations and Python code, is available in [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb).

<div align="right"><a href="#top">Back to top ↑</a></div>

## 5. Repository Structure

We have organised the project into separate modules to keep the code clear, reusable and easy to maintain. Each file has a specific responsibility, making it easier to understand the different stages of our machine learning workflow.

```text
airline-passenger-satisfaction/
│
├── assets/
│   └── banner.svg
│
├── configs/
│   └── config.yaml
│
├── Dataset/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   └── best_model.pkl
│
├── main.py
├── test.py
├── requirements.txt
└── README.md
```

### How the files work together

| File | Responsibility |
|:---|:---|
| `main.py` | Main entry point that will connect the different modules, run training and save the selected model. |
| `test.py` | Will load the saved model to make predictions without retraining. |
| `configs/config.yaml` | Stores dataset paths and project settings. |
| `src/data.py` | Loads the datasets and prepares the training and validation split. |
| `src/preprocessing.py` | Handles missing values, categorical encoding, feature scaling and zero-rating representation. |
| `src/model.py` | Defines the classification algorithms used in the project. |
| `src/train.py` | Handles model training and hyperparameter selection. |
| `src/evaluate.py` | Calculates evaluation metrics and confusion matrices. |
| `src/utils.py` | Provides reusable functions for configuration and model saving/loading. |

The idea is to keep each part of the project independent while allowing the modules to work together as one complete pipeline.

*Some files are still under development. Currently, `main.py` loads the project configuration; the training modules, independent inference script and saved model will be integrated progressively.*

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
