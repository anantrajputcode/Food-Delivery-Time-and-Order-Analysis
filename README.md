# DeliverIQ — Food Delivery Time Prediction

> **Predict smarter. Deliver faster.**

DeliverIQ is a machine learning web application that estimates how long a food order will take to be delivered. It combines a trained regression model with an interactive Streamlit interface, so anyone can enter order details and get an estimated delivery time in minutes.

**Author:** Anant Rajput

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Features](#features)
4. [Dataset](#dataset)
5. [Machine Learning Approach](#machine-learning-approach)
6. [Model Performance](#model-performance)
7. [Technology Stack](#technology-stack)
8. [Project Structure](#project-structure)
9. [Installation](#installation)
10. [Running the Application](#running-the-application)
11. [How Predictions Work](#how-predictions-work)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)
14. [Author](#author)

---

## Overview

Delivery time depends on many factors: how far the order has to travel, how busy the roads are, the weather, the delivery partner, and the time of day. This project learns those patterns from historical delivery records and uses them to estimate the delivery time of a new order.

The estimate is shown in a clean dashboard with a result card, supporting visualizations, and an explanation of how the model works.

## Problem Statement

Given details about an order, the delivery partner, and the conditions on the road, predict the time in minutes the delivery will take.

This is a **supervised learning** problem and, because the target is a continuous number, a **regression** problem.

## Features

- **Prediction page:** a grouped input form (delivery, delivery partner, traffic and environment, order) that returns an estimated delivery time with a short comparison to historical deliveries.
- **Analytics page:** visualizations from the exploratory data analysis, loaded from the `visualizations/` folder.
- **Model Performance page:** MAE, RMSE and R², plain-language explanations, and a feature importance chart.
- **About the Model page:** an explanation of how a Random Forest works, what each input feature means, and the model's limitations.
- **Technology Stack page:** the tools used in the project and the role of each.
- **Dropdowns built from data:** category options are read from the dataset rather than hard-coded, so the model is never sent an unknown category.
- **Error handling:** clear messages for a missing model, a missing dataset, invalid input, or a failed prediction.
- **Cached loading:** the model and dataset are cached so the app stays responsive.

## Dataset

| Property | Value |
|---|---|
| Records | 45,584 |
| Columns | 20 |
| Target | `Time_taken (min)` — actual delivery duration in minutes |
| Raw file | `dataset/food_delivery.csv` |
| Cleaned file | `dataset/cleaned_food_delivery.csv` |

The raw dataset includes delivery partner details, restaurant and delivery coordinates, order date and times, weather, traffic, vehicle information, order type, festival and city category.

The data cleaning and feature engineering steps are documented in `notebooks/food_delivery_analysis.ipynb`. The original dataset is not modified by the application.

## Machine Learning Approach

### Model inputs

The model uses 12 features:

| Feature | Description |
|---|---|
| `distance_km` | Distance between restaurant and customer, engineered from coordinates |
| `order_hour` | Hour of day the order was placed (0–23) |
| `Delivery_person_Age` | Age of the delivery partner |
| `Delivery_person_Ratings` | Rating of the delivery partner (about 1–5) |
| `Road_traffic_density` | Road congestion level |
| `Weather_conditions` | Weather during the delivery |
| `Type_of_order` | Category of food ordered |
| `Type_of_vehicle` | Vehicle used for the delivery |
| `Vehicle_condition` | Vehicle condition code as recorded in the dataset |
| `multiple_deliveries` | Number of deliveries handled together |
| `Festival` | Whether the delivery took place during a festival |
| `City` | City category |

### Algorithm

The saved model is a **Random Forest Regressor** built with Scikit-learn. A Random Forest:

1. Trains many decision trees, each on a random sample of the training data.
2. Lets every tree predict the delivery time on its own.
3. Averages the predictions of all trees into one final estimate.

Averaging many trees reduces the chance of overfitting to any single pattern in the data.

The trained model is stored at `models/best_delivery_model.pkl` and loaded with Joblib. It is **not** retrained by the application.

## Model Performance

Results previously reported on the held-out test set (see `models/model_evaluation.txt`):

| Metric | Value | Meaning |
|---|---:|---|
| MAE | 3.22 min | On average, predictions differ from the actual delivery time by about 3.22 minutes |
| RMSE | 4.09 min | Square root of the average squared error; penalizes large mistakes more heavily |
| R² | 0.8107 | The model explains about 81.07% of the variance in delivery time, relative to always predicting the mean |

> **Note:** R² is not accuracy. A value of 0.8107 does **not** mean 81% of predictions are correct.

## Technology Stack

| Technology | Purpose in this project |
|---|---|
| Python | Core programming language |
| Pandas | Loading, cleaning and manipulating data |
| NumPy | Numerical computation |
| Matplotlib | Charts in the analysis notebook |
| Seaborn | Statistical visualizations in the notebook |
| Scikit-learn | Model training, preprocessing and evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | The web application |
| Plotly | Interactive charts in the application |
| Git / GitHub | Version control and hosting |
| Git LFS | Storing the large model file (about 104 MB) |

## Project Structure

```
Food-Delivery-Time-and-Order-Analysis/
├── app.py                         # Streamlit application (DeliverIQ)
├── README.md
├── requirements.txt
├── dataset/
│   ├── food_delivery.csv          # Original dataset
│   └── cleaned_food_delivery.csv  # Cleaned dataset
├── models/
│   ├── best_delivery_model.pkl    # Trained model (tracked with Git LFS)
│   ├── model_metadata.json
│   └── model_evaluation.txt
├── notebooks/
│   └── food_delivery_analysis.ipynb
├── visualizations/                # Charts saved from the notebook
└── reports/
```

## Installation

**Prerequisites:** Python 3, Git, and [Git LFS](https://git-lfs.com/) (the model file is stored with Git LFS).

```bash
git lfs install
git clone <repository-url>
cd Food-Delivery-Time-and-Order-Analysis
git lfs pull
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (Command Prompt):

```bat
.venv\Scripts\activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> **Important:** a scikit-learn model should be loaded with the same scikit-learn version it was trained with. If loading fails, check the version you installed against the one used for training.

## Running the Application

```bash
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## How Predictions Work

1. You enter the order details in the form.
2. The app validates the inputs and builds a one-row DataFrame with the model's 12 features, in the order the model expects.
3. The cached model produces a prediction with `model.predict(sample)[0]`.
4. The result is shown in minutes, together with how it compares to the median of historical deliveries in the dataset.

The prediction is an **estimate**, not a guaranteed delivery time. The app does not display a confidence percentage, because a Random Forest does not produce a calibrated one by default.

## Limitations

- Traffic can change unexpectedly after an order is placed.
- Restaurant preparation time is not a direct input to the model.
- The distance feature is derived from coordinates and is a straight-line measure, so real road distance can be longer.
- Weather and order demand can shift during a delivery.
- Predictions reflect historical patterns and may not generalize to unseen conditions.
- Actual-vs-predicted and residual charts are only shown if test-set predictions are saved to `reports/test_predictions.csv` (columns: `actual`, `predicted`).

## Future Improvements

The following are ideas for future work and are **not** implemented in the current version:

- Real road-distance calculation through a routing API
- Restaurant preparation time as an input
- Live traffic and weather data
- Better encoding of time of day
- Comparison with other models such as XGBoost or CatBoost
- Hyperparameter tuning and cross-validation
- Prediction intervals
- Model monitoring
- Cloud deployment
- A REST API with FastAPI

## Author

**Anant Rajput**