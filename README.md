# Turbofan Engine Remaining Useful Life (RUL) Prediction using Machine Learning and Deep Learning

## Introduction

Predictive Maintenance has become one of the most important applications of Artificial Intelligence in modern industry. Instead of waiting for equipment to fail, predictive maintenance systems analyze historical operational data and sensor readings to estimate when a machine is likely to fail.

This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan aircraft engines using the NASA CMAPSS FD004 dataset. The objective is to estimate how many operational cycles remain before an engine reaches failure conditions. Accurate RUL prediction enables maintenance teams to schedule repairs proactively, reduce unexpected downtime, improve safety, and lower maintenance costs.

To achieve this goal, several Machine Learning and Deep Learning models were developed, trained, evaluated, and compared. The best-performing model was then deployed through a Flask REST API for real-time inference.

---

# Problem Statement

Aircraft engines generate large amounts of sensor data during operation. Over time, engine components degrade, causing subtle changes in sensor measurements.

The challenge is to learn the relationship between sensor behavior and engine degradation in order to predict:

> How many cycles remain before the engine reaches failure?

This is formulated as a supervised regression problem where:

- **Input:** Sensor readings + Operational settings
- **Output:** Remaining Useful Life (RUL)

---

## Dataset Source

This project uses the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset for Remaining Useful Life (RUL) prediction.

Official NASA Dataset:

https://data.nasa.gov/docs/legacy/CMAPSSData.zip

The FD004 subset was used in this project because it contains multiple operating conditions and multiple fault modes, making it one of the most challenging benchmark datasets for predictive maintenance and RUL estimation.

# Dataset Description

The project uses the NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset.

The FD004 subset was selected because it is considered the most challenging scenario due to:

- Multiple operating conditions
- Multiple fault modes
- Complex degradation patterns

Each record contains:

## Engine Information

- Unit Number
- Time in Cycles

## Operational Settings

- Operational Setting 1
- Operational Setting 2
- Operational Setting 3

## Sensor Measurements

Twenty-one sensor readings describing the engine's operating state and health condition.

---

# Target Variable Generation

The original dataset does not directly provide RUL values for training samples.

Therefore, RUL was calculated using:

```python
RUL = Maximum Cycle of Engine - Current Cycle
```

This means:

- Engines near the beginning of their life have high RUL values.
- Engines near failure have low RUL values.

This generated target variable is then used during model training.

---

# Exploratory Data Analysis (EDA)

Several exploratory analyses were performed to better understand the dataset.

## Engine Lifetime Analysis

The following statistics were computed:

- Number of engines
- Minimum engine lifetime
- Maximum engine lifetime
- Average engine lifetime

## Data Quality Checks

- Missing value analysis
- Duplicate row detection
- Feature distribution analysis

## Sensor Behavior Analysis

Sensor trends were analyzed across engine life cycles to identify measurements strongly correlated with degradation.

The analysis showed that several sensors exhibit clear degradation patterns and therefore provide valuable information for RUL prediction.

---

# Feature Engineering

Feature engineering plays a critical role in predictive maintenance applications.

## Selected Features

The most informative operational settings and sensor measurements were selected:

- operational_setting_1
- sensor_2
- sensor_3
- sensor_4
- sensor_6
- sensor_7
- sensor_8
- sensor_9
- sensor_11
- sensor_12
- sensor_13
- sensor_14
- sensor_15
- sensor_20
- sensor_21

## Lag Features

To capture temporal dependencies and degradation history, lag-based features were created.

Examples:

- sensor_13_lag1
- sensor_11_lag1
- sensor_15_lag1

These features represent sensor values from the previous cycle and provide the model with short-term historical context.

### Total Features

```text
15 Original Features
+ 3 Lag Features
= 18 Features
```

---

# Data Preprocessing

## Log Transformation

The RUL target was transformed using:

```python
np.log1p(RUL)
```

Benefits:

- Reduces skewness
- Stabilizes variance
- Improves model learning

Predictions were transformed back using:

```python
np.expm1(predictions)
```

## Feature Scaling

### StandardScaler

Applied to:

- Random Forest
- XGBoost
- SVR
- LightGBM
- LSTM

### MinMaxScaler

Applied to:

- CNN + BiGRU + Attention

Scaling ensures numerical stability and faster convergence during training.

---

# Validation Strategy

A Unit-wise Cross Validation approach was adopted.

Instead of randomly splitting rows, complete engines were separated into different folds.

Advantages:

- Prevents information leakage
- Better simulates real-world deployment
- Produces more realistic performance estimates

Five-fold validation was used to evaluate model robustness.

---

# Machine Learning Models

## Random Forest Regressor

Random Forest combines multiple decision trees to improve generalization and reduce overfitting.

### Parameters

```python
n_estimators=300
max_depth=15
min_samples_split=5
min_samples_leaf=4
max_features='sqrt'
```

Advantages:

- Handles nonlinear relationships
- Robust against noise
- Strong baseline performance

---

## XGBoost Regressor

### Parameters

```python
n_estimators=400
max_depth=8
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
```

Advantages:

- Excellent handling of complex interactions
- Strong regularization
- High predictive accuracy

---

## Support Vector Regressor (SVR)

### Parameters

```python
kernel='rbf'
C=100
epsilon=0.1
```

Advantages:

- Effective on nonlinear datasets
- Strong mathematical foundation

---

## LightGBM Regressor

### Parameters

```python
n_estimators=500
max_depth=8
learning_rate=0.05
```

Advantages:

- Fast training
- Efficient memory usage
- High scalability

---

# Deep Learning Models

## LSTM Model

Long Short-Term Memory (LSTM) networks are specifically designed for sequential and time-series data.

### Architecture

```text
Input
↓
LSTM (128)
↓
Dropout (0.2)
↓
LSTM (64)
↓
Dense (32)
↓
Output
```

### Sequence Length

```text
30 Cycles
```

The model learns long-term degradation patterns directly from historical sensor sequences.

---

## CNN + BiGRU + Attention Model

This architecture combines three powerful components:

### CNN Layer

Extracts local degradation patterns from sensor sequences.

### Bidirectional GRU

Captures temporal dependencies in both forward and backward directions.

### Attention Mechanism

Allows the model to focus on the most important parts of the sequence.

### Architecture

```text
Input
↓
Conv1D
↓
MaxPooling1D
↓
Conv1D
↓
BiGRU
↓
Attention
↓
GlobalAveragePooling
↓
Dense Output
```

---

# Evaluation Metrics

The models were evaluated using:

## Mean Absolute Error (MAE)

Measures the average prediction error.

Lower values indicate better performance.

## Root Mean Squared Error (RMSE)

Penalizes larger prediction errors.

Lower values indicate better performance.

## R² Score

Measures how much variance in the target variable is explained by the model.

Higher values indicate better performance.

---

# Model Performance

| Model | Test R² | Test MAE | Test RMSE |
|---------|---------|---------|---------|
| Random Forest | 0.6227 | 24.47 | 33.49 |
| XGBoost | 0.6023 | 24.91 | 34.39 |
| SVR | 0.5079 | 27.64 | 38.25 |
| LightGBM | 0.5970 | 24.98 | 34.61 |
| LSTM | 0.5350 | 28.27 | 37.19 |
| CNN + BiGRU + Attention | 0.4330 | 32.01 | 40.08 |

## Best Performing Model

🏆 **Random Forest Regressor**

- Test R² = 0.6227
- Test MAE = 24.47
- Test RMSE = 33.49

Due to its superior performance, Random Forest was selected for deployment.

---

# Model Deployment

The trained Random Forest model was exported and deployed using Flask.

## Saved Files

```text
random_forest_model_05.pkl
scaler_1.pkl
```

## Deployment Workflow

1. Receive sensor measurements through a POST request.
2. Convert input data into NumPy arrays.
3. Apply feature scaling using the saved scaler.
4. Generate predictions using the trained Random Forest model.
5. Return predicted RUL as a JSON response.

---

# API Endpoints

## Home Endpoint

### Request

```http
GET /
```

### Response

```text
FD004 Random Forest model is live!
```

---

## Prediction Endpoint

### Request

```http
POST /predict
```

### Example JSON

```json
{
  "features": [
    0.12,
    0.34,
    0.56,
    0.78,
    0.91,
    1.11,
    1.22,
    1.33,
    1.44,
    1.55,
    1.66,
    1.77,
    1.88,
    1.99,
    2.11,
    2.22,
    2.33,
    2.44
  ]
}
```

### Example Response

```json
{
  "predicted_RUL": 84.37
}
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Turbofan-RUL-Prediction.git
```

Move into the project directory:

```bash
cd Turbofan-RUL-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

The API will start at:

```text
http://127.0.0.1:5000
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- TensorFlow
- Keras
- Flask
- Matplotlib
- Seaborn
- Joblib

---

# Future Improvements

Potential enhancements include:

- Hyperparameter optimization using Optuna
- Ensemble stacking models
- Transformer-based architectures
- Explainable AI (SHAP, LIME)
- Docker containerization
- Cloud deployment (AWS / Azure)
- Real-time streaming predictions
- MLOps pipeline integration

---

# Conclusion

This project presents a complete end-to-end Predictive Maintenance pipeline for turbofan engine Remaining Useful Life prediction.

Starting from raw sensor data, the workflow includes data preprocessing, feature engineering, machine learning experimentation, deep learning modeling, evaluation, and deployment through a Flask API.

The final deployed Random Forest model demonstrates strong predictive capability and provides a practical foundation for industrial predictive maintenance applications.
