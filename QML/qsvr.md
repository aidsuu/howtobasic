# Converting SVR Data to QSVR: A Step-by-Step Guide 🤓

In this tutorial, we will walk through the process of transforming data from classical Support Vector Regression (SVR) into Quantum Support Vector Regression (QSVR) using Qiskit. We'll start with the necessary setup, load the data, apply classical SVR, and finally transition to QSVR to compare the results.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting up the Environment](#setting-up-the-environment)
4. [Loading and Preprocessing Data](#loading-and-preprocessing-data)
5. [Training and Test Data Sharing](#training-and-test-data-sharing)
6. [Building and Tuning the SVR Model](#building-and-tuning-the-svr-model)
7. [Train the SVR model with the best parameters](#train-the-svr-model-with-the-best-parameters)
8. [Quantum Kernel and QSVR](#quantum-kernel-and-qsvr)
9. [Model Evaluation and Comparison](#model-evaluation-and-comparison)
10. [Conclusion](#conclusion)
11. [References](#references)

## Introduction
This tutorial demonstrates how to use classical machine learning techniques, such as Support Vector Regression (SVR), alongside quantum computing with Qiskit to build a Quantum Support Vector Regression (QSVR) model. We will compare the classical model’s performance with the quantum-enhanced model.

## Prerequisites
Before beginning, ensure that you have the following:
- Python 3.x installed.
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `qiskit`, and `qiskit_machine_learning`.
- A basic understanding of Support Vector Regression and quantum computing concepts.

To install the necessary libraries, run:

```bash
pip install pandas numpy scikit-learn matplotlib qiskit qiskit-machine-learning
```

## Setting up the Environment
Start by importing the necessary libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
```
## Loading and Preprocessing Data
For this tutorial, we will use weather data (rainfall data). Here’s how to load and preprocess the data:
```python
# Input Data    
NAMA_FILE = 'datacurahhujan.xlsx'
df = pd.read_excel(NAMA_FILE, skiprows=6)

# Rename columns for clarity and select relevant columns (Date and Rainfall)
df = df.rename(columns={'Unnamed: 0': 'Tanggal', 'Unnamed: 1': 'RR'})[['Tanggal', 'RR']]
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
df.dropna(subset=['Tanggal'], inplace=True)
df['RR'] = df['RR'].replace(8888.0, np.nan).interpolate(method='linear').fillna(0) # Replace invalid values and interpolate missing data

# Add lag features for previous day's rainfall (e.g., lag_1, lag_2, lag_3) and month
n_lags = 3
features_list = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['month']
for i in range(1, n_lags + 1):
    df[f'lag_{i}'] = df['RR'].shift(i)
df['month'] = df['Tanggal'].dt.month
df_model = df.dropna().copy()

# Separate the features (X) and target variable (y)
X = df_model[features_list]
y = df_model['RR']

# Standardize the features and target variable
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
```

## Training and Test Data Sharing
The data is divided into training data (80%) and test data (20%).
```python
# Split the data into training (80%) and testing (20%) sets
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
test_dates = df_model['Tanggal'].iloc[train_size:]
```

## Building and Tuning the SVR Model
In this step, we build and tune the classical SVR model using GridSearchCV:
```python
# GridSearch for best parameters for SVR
param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.01], 'epsilon': [0.01, 0.1]}
svr = SVR(kernel='rbf')
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_params_classical = grid_search.best_params_
```

## Train the SVR model with the best parameters
```python
svr_model = grid_search.best_estimator_
y_pred_scaled = svr_model.predict(X_test)  # Predict using the SVR model
y_pred_classical = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel() 
rmse_classical = np.sqrt(mean_squared_error(y_test_original, y_pred_classical)) 
print(f"RMSE for Classical SVR: {rmse_classical:.4f} mm")
```

## Quantum Kernel and QSVR
Here we create the quantum kernel using Qiskit and use it for QSVR:
```python
# Quantum Kernel Setup
num_features = len(features_list) 
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')  # Create quantum feature map
sampler = Sampler()
fidelity_algorithm = ComputeUncompute(sampler=sampler)
fidelity_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity_algorithm)

# Compute quantum kernel matrices
kernel_matrix_train = fidelity_kernel.evaluate(x_vec=X_train)
kernel_matrix_test = fidelity_kernel.evaluate(x_vec=X_test, y_vec=X_train)
```

## Model Evaluation and Comparison
We will now compare the classical and quantum models by plotting the predictions and calculating the RMSE:
```python
# Classical SVR
svr_model = grid_search.best_estimator_
y_pred_scaled = svr_model.predict(X_test)
y_pred_classical = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
rmse_classical = np.sqrt(mean_squared_error(y_test_original, y_pred_classical))

# QSVR Model
svr_quantum = SVR(kernel='precomputed', C=C_best, epsilon=epsilon_best)
svr_quantum.fit(kernel_matrix_train, y_train)

y_pred_scaled_q = svr_quantum.predict(kernel_matrix_test)
y_pred_quantum = scaler_y.inverse_transform(y_pred_scaled_q.reshape(-1, 1)).ravel()
rmse_quantum = np.sqrt(mean_squared_error(y_test_original, y_pred_quantum))
print(f"RMSE for quantum SVR: {rmse_quantum:.4f} mm")

# Plotting the results
plt.figure(figsize=(15, 8))
plt.plot(test_dates, y_test_original, label='Data Aktual', marker='o', linestyle='-', color='black', linewidth=2)
plt.plot(test_dates, y_pred_classical, label=f'SVR Klasik (RMSE: {rmse_classical:.2f})', linestyle='--', color='red')
plt.plot(test_dates, y_pred_quantum, label=f'QSVR (RMSE: {rmse_quantum:.2f})', linestyle='-.', color='green', linewidth=2.5)
plt.title('Perbandingan Prediksi: SVR Klasik vs. Quantum Kernel SVR', fontsize=18)
plt.xlabel('Tanggal', fontsize=14); plt.ylabel('Curah Hujan (mm)', fontsize=14)
plt.legend(fontsize=12); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout()
plt.show()
```

## Conclusion
In this tutorial, we have demonstrated how to use classical SVR and transform it into QSVR using quantum kernels. We have compared the results of the classical and quantum models based on RMSE and visualized the predictions.

## References
[Qiskit](https://quantum.cloud.ibm.com/docs/en) \\
[Scikit-learn](https://scikit-learn.org/stable/) \\
[Pandas](https://pandas.pydata.org/docs/user_guide/index.html)
