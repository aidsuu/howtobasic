# Converting SVR Data to QSVR: A Step-by-Step Guide

In this tutorial, we will walk through the process of transforming data from classical Support Vector Regression (SVR) into Quantum Support Vector Regression (QSVR) using Qiskit. We'll start with the necessary setup, load the data, apply classical SVR, and finally transition to QSVR to compare the results.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting up the Environment](#setting-up-the-environment)
4. [Loading and Preprocessing Data](#loading-and-preprocessing-data)
5. [Building and Tuning the SVR Model](#building-and-tuning-the-svr-model)
6. [Quantum Kernel and QSVR](#quantum-kernel-and-qsvr)
7. [Model Evaluation and Comparison](#model-evaluation-and-comparison)
8. [Conclusion](#conclusion)
9. [References](#references)

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
import seaborn as sns
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

df = df.rename(columns={'Unnamed: 0': 'Tanggal', 'Unnamed: 1': 'RR'})[['Tanggal', 'RR']]
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
df.dropna(subset=['Tanggal'], inplace=True)
df['RR'] = df['RR'].replace(8888.0, np.nan).interpolate(method='linear').fillna(0)

# Lag features and month feature
n_lags = 3
features_list = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['month']
for i in range(1, n_lags + 1):
    df[f'lag_{i}'] = df['RR'].shift(i)
df['month'] = df['Tanggal'].dt.month
df_model = df.dropna().copy()

# Feature and target variable
X = df_model[features_list]
y = df_model['RR']

# Standardization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
```
