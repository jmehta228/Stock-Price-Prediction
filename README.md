# Stock-Price-Prediction

This repository contains a Jupyter Notebook for predicting stock prices using both **classical machine learning** and **deep learning** approaches. The notebook walks through data collection, feature engineering, model training, evaluation, and visualization.

## Notebook Overview

- **File analyzed**: `Stock-Price-Pred.ipynb`
- **Total cells**: 15 (Code: 15, Markdown: 0)

### Key Features
- Fetches historical stock data (e.g., Apple) using **Yahoo Finance** (`yfinance`) with fallback options.
- Data preprocessing, cleaning, and caching.
- Feature engineering with **lag variables** and rolling statistics.
- Implements baseline models:
  - **Naive forecast** (today’s price = yesterday’s price)
  - **Linear Regression** with lag features
  - **LSTM (Long Short-Term Memory)** deep learning model
- Evaluation metrics: MAE, RMSE, and MAPE.
- Visualization of predicted vs actual stock prices, and loss curves.

### Main Functions in the Notebook
- `_clean_ohlcv()`
- `_write()`
- `dropna_align()`
- `fetch_from_stooq()`
- `fetch_from_yahoo()`
- `inverse_close()`
- `load_or_fetch_cache()`
- `make_lag_features()`
- `metrics()`
- `naive_forecast()`
- `rolling_backtest_linear()`
- `time_split()`

### Libraries Used
```python
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, callbacks, Model
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import yfinance as yf
```

## How It Works

1. **Data Collection**
   - Uses `yfinance` to download OHLCV stock data.
   - Falls back to Stooq if Yahoo fails.
   - Data is cached locally for faster re-runs.

2. **Preprocessing**
   - Handles missing values and ensures numeric columns.
   - Splits dataset into **train, validation, and test** either by ratio or date.

3. **Feature Engineering**
   - Creates lag features (`close_lag_1`, `close_lag_2`, etc.).
   - Computes rolling mean and standard deviation windows.

4. **Models**
   - **Naive baseline**: previous day’s close.
   - **Linear Regression**: fits engineered lag features.
   - **LSTM**: sequence model using 60-day lookback with dropout regularization.

5. **Evaluation**
   - Metrics: MAE, RMSE, MAPE.
   - Plots: actual vs predicted prices, and training/validation loss curves.

6. **Artifacts**
   - Trained Linear Regression and LSTM models are saved (`.pkl`, `.keras`).
   - Scalers are persisted for reproducibility.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (or individually: `numpy pandas matplotlib scikit-learn tensorflow yfinance`)

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Stock-Price-Pred.ipynb
   ```

3. Run all cells top to bottom.

## Results

- The **Naive baseline** often performs surprisingly well.
- **Linear Regression** with lag features achieves slightly improved or similar results.
- **LSTM** may outperform on validation but can overfit, so test performance is carefully compared.

## Next Steps / Extensions

- Try more advanced architectures: GRU, stacked LSTMs, Transformers.
- Incorporate more features (technical indicators, macroeconomic variables).
- Deploy as a dashboard using **Streamlit** or **Dash**.
- Extend to multi-step forecasting (predicting several days ahead).

---
*Generated README based on notebook analysis.*
