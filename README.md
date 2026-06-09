# Tesla_Stock_Forecast_Engine
## 🚗 TSLA Stock Price Forecasting using LSTM
Developer6316 – Deep Learning for Financial Time Series
---
## 📌 Overview
This project implements a Long Short-Term Memory (LSTM) neural network to forecast Tesla (TSLA) stock prices based on historical Open price data. The model captures temporal dependencies in stock market trends and provides future price predictions with measurable accuracy.

Built as part of Developer6316's portfolio in applied deep learning and quantitative finance.

## 🎯 Objective
To demonstrate the capability of recurrent neural networks (specifically LSTM) in modeling non-linear, sequential financial data and generating meaningful short-term forecasts for Tesla stock.

## 📊 Dataset
Source: Tesla stock data from 2010 to 2025 (tesla_stock_data_2010_2025.csv)
Feature used: Open price (daily)
Split: 75% training, 25% testing
Time steps: 60 days look-back window

## 🧠 Model Architecture
LSTM(50, return_sequences=True)  
LSTM(64, return_sequences=False)  
Dense(32)  
Dense(16)  
Dense(1)
Optimizer: Adam

Loss function: Mean Squared Error (MSE)

Regularization: Early stopping (patience=50)

📈 Results
RMSE: ~3.64 USD (on test set)

MAE: ~2.81 USD

The model captures general price trends and directional movements, though actual peaks/troughs may vary due to market volatility.

## 🛠️ Technologies Used
Python 3.10+
TensorFlow / Keras,
Pandas, NumPy,
Scikit-learn,
Matplotlib, Seaborn,
yFinance (optional for live data)

## 🚀 How to Run
### Clone the repo:

```bash
git clone https://github.com/Developer6316/Tesla_Stock_Forecast_Engine.git
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
Run the script:
```bash
python tesla_lstm_forecast.py
```
## 📌 Future Improvements
📌Incorporate additional features (Volume, High, Low, Close)

📌Hyperparameter tuning with GridSearchCV

📌Deploy as a Streamlit web app

📌Add live data fetching from Yahoo Finance

