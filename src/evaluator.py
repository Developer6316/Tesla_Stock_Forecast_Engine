# -*- coding: utf-8 -*-
"""
Model evaluation and prediction module for Tesla stock prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_predictions(model, x_test, y_test, scaler):
    """
    Generate predictions and calculate metrics.
    
    Args:
        model (keras.Model): Trained model
        x_test (np.ndarray): Test sequences
        y_test (np.ndarray): Test targets
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler for inverse transform
    
    Returns:
        tuple: (predictions, y_test_actual, rmse, mae)
    """
    # Predict
    predictions_scaled = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    # Inverse transform actual values
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)
    
    print(f"\n{'='*50}")
    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Test MAE:  ${mae:.2f}")
    print(f"{'='*50}\n")
    
    return predictions, y_test_actual, rmse, mae


def plot_predictions(df, train_size, look_back, predictions):
    """
    Visualize predictions vs actual prices.
    
    Args:
        df (pd.DataFrame): Stock data
        train_size (int): Number of training samples
        look_back (int): Lookback window size
        predictions (np.ndarray): Model predictions
    """
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    test_df = test_df.iloc[look_back:]
    test_df["Predictions"] = predictions
    
    plt.figure(figsize=(16, 6))
    plt.title("Tesla (TSLA) Stock Price Prediction (LSTM)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Open Price (USD)", fontsize=12)
    plt.plot(train_df["Open"], linewidth=2, label="Train", alpha=0.8)
    plt.plot(test_df["Open"], linewidth=2, label="Actual Test", alpha=0.8)
    plt.plot(test_df["Predictions"], linewidth=2, label="Predictions", alpha=0.8)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_prediction_metrics_summary(y_test_actual, predictions):
    """
    Get detailed prediction metrics summary.
    
    Args:
        y_test_actual (np.ndarray): Actual test values
        predictions (np.ndarray): Predicted values
    
    Returns:
        dict: Dictionary with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'min_error': np.min(np.abs(y_test_actual - predictions)),
        'max_error': np.max(np.abs(y_test_actual - predictions)),
        'mean_error': np.mean(y_test_actual - predictions)
    }
    
    return metrics


def print_metrics_summary(metrics):
    """
    Print detailed metrics summary.
    
    Args:
        metrics (dict): Metrics dictionary
    """
    print("\n" + "="*60)
    print("DETAILED PREDICTION METRICS")
    print("="*60)
    print(f"RMSE:       ${metrics['rmse']:.2f}")
    print(f"MAE:        ${metrics['mae']:.2f}")
    print(f"MAPE:       {metrics['mape']:.2f}%")
    print(f"Min Error:  ${metrics['min_error']:.2f}")
    print(f"Max Error:  ${metrics['max_error']:.2f}")
    print(f"Mean Error: ${metrics['mean_error']:.2f}")
    print("="*60 + "\n")
