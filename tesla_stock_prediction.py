# -*- coding: utf-8 -*-
"""
Tesla Stock Price Prediction using LSTM
Predicts Tesla's opening stock price using historical data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    "data_path": "/content/tesla_stock_data_2010_2025.csv",
    "look_back": 60,
    "train_split": 0.75,
    "epochs": 100,
    "batch_size": 32,
    "early_stopping_patience": 50,
    "model_save_path": "tesla_open_lstm.keras",
}


# ============================================================================
# Data Loading & Exploration
# ============================================================================
def load_and_explore_data(file_path):
    """Load CSV data and display basic information."""
    df = pd.read_csv(file_path, parse_dates=True, index_col="Date")
    print(f"Data shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isna().any()}")
    return df


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric columns."""
    plt.figure(figsize=(15, 7))
    sns.heatmap(
        df.select_dtypes(include=np.number).corr(),
        cbar=True,
        annot=True,
        cmap="Blues"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_price_history(df):
    """Plot historical opening prices."""
    plt.figure(figsize=(16, 8))
    plt.plot(df.index, df["Open"], color="red", linewidth=2)
    plt.title("Tesla Open Stock Price History", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Open Price (USD)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_price_distribution(df):
    """Plot distribution of opening prices."""
    price_counts = df["Open"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.bar(price_counts.index, price_counts.values)
    plt.title("Open Share Price Distribution", fontsize=16)
    plt.xlabel("Price (USD)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Data Preparation
# ============================================================================
def prepare_data(df, column="Open", train_split=0.75, look_back=60):
    """Normalize and split data into train/test sets."""
    data = df[[column]].values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split data
    train_size = int(len(scaled_data) * train_split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - look_back:]
    
    print(f"Train size: {len(train_data)} | Test size: {len(test_data)}")
    
    return scaled_data, train_data, test_data, scaler


def create_sequences(data, look_back=60):
    """Create sequences for LSTM training."""
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x, y


# ============================================================================
# Model Building & Training
# ============================================================================
def build_model(look_back):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def train_model(model, x_train, y_train, epochs, batch_size, patience):
    """Train the model with early stopping."""
    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_training_history(history):
    """Plot training loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], linewidth=2, label="Loss")
    plt.plot(history.history["mse"], linewidth=2, label="MSE")
    plt.title("Training History", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_predictions(model, x_test, y_test, scaler):
    """Generate predictions and calculate metrics."""
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
    """Visualize predictions vs actual prices."""
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    test_df["Predictions"] = predictions
    
    plt.figure(figsize=(16, 6))
    plt.title("Tesla Stock Price Prediction (LSTM)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Open Price (USD)", fontsize=12)
    plt.plot(train_df["Open"], linewidth=2, label="Train", alpha=0.8)
    plt.plot(test_df["Open"], linewidth=2, label="Actual Test", alpha=0.8)
    plt.plot(test_df["Predictions"], linewidth=2, label="Predictions", alpha=0.8)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main pipeline."""
    print("Loading data...")
    df = load_and_explore_data(CONFIG["data_path"])
    
    print("\nGenerating visualizations...")
    plot_correlation_heatmap(df)
    plot_price_history(df)
    plot_price_distribution(df)
    
    print("\nPreparing data...")
    scaled_data, train_data, test_data, scaler = prepare_data(
        df,
        column="Open",
        train_split=CONFIG["train_split"],
        look_back=CONFIG["look_back"]
    )
    
    print("Creating sequences...")
    x_train, y_train = create_sequences(train_data, CONFIG["look_back"])
    x_test, y_test = create_sequences(test_data, CONFIG["look_back"])
    
    print(f"x_train: {x_train.shape} | y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape} | y_test: {y_test.shape}")
    
    print("\nBuilding model...")
    model = build_model(CONFIG["look_back"])
    model.summary()
    
    print("\nTraining model...")
    history = train_model(
        model,
        x_train,
        y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        patience=CONFIG["early_stopping_patience"]
    )
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nEvaluating on test set...")
    predictions, y_test_actual, rmse, mae = evaluate_predictions(
        model, x_test, y_test, scaler
    )
    
    print("Visualizing predictions...")
    train_size = int(len(df) * CONFIG["train_split"])
    plot_predictions(df, train_size, CONFIG["look_back"], predictions)
    
    print(f"\nSaving model to {CONFIG['model_save_path']}...")
    model.save(CONFIG["model_save_path"])
    print("Done!")


if __name__ == "__main__":
    main()
