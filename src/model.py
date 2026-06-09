# -*- coding: utf-8 -*-
"""
Model building and training module for Tesla stock prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def build_model(look_back, lstm_units_1=50, lstm_units_2=64, dense_units_1=32, dense_units_2=16):
    """
    Build LSTM model architecture.
    
    Args:
        look_back (int): Input sequence length
        lstm_units_1 (int): First LSTM layer units (default: 50)
        lstm_units_2 (int): Second LSTM layer units (default: 64)
        dense_units_1 (int): First Dense layer units (default: 32)
        dense_units_2 (int): Second Dense layer units (default: 16)
    
    Returns:
        keras.Model: Compiled model
    """
    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(lstm_units_2, return_sequences=False),
        Dense(dense_units_1, activation="relu"),
        Dense(dense_units_2, activation="relu"),
        Dense(1),
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def train_model(model, x_train, y_train, epochs, batch_size, patience):
    """
    Train the model with early stopping.
    
    Args:
        model (keras.Model): Model to train
        x_train (np.ndarray): Training sequences
        y_train (np.ndarray): Training targets
        epochs (int): Number of epochs
        batch_size (int): Batch size
        patience (int): Early stopping patience
    
    Returns:
        keras.callbacks.History: Training history
    """
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
    """
    Plot training loss.
    
    Args:
        history (keras.callbacks.History): Training history
    """
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


def save_model(model, model_path):
    """
    Save trained model.
    
    Args:
        model (keras.Model): Model to save
        model_path (str): Path to save model
    """
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")


def load_model(model_path):
    """
    Load trained model.
    
    Args:
        model_path (str): Path to model file
    
    Returns:
        keras.Model: Loaded model
    """
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model
