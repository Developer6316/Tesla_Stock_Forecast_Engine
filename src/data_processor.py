# -*- coding: utf-8 -*-
"""
Data preparation and preprocessing module for Tesla stock prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df, column="Open", train_split=0.75, look_back=60):
    """
    Normalize and split data into train/test sets.
    
    Args:
        df (pd.DataFrame): Stock data
        column (str): Column to use for prediction (default: 'Open')
        train_split (float): Fraction of data for training (default: 0.75)
        look_back (int): Number of days for lookback window (default: 60)
    
    Returns:
        tuple: (df, scaled_data, train_data, test_data, scaler)
    """
    data = df[[column]].values
    
    # Remove any NaN values
    if np.isnan(data).any():
        print(f"Warning: Found NaN values. Dropping them...")
        df = df.dropna()
        data = df[[column]].values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split data
    train_size = int(len(scaled_data) * train_split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - look_back:]
    
    print(f"Train size: {len(train_data)} | Test size: {len(test_data)}")
    
    return df, scaled_data, train_data, test_data, scaler


def create_sequences(data, look_back=60):
    """
    Create sequences for LSTM training.
    
    Args:
        data (np.ndarray): Normalized data
        look_back (int): Number of days to look back (default: 60)
    
    Returns:
        tuple: (X sequences, y targets)
    """
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x, y


def validate_data(x, y):
    """
    Validate prepared data.
    
    Args:
        x (np.ndarray): Input sequences
        y (np.ndarray): Target values
    
    Returns:
        bool: True if data is valid
    """
    assert len(x) == len(y), "X and y must have same length"
    assert x.ndim == 3, "X must be 3D (samples, timesteps, features)"
    assert y.ndim == 1, "y must be 1D"
    print(f"✓ Data validation passed")
    print(f"  X shape: {x.shape} | y shape: {y.shape}")
    return True
