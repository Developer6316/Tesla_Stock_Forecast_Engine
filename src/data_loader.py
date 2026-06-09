# -*- coding: utf-8 -*-
"""
Data loading and exploration module for Tesla stock prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'TSLA')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD' or None for today
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    print(f"Downloading {ticker} stock data from {start_date} to {end_date or 'today'}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✓ Downloaded {len(df)} trading days")
        return df
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        raise


def explore_data(df):
    """
    Display basic information about the data.
    
    Args:
        df (pd.DataFrame): Stock data
    
    Returns:
        pd.DataFrame: Input dataframe
    """
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData info:")
    print(df.info())
    print(f"\nMissing values:\n{df.isna().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
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
    plt.title("Tesla (TSLA) Open Stock Price History", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Open Price (USD)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_price_distribution(df):
    """Plot distribution of opening prices."""
    price_counts = df["Open"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.bar(price_counts.index, price_counts.values, color="steelblue")
    plt.title("Open Share Price Distribution", fontsize=16)
    plt.xlabel("Price (USD)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
