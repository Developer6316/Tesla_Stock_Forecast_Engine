# -*- coding: utf-8 -*-
"""
Tesla Stock Price Prediction using LSTM - Main Pipeline
Predicts Tesla's opening stock price using historical data.
Automatically downloads data from Yahoo Finance.
"""

import json
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import download_stock_data, explore_data, plot_correlation_heatmap, plot_price_history, plot_price_distribution
from data_processor import prepare_data, create_sequences, validate_data
from model import build_model, train_model, plot_training_history, save_model
from evaluator import evaluate_predictions, plot_predictions, get_prediction_metrics_summary, print_metrics_summary


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main pipeline."""
    print("="*60)
    print("TESLA STOCK PRICE PREDICTION - LSTM MODEL")
    print("="*60)
    
    # Load config
    config = load_config()
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    paths_config = config['paths']
    
    # [1/8] Download data
    print("\n[1/8] Downloading data...")
    df = download_stock_data(
        model_config['ticker'],
        model_config['start_date'],
        model_config['end_date']
    )
    
    # [2/8] Explore data
    print("\n[2/8] Exploring data...")
    df = explore_data(df)
    
    # [3/8] Visualizations
    print("\n[3/8] Generating visualizations...")
    plot_correlation_heatmap(df)
    plot_price_history(df)
    plot_price_distribution(df)
    
    # [4/8] Prepare data
    print("\n[4/8] Preparing data...")
    df, scaled_data, train_data, test_data, scaler = prepare_data(
        df,
        column="Open",
        train_split=data_config['train_split'],
        look_back=data_config['look_back']
    )
    
    # [5/8] Create sequences
    print("\n[5/8] Creating sequences...")
    x_train, y_train = create_sequences(train_data, data_config['look_back'])
    x_test, y_test = create_sequences(test_data, data_config['look_back'])
    
    validate_data(x_train, y_train)
    validate_data(x_test, y_test)
    
    # [6/8] Build model
    print("\n[6/8] Building model...")
    model = build_model(data_config['look_back'])
    model.summary()
    
    # [7/8] Train model
    print("\n[7/8] Training model...")
    history = train_model(
        model,
        x_train,
        y_train,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        patience=training_config['early_stopping']['patience']
    )
    
    plot_training_history(history)
    
    # [8/8] Evaluate
    print("\n[8/8] Evaluating model...")
    predictions, y_test_actual, rmse, mae = evaluate_predictions(
        model, x_test, y_test, scaler
    )
    
    metrics = get_prediction_metrics_summary(y_test_actual, predictions)
    print_metrics_summary(metrics)
    
    plot_predictions(df, int(len(df) * data_config['train_split']), data_config['look_back'], predictions)
    
    # Save model
    print(f"\nSaving model to {paths_config['model_save']}...")
    save_model(model, paths_config['model_save'])
    
    # Save data
    print(f"Saving data to {paths_config['data_save']}...")
    df.to_csv(paths_config['data_save'])
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
