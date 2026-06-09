# Tesla Stock Forecast Engine

A professional machine learning project that predicts Tesla (TSLA) stock prices using LSTM (Long Short-Term Memory) neural networks with a modular, production-ready architecture.

---

## 📊 Project Overview

This project leverages deep learning to forecast Tesla's opening stock prices based on historical data. The model uses a 60-day lookback window to predict the next day's price, combining data science best practices with real financial data from Yahoo Finance.

**Key Features:**
- 🤖 LSTM neural network architecture
- 📈 Automatic data download from Yahoo Finance
- 📊 Comprehensive data visualization
- 📉 Model performance metrics (RMSE, MAE, MAPE)
- 💾 Trained model persistence
- 🔄 Easy-to-modify JSON configuration
- 🧩 Modular architecture with reusable components
- 🧪 Comprehensive unit tests
- 📦 Clean separation of concerns

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Developer6316/Tesla_Stock_Forecast_Engine.git
cd Tesla_Stock_Forecast_Engine
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Model

```bash
python tesla_stock_prediction.py
```

The script will automatically:
- 📥 Download Tesla stock data from Yahoo Finance
- 📊 Generate correlation and trend visualizations
- 🧹 Prepare and normalize the data
- 🤖 Train the LSTM model with early stopping
- 📈 Evaluate predictions on test data
- 📊 Display detailed performance metrics
- 💾 Save the trained model and data

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
yfinance>=0.2.0
```

**Quick install:**
```bash
pip install -r requirements.txt
```

### Installing Individual Packages

If you prefer manual installation:

```bash
# Core ML libraries
pip install tensorflow keras

# Data processing
pip install numpy pandas scikit-learn

# Visualization
pip install matplotlib seaborn

# Stock data download
pip install yfinance
```

---

## 📁 Project Structure

```
Tesla_Stock_Forecast_Engine/
├── tesla_stock_prediction.py    # Main entry point
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── src/                         # Main package
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Download & explore data
│   ├── data_processor.py       # Prepare & create sequences
│   ├── model.py                # Build & train model
│   └── evaluator.py            # Evaluate & predict
│
├── tests/
│   ├── __init__.py             # Tests package init
│   └── test_model.py           # Unit tests
│
└── (Generated after running)
    ├── tesla_lstm_model.keras  # Trained model
    └── tesla_stock_data.csv    # Downloaded stock data
```

---

## ⚙️ Configuration

All model parameters are configured in `config.json`. Modify this file to customize:

**Model Settings:**
```json
"model": {
  "ticker": "TSLA",              # Stock ticker symbol
  "start_date": "2010-01-01",   # Historical data start
  "end_date": null              # null = today's date
}
```

**Data Settings:**
```json
"data": {
  "look_back": 60,              # Days of history to use
  "train_split": 0.75,          # 75% train, 25% test
  "normalization": "MinMaxScaler"
}
```

**Training Settings:**
```json
"training": {
  "epochs": 100,
  "batch_size": 32,
  "early_stopping": {
    "monitor": "loss",
    "patience": 50
  }
}
```

---

## 🧠 Model Architecture

The LSTM model consists of:

| Layer | Configuration |
|-------|---------------|
| **LSTM Layer 1** | 50 units, return_sequences=True |
| **LSTM Layer 2** | 64 units, return_sequences=False |
| **Dense Layer 1** | 32 units, ReLU activation |
| **Dense Layer 2** | 16 units, ReLU activation |
| **Output Layer** | 1 unit (price prediction) |

**Optimizer:** Adam  
**Loss Function:** Mean Squared Error (MSE)  
**Regularization:** Early Stopping

---

## 📈 Output & Metrics

After training, the model generates:

1. **Correlation Heatmap** - Feature relationships visualization
2. **Price History Chart** - Historical TSLA prices over time
3. **Price Distribution** - Frequency analysis of prices
4. **Training History** - Loss and MSE metrics over epochs
5. **Prediction Visualization** - Actual vs. Predicted prices comparison

**Performance Metrics:**
- **RMSE** (Root Mean Squared Error) - Average prediction error
- **MAE** (Mean Absolute Error) - Average absolute deviation
- **MAPE** (Mean Absolute Percentage Error) - Percentage error
- **Min/Max Error** - Error range analysis

Example output:
```
============================================================
DETAILED PREDICTION METRICS
============================================================
RMSE:       $2.45
MAE:        $1.89
MAPE:       1.23%
Min Error:  $0.12
Max Error:  $5.67
Mean Error: $0.34
============================================================
```

---

## 🎯 How It Works

### Data Pipeline

```
1. Download Data (Yahoo Finance)
   ↓
2. Data Exploration & Visualization
   ↓
3. Normalization (MinMaxScaler: 0-1 range)
   ↓
4. Train/Test Split (75/25)
   ↓
5. Sequence Creation (60-day lookback windows)
   ↓
6. Model Training (LSTM with Early Stopping)
   ↓
7. Prediction & Evaluation
   ↓
8. Results Visualization
```

### Module Breakdown

**data_loader.py** - Download and explore stock data
- `download_stock_data()` - Fetch data from Yahoo Finance
- `explore_data()` - Display data info and statistics
- Visualization functions

**data_processor.py** - Prepare data for training
- `prepare_data()` - Normalize and split data
- `create_sequences()` - Create LSTM input sequences
- `validate_data()` - Ensure data integrity

**model.py** - Build and train the model
- `build_model()` - Create LSTM architecture
- `train_model()` - Train with early stopping
- `save_model()` / `load_model()` - Model persistence

**evaluator.py** - Evaluate predictions
- `evaluate_predictions()` - Generate metrics
- `plot_predictions()` - Visualize results
- `get_prediction_metrics_summary()` - Detailed metrics

### Key Concepts

- **LSTM Networks:** Handle sequential data and long-term dependencies in time series
- **Normalization:** Scales data to 0-1 range for better model convergence
- **Lookback Window:** Uses 60 days of history to predict the next day
- **Early Stopping:** Prevents overfitting by monitoring validation loss
- **Modular Design:** Each function is independent and reusable

---

## 🧪 Testing

Run comprehensive unit tests:

```bash
# Run all tests
python -m unittest tests/test_model.py

# Run specific test class
python -m unittest tests.test_model.TestDataPreparation

# Run with verbose output
python -m unittest tests/test_model.py -v
```

**Test Coverage:**
- Data preparation and normalization
- Sequence creation validation
- Train/test split verification
- Metrics calculation accuracy
- Configuration file validation
- Model parameter checks

---

## 🔍 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'yfinance'`
**Solution:** Install yfinance
```bash
pip install yfinance
```

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution:** Install TensorFlow
```bash
pip install tensorflow
```

### Issue: Data download fails
**Solution:** 
- Check your internet connection
- Ensure Yahoo Finance is accessible
- Try a different date range in config.json

### Issue: Model takes too long to train
**Solution:** 
- Reduce `epochs` in config.json
- Reduce `batch_size` (but not below 16)
- Use a shorter date range (fewer training samples)

### Issue: GPU not being used
**Solution:**
- Install GPU support: `pip install tensorflow[and-cuda]`
- Check CUDA and cuDNN are installed
- Verify GPU availability: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## 📚 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)
- [LSTM Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 📝 License

This project is open source and available under the MIT License. See LICENSE file for details.

---

## 👤 Author

**Developer6316**

Feel free to fork, modify, and improve this project! Contributions are welcome.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes and add tests
4. Commit changes (`git commit -m 'Add YourFeature'`)
5. Push to branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

Please ensure:
- Your code follows PEP 8 style guide
- All tests pass
- New features have test coverage
- Documentation is updated

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Stock market predictions are inherently uncertain and subject to market volatility. 

**Important:**
- Past performance does not guarantee future results
- Use predictions at your own risk
- Never rely solely on ML models for investment decisions
- Always conduct thorough research and due diligence
- Consult with financial advisors before making investment decisions
- This is not financial advice

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Multi-stock prediction capability
- [ ] Additional features (Volume, High, Low, Close)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble models (Multiple LSTM, GRU, etc.)
- [ ] Web API deployment (FastAPI, Flask)
- [ ] Real-time predictions
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Interactive dashboard (Streamlit)
- [ ] Attention mechanisms for better predictions

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open a GitHub Issue in the repository
- Check existing issues for solutions
- Provide detailed error messages and reproduction steps

**Happy forecasting! 🚀📈**

---

*Last Updated: June 2026 | Version 1.0.0*
