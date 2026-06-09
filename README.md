# Tesla Stock Forecast Engine

A professional machine learning project that predicts Tesla (TSLA) stock prices using LSTM (Long Short-Term Memory) neural networks.

---

## 📊 Project Overview

This project leverages deep learning to forecast Tesla's opening stock prices based on historical data. The model uses a 60-day lookback window to predict the next day's price, combining data science best practices with real financial data.

**Key Features:**
- 🤖 LSTM neural network architecture
- 📈 Automatic data download from Yahoo Finance
- 📊 Comprehensive data visualization
- 📉 Model performance metrics (RMSE, MAE)
- 💾 Trained model persistence
- 🔄 Easy-to-modify configuration

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
- 🤖 Train the LSTM model
- 📈 Evaluate predictions on test data
- 💾 Save the trained model

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
├── tesla_stock_prediction.py    # Main model script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── tesla_lstm_model.keras       # Trained model (generated after running)
└── tesla_stock_data.csv         # Downloaded stock data (generated after running)
```

---

## 🔧 Configuration

Modify the `CONFIG` dictionary in `tesla_stock_prediction.py` to customize:

```python
CONFIG = {
    "ticker": "TSLA",                    # Stock ticker symbol
    "start_date": "2010-01-01",         # Historical data start date
    "end_date": None,                    # None = today's date
    "look_back": 60,                     # Days of history for prediction
    "train_split": 0.75,                 # 75% train, 25% test
    "epochs": 100,                       # Training iterations
    "batch_size": 32,                    # Samples per iteration
    "early_stopping_patience": 50,       # Stop if no improvement
    "model_save_path": "tesla_lstm_model.keras",
    "data_save_path": "tesla_stock_data.csv",
}
```

---

## 📊 Model Architecture

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

---

## 📈 Output & Metrics

After training, the model generates:

1. **Correlation Heatmap** - Feature relationships
2. **Price History Chart** - Historical TSLA prices
3. **Price Distribution** - Frequency analysis
4. **Training History** - Loss and MSE over epochs
5. **Prediction Visualization** - Actual vs. Predicted prices

**Performance Metrics:**
- **RMSE** (Root Mean Squared Error) - Average prediction error
- **MAE** (Mean Absolute Error) - Average absolute deviation

Example output:
```
==================================================
Test RMSE: $2.45
Test MAE:  $1.89
==================================================
```

---

## 🎯 How It Works

### Data Pipeline

```
1. Download Data (Yahoo Finance)
   ↓
2. Data Exploration & Visualization
   ↓
3. Normalization (MinMaxScaler)
   ↓
4. Train/Test Split (75/25)
   ↓
5. Sequence Creation (60-day lookback)
   ↓
6. Model Training (LSTM)
   ↓
7. Prediction & Evaluation
   ↓
8. Results Visualization
```

### Key Concepts

- **LSTM Networks:** Handle sequential data and long-term dependencies
- **Normalization:** Scales data to 0-1 range for better model performance
- **Lookback Window:** Uses 60 days of history to predict the next day
- **Early Stopping:** Prevents overfitting by monitoring training loss

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
**Solution:** Check your internet connection and ensure Yahoo Finance is accessible

### Issue: Model takes too long to train
**Solution:** Reduce `epochs` or `batch_size` in CONFIG

---

## 📚 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)
- [LSTM Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Developer6316**

Feel free to fork, modify, and improve this project!

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain. Always conduct thorough research and consult financial advisors before making investment decisions.

---

## 📧 Contact & Support

For questions or issues, please open a GitHub Issue in the repository.

**Happy forecasting! 🚀📈**
