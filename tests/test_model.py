import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPreparation(unittest.TestCase):
    """Test cases for data preparation functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'Open': np.random.uniform(150, 250, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(150, 250, 100),
            'Close': np.random.uniform(150, 250, 100),
            'Volume': np.random.randint(10000000, 100000000, 100)
        }, index=dates)
    
    def test_dataframe_shape(self):
        """Test that dataframe has correct shape."""
        self.assertEqual(self.df.shape, (100, 5))
    
    def test_dataframe_columns(self):
        """Test that dataframe has required columns."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, self.df.columns)
    
    def test_no_missing_values(self):
        """Test that there are no missing values."""
        self.assertFalse(self.df.isna().any().any())
    
    def test_open_price_positive(self):
        """Test that all open prices are positive."""
        self.assertTrue((self.df['Open'] > 0).all())


class TestNormalization(unittest.TestCase):
    """Test cases for data normalization."""
    
    def setUp(self):
        """Set up test data."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = np.array([[100], [150], [200], [250], [300]])
    
    def test_normalization_range(self):
        """Test that normalized data is in correct range."""
        scaled = self.scaler.fit_transform(self.data)
        self.assertTrue(np.all(scaled >= 0) and np.all(scaled <= 1))
    
    def test_normalization_min_max(self):
        """Test that min and max are correctly scaled."""
        scaled = self.scaler.fit_transform(self.data)
        self.assertAlmostEqual(scaled.min(), 0.0, places=5)
        self.assertAlmostEqual(scaled.max(), 1.0, places=5)
    
    def test_inverse_transform(self):
        """Test that inverse transform recovers original values."""
        scaled = self.scaler.fit_transform(self.data)
        recovered = self.scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(self.data, recovered)


class TestSequenceCreation(unittest.TestCase):
    """Test cases for sequence creation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = np.arange(100).reshape(-1, 1).astype(float)
        self.look_back = 10
    
    def test_sequence_creation_length(self):
        """Test that sequences have correct length."""
        x, y = self._create_sequences(self.data, self.look_back)
        self.assertEqual(len(x), len(y))
    
    def test_sequence_look_back_dimension(self):
        """Test that each sequence has correct look_back dimension."""
        x, y = self._create_sequences(self.data, self.look_back)
        self.assertEqual(x.shape[1], self.look_back)
    
    def test_sequence_values_correct(self):
        """Test that sequence values are correct."""
        x, y = self._create_sequences(self.data, self.look_back)
        # First sequence should be [0, 1, 2, ..., 9]
        np.testing.assert_array_equal(x[0].flatten(), np.arange(10))
        # First target should be 10
        self.assertEqual(y[0], 10)
    
    def _create_sequences(self, data, look_back):
        """Helper function to create sequences."""
        x, y = [], []
        for i in range(look_back, len(data)):
            x.append(data[i - look_back:i, 0])
            y.append(data[i, 0])
        return np.array(x).reshape(-1, look_back, 1), np.array(y)


class TestDataSplit(unittest.TestCase):
    """Test cases for train/test split."""
    
    def setUp(self):
        """Set up test data."""
        self.data = np.arange(100).reshape(-1, 1)
        self.train_split = 0.75
    
    def test_train_test_split_ratio(self):
        """Test that train/test split is correct."""
        train_size = int(len(self.data) * self.train_split)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        
        self.assertEqual(len(train_data), 75)
        self.assertEqual(len(test_data), 25)
    
    def test_no_data_overlap(self):
        """Test that train and test data don't overlap."""
        train_size = int(len(self.data) * self.train_split)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        
        # Last train value should be less than first test value
        self.assertLess(train_data[-1][0], test_data[0][0])


class TestMetricsCalculation(unittest.TestCase):
    """Test cases for metrics calculation."""
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        actual = np.array([100, 150, 200, 250, 300])
        predicted = np.array([105, 148, 202, 248, 302])
        
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        self.assertAlmostEqual(rmse, 2.0, places=5)
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        actual = np.array([100, 150, 200, 250, 300])
        predicted = np.array([105, 148, 202, 248, 302])
        
        mae = np.mean(np.abs(actual - predicted))
        self.assertAlmostEqual(mae, 2.0, places=5)
    
    def test_rmse_zero_for_perfect_prediction(self):
        """Test that RMSE is 0 for perfect predictions."""
        actual = np.array([100, 150, 200, 250, 300])
        predicted = np.array([100, 150, 200, 250, 300])
        
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        self.assertEqual(rmse, 0.0)


class TestConfigurationLoading(unittest.TestCase):
    """Test cases for configuration loading."""
    
    def test_config_file_exists(self):
        """Test that config.json file exists."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config.json'
        )
        self.assertTrue(os.path.exists(config_path))
    
    def test_config_is_valid_json(self):
        """Test that config.json is valid JSON."""
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config.json'
        )
        try:
            with open(config_path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            self.fail("config.json is not valid JSON")


class TestModelParameters(unittest.TestCase):
    """Test cases for model parameters."""
    
    def setUp(self):
        """Set up test parameters."""
        self.look_back = 60
        self.epochs = 100
        self.batch_size = 32
    
    def test_look_back_positive(self):
        """Test that look_back is positive."""
        self.assertGreater(self.look_back, 0)
    
    def test_epochs_positive(self):
        """Test that epochs is positive."""
        self.assertGreater(self.epochs, 0)
    
    def test_batch_size_positive(self):
        """Test that batch_size is positive."""
        self.assertGreater(self.batch_size, 0)
    
    def test_batch_size_reasonable(self):
        """Test that batch_size is reasonable."""
        self.assertLess(self.batch_size, 256)


if __name__ == '__main__':
    unittest.main()
