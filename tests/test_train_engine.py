import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np
import os
import json
import pickle
from datetime import datetime
from modules.configs import CPUTrainingConfig, ModelType, TrainingState
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.train_engine import TrainingEngine

class TestTrainingEngine(unittest.TestCase):

    def setUp(self):
        # Create a mock config
        self.mock_config = CPUTrainingConfig(
            model_training={
                "model_type": ModelType.SKLEARN,
                "model_class": "sklearn.linear_model.LogisticRegression",
                "problem_type": "classification",
                "hyperparameters": {"C": 1.0}
            },
            dataset={
                "train_path": "data/train.csv",
                "target_column": "target",
                "handle_missing": "mean",
                "handle_outliers": "clip",
                "handle_categorical": "one_hot"
            },
            feature_engineering={
                "enable_scaling": True,
                "scaling_method": "standard"
            },
            model_registry={
                "registry_path": "models",
                "model_name": "test_model",
                "auto_version": True
            }
        )
        
        # Mock logger
        self.mock_logger = MagicMock()
        
        # Initialize TrainingEngine with mock config and logger
        self.engine = TrainingEngine(self.mock_config, self.mock_logger)

    def test_initialization(self):
        self.assertEqual(self.engine.state, TrainingState.INITIALIZING)
        self.assertIsNotNone(self.engine.training_id)
        self.assertEqual(self.engine.config, self.mock_config)
        self.assertEqual(self.engine.logger, self.mock_logger)
        self.assertIsNone(self.engine.model)
        self.assertIsNone(self.engine.best_model)
        self.assertEqual(self.engine.runtime_metrics["total_time"], 0)

    @patch('polars.read_csv')
    def test_load_data(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        self.engine.load_data()
        
        # Verify data loading
        self.assertIsNotNone(self.engine.train_data)
        self.assertEqual(len(self.engine.train_data), 3)
        self.assertEqual(self.engine.feature_columns, ["feature1", "feature2"])
        self.assertEqual(self.engine.state, TrainingState.PREPROCESSING)

    @patch('polars.read_csv')
    def test_load_data_with_missing_columns(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame with missing columns
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df
        
        with self.assertRaises(ValueError):
            self.engine.load_data()

    @patch('polars.read_csv')
    def test_load_data_with_chunking(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Enable chunking in config
        self.mock_config.dataset.enable_chunking = True
        self.mock_config.dataset.chunk_size = 1000
        
        self.engine.load_data()
        
        # Verify data loading with chunking
        self.assertIsNotNone(self.engine.train_data)
        self.assertEqual(len(self.engine.train_data), 3)
        self.assertEqual(self.engine.feature_columns, ["feature1", "feature2"])
        self.assertEqual(self.engine.state, TrainingState.PREPROCESSING)

    @patch('polars.read_csv')
    def test_split_dataset(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Set split strategy
        self.mock_config.dataset.split_strategy = "random"
        self.mock_config.dataset.test_size = 0.2
        self.mock_config.dataset.validation_size = 0.2
        
        self.engine.load_data()
        self.engine._split_dataset()
        
        # Verify dataset splits
        self.assertIsNotNone(self.engine.train_data)
        self.assertIsNotNone(self.engine.validation_data)
        self.assertIsNotNone(self.engine.test_data)
        self.assertEqual(len(self.engine.train_data), 3)
        self.assertEqual(len(self.engine.validation_data), 1)
        self.assertEqual(len(self.engine.test_data), 1)

    @patch('polars.read_csv')
    def test_apply_preprocessing(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = mock_df
        self.engine.preprocessor = mock_preprocessor
        
        self.engine.load_data()
        self.engine._apply_preprocessing()
        
        # Verify preprocessing
        mock_preprocessor.fit.assert_called_once()
        mock_preprocessor.transform.assert_called()

    @patch('polars.read_csv')
    def test_train_model(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        self.engine.model = mock_model
        
        self.engine.load_data()
        self.engine.train_model()
        
        # Verify model training
        mock_model.fit.assert_called_once()
        self.assertEqual(self.engine.state, TrainingState.TRAINING)

    @patch('polars.read_csv')
    def test_evaluate_model(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        self.engine.model = mock_model
        
        self.engine.load_data()
        self.engine.evaluate_model()
        
        # Verify model evaluation
        self.assertIn("train", self.engine.evaluation_results)
        self.assertIn("accuracy", self.engine.evaluation_results["train"]["metrics"])
        self.assertEqual(self.engine.state, TrainingState.EVALUATING)

    @patch('polars.read_csv')
    def test_save_model(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        self.engine.model = mock_model
        
        # Mock preprocessor
        mock_preprocessor = MagicMock()
        self.engine.preprocessor = mock_preprocessor
        
        self.engine.load_data()
        self.engine.train_model()
        self.engine.save_model()
        
        # Verify model saving
        model_path = os.path.join(
            self.mock_config.model_registry.registry_path,
            self.mock_config.model_registry.model_name,
            self.engine._resolve_model_version(),
            "model.pkl"
        )
        self.assertTrue(os.path.exists(model_path))
        self.assertEqual(self.engine.state, TrainingState.SAVING)

    @patch('polars.read_csv')
    def test_inference(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        self.engine.model = mock_model
        
        self.engine.load_data()
        
        # Test inference with a DataFrame
        test_data = pl.DataFrame({
            "feature1": [1, 2],
            "feature2": [4, 5]
        })
        result = self.engine.inference(test_data)
        
        # Verify inference results
        self.assertIn("predictions", result)
        self.assertEqual(len(result["predictions"]), 2)
        mock_model.predict.assert_called_once()

    @patch('polars.read_csv')
    def test_inference_with_dict(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        self.engine.model = mock_model
        
        self.engine.load_data()
        
        # Test inference with a dictionary
        test_data = {"feature1": 1, "feature2": 4}
        result = self.engine.inference(test_data)
        
        # Verify inference results
        self.assertIn("predictions", result)
        self.assertEqual(len(result["predictions"]), 1)
        mock_model.predict.assert_called_once()

    @patch('polars.read_csv')
    def test_inference_with_array(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        self.engine.model = mock_model
        
        self.engine.load_data()
        
        # Test inference with a numpy array
        test_data = np.array([[1, 4], [2, 5]])
        result = self.engine.inference(test_data)
        
        # Verify inference results
        self.assertIn("predictions", result)
        self.assertEqual(len(result["predictions"]), 2)
        mock_model.predict.assert_called_once()

    @patch('polars.read_csv')
    def test_inference_with_missing_columns(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        self.engine.model = mock_model
        
        self.engine.load_data()
        
        # Test inference with missing columns
        test_data = pl.DataFrame({
            "feature1": [1, 2]
        })
        with self.assertRaises(ValueError):
            self.engine.inference(test_data)

    @patch('polars.read_csv')
    def test_batch_predict(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        self.engine.model = mock_model
        
        self.engine.load_data()
        
        # Test batch prediction
        result = self.engine.batch_predict("data/test.csv", "output/predictions.csv")
        
        # Verify batch prediction results
        self.assertTrue(result["success"])
        self.assertEqual(result["rows_processed"], 3)
        self.assertGreater(result["processing_time_seconds"], 0)
        self.assertGreater(result["rows_per_second"], 0)

    @patch('polars.read_csv')
    def test_batch_predict_with_invalid_file(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock model
        mock_model = MagicMock()
        self.engine.model = mock_model
        
        self.engine.load_data()
        
        # Test batch prediction with invalid file
        with self.assertRaises(ValueError):
            self.engine.batch_predict("data/test.txt", "output/predictions.csv")

    @patch('polars.read_csv')
    def test_get_status(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        self.engine.load_data()
        
        # Test get_status
        status = self.engine.get_status()
        
        # Verify status
        self.assertIn("training_id", status)
        self.assertIn("state", status)
        self.assertIn("start_time", status)
        self.assertIn("model_name", status)
        self.assertIn("feature_count", status)
        self.assertIn("runtime_metrics", status)

    @patch('polars.read_csv')
    def test_to_dict(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        self.engine.load_data()
        
        # Test to_dict
        engine_dict = self.engine.to_dict()
        
        # Verify dictionary
        self.assertIn("training_id", engine_dict)
        self.assertIn("state", engine_dict)
        self.assertIn("config", engine_dict)
        self.assertIn("runtime_metrics", engine_dict)
        self.assertIn("feature_columns", engine_dict)

    @patch('polars.read_csv')
    def test_cleanup(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        self.engine.load_data()
        
        # Test cleanup
        self.engine.cleanup()
        
        # Verify cleanup
        self.assertIsNone(self.engine.train_data)
        self.assertIsNone(self.engine.validation_data)
        self.assertIsNone(self.engine.test_data)
        self.assertIsNone(self.engine.model)
        self.assertIsNone(self.engine.best_model)
        self.assertIsNone(self.engine.preprocessor)

    @patch('polars.read_csv')
    def test_context_manager(self, mock_read_csv):
        # Mock polars read_csv to return a DataFrame
        mock_df = pl.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Test context manager
        with TrainingEngine(self.mock_config, self.mock_logger) as engine:
            engine.load_data()
            self.assertEqual(engine.state, TrainingState.PREPROCESSING)
        
        # Verify cleanup after context manager
        self.assertIsNone(engine.train_data)
        self.assertIsNone(engine.validation_data)
        self.assertIsNone(engine.test_data)
        self.assertIsNone(engine.model)
        self.assertIsNone(engine.best_model)
        self.assertIsNone(engine.preprocessor)

if __name__ == '__main__':
    unittest.main()
