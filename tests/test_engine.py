import os
import unittest
import tempfile
import numpy as np
import pickle
import shutil
import json
import gc
from unittest.mock import patch, MagicMock

from typing import Dict, Any, Optional

# Assuming these are the correct import paths - adjust as needed
from modules.engine.inference_engine import InferenceEngine
from modules.configs import InferenceEngineConfig, ModelType, EngineState

####################
# Mock / Test Model Classes at the Top Level
####################

class MockSklearnModel:
    """Mock class mimicking a minimal sklearn model."""
    def __init__(self):
        self.n_features_in_ = 3
        self.feature_names_in_ = np.array(["f1", "f2", "f3"])

    def predict(self, X):
        return np.ones((X.shape[0],), dtype=float)
    
    def get_params(self, deep=True):
        return {"mock_param": "value"}


class SimpleAddModel:
    """Mock custom model that sums features."""
    def predict(self, X):
        return np.sum(X, axis=1)


class TrivialModel:
    """Trivial model that returns all zeros (or a constant)."""
    def predict(self, X):
        return np.zeros((X.shape[0],))


class TrivialModel42:
    """Trivial model that returns all 42."""
    def predict(self, X):
        return np.array([42] * X.shape[0])


class SimpleModelSum:
    """Simple model that returns the sum of features for each row."""
    def predict(self, X):
        return np.sum(X, axis=1)


class SimpleModelOnes:
    """Simple model that returns an array of ones."""
    def predict(self, X):
        return np.ones(X.shape[0])


####################
# Test Suite
####################

class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Runs once at the beginning of the test suite.
        """
        cls.temp_dir = tempfile.mkdtemp(prefix="test_inference_engine_")

    @classmethod
    def tearDownClass(cls):
        """
        Runs once at the end of the test suite.
        """
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """
        Runs before each test method.
        Initialize a basic engine config and the InferenceEngine object.
        """
        self.config = InferenceEngineConfig(
            debug_mode=True, 
            enable_request_deduplication=True,
            enable_batching=False,  # turn on/off as needed for specific tests
            enable_quantization=False,
            model_version="test-version",  # Add a model version for tests
            memory_high_watermark_mb=100  # Set a reasonable value for tests
        )
        self.engine = InferenceEngine(self.config)
    
    def tearDown(self):
        """
        Runs after each test method.
        Ensure the engine is shutdown to free resources.
        """
        if hasattr(self, 'engine'):
            self.engine.shutdown()

    def test_initial_state(self):
        """
        Test that the engine initializes in the correct state (READY).
        """
        self.assertEqual(self.engine.state, EngineState.READY, 
                         "Engine should be in READY state after initialization.")
    
    def test_set_config_updates_engine(self):
        """
        Test that updating the config modifies engine behavior or state accordingly.
        """
        new_config = InferenceEngineConfig(
            debug_mode=False,
            enable_request_deduplication=False,
            enable_batching=True,
            enable_quantization=True,
            num_threads=2,
            model_version="test-version-updated"
        )
        self.engine.set_config(new_config)
        
        self.assertIn(self.engine.state, [EngineState.READY, EngineState.RUNNING],
                      "Engine should remain READY/RUNNING after config update.")
        self.assertTrue(self.engine.config.enable_batching)
        self.assertTrue(self.engine.config.enable_quantization)
        self.assertFalse(self.engine.config.enable_request_deduplication)

    def test_predict_no_model_loaded(self):
        """
        Test predicting when no model is loaded. Expect an error/False success.
        """
        features = np.random.rand(5, 3)
        success, preds, meta = self.engine.predict(features)
        
        self.assertFalse(success, "Predict should fail if no model is loaded.")
        self.assertIsNone(preds, "Predictions should be None if no model is loaded.")
        self.assertIn("error", meta, "Metadata should contain an error message.")

    def test_load_invalid_model_path(self):
        """
        Test loading a model from an invalid path.
        """
        invalid_path = os.path.join(self.temp_dir, "non_existent_model.pkl")
        result = self.engine.load_model(invalid_path, model_type=ModelType.SKLEARN)
        self.assertFalse(result, "Loading should fail for a non-existent model file.")
        self.assertEqual(self.engine.state, EngineState.ERROR, 
                         "Engine state should be ERROR when load fails.")

    def test_load_sklearn_model(self):
        """
        Test loading a simple (mock) sklearn model and making a prediction.
        """
        model_path = os.path.join(self.temp_dir, "mock_sklearn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(MockSklearnModel(), f)
        
        loaded_ok = self.engine.load_model(model_path, model_type=ModelType.SKLEARN)
        self.assertTrue(loaded_ok, "Engine should load the mock sklearn model successfully.")
        self.assertEqual(self.engine.state, EngineState.READY, 
                         "Engine state should be READY after loading a valid model.")
        
        test_input = np.random.rand(5, 3)
        success, predictions, meta = self.engine.predict(test_input)
        self.assertTrue(success, "Prediction should succeed with a valid model.")
        self.assertIsNotNone(predictions, "Predictions should not be None.")
        self.assertEqual(predictions.shape, (5,), "Predictions shape should match [n_samples].")
        self.assertTrue((predictions == 1.0).all(), "Mock model should always return 1.0.")

    def test_cache_functionality(self):
        """
        Test that the cache is used correctly when request deduplication is enabled.
        """
        model_path = os.path.join(self.temp_dir, "simple_add_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(SimpleAddModel(), f)
        
        self.engine.load_model(model_path, model_type=ModelType.CUSTOM)
        
        features = np.array([[1, 2, 3]], dtype=float)
        success1, preds1, meta1 = self.engine.predict(features)
        self.assertTrue(success1, "First predict call should succeed.")
        
        success2, preds2, meta2 = self.engine.predict(features)
        self.assertTrue(success2, "Second predict call should also succeed.")
        
        self.assertTrue(meta2.get("cached", False), 
                        "Second identical request should be served from cache.")
        self.assertTrue(np.array_equal(preds1, preds2), 
                        "Predictions should match for identical inputs.")

    def test_predict_batch_when_batching_disabled(self):
        """
        Test `predict_batch` call when batching is disabled.
        Expect immediate synchronous prediction.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "trivial_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(TrivialModel(), f)

            # Load model with batching turned off
            self.engine.load_model(model_path, model_type=ModelType.CUSTOM)

            # Generate 10 random feature rows for test
            features = np.random.rand(10, 3)
            future = self.engine.predict_batch(features_batch=features, priority=0)

            # Batching is disabled, so this future should already be finished
            self.assertTrue(
                future.done(),
                "Future should be done immediately with batching disabled."
            )

            preds, meta = future.result()
            self.assertIsInstance(preds, np.ndarray, "Predictions should be a numpy array.")
            self.assertEqual(preds.shape, (10,), "Should predict one value per input row.")
            self.assertTrue((preds == 0).all(), "Trivial model always returns zero.")

    def test_batch_processor_enabled(self):
        """
        Test that enabling batching initializes the batch processor
        and that a submitted batch is processed asynchronously.
        """
        self.engine.shutdown()  # shut down the old engine
        config_with_batch = InferenceEngineConfig(
            debug_mode=True,
            enable_batching=True,
            enable_request_deduplication=False,
            model_version="test-batching"
        )
        self.engine = InferenceEngine(config_with_batch)
        
        model_path = os.path.join(self.temp_dir, "trivial_model_batched.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(TrivialModel42(), f)

        self.engine.load_model(model_path, model_type=ModelType.CUSTOM)

        features = np.random.rand(5, 4)
        future = self.engine.predict_batch(features, priority=0)
        
        # Wait for result
        result = future.result(timeout=5.0)
        self.assertIsInstance(result, tuple, "Result should be (predictions, metadata).")
        
        predictions, metadata = result
        self.assertEqual(predictions.shape[0], 5, "Predictions should match the input batch size.")
        self.assertTrue((predictions == 42).all(), "TrivialModel42 should return 42 for every entry.")

    def test_metrics_update_on_prediction(self):
        """
        Test that metrics are updated properly after a successful prediction.
        """
        model_path = os.path.join(self.temp_dir, "simple_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(SimpleModelOnes(), f)

        self.engine.load_model(model_path, ModelType.CUSTOM)

        features = np.random.rand(10, 3)
        success, preds, meta = self.engine.predict(features)
        self.assertTrue(success, "Prediction should succeed.")
        
        metrics = self.engine.get_metrics()
        self.assertIn("avg_inference_time_ms", metrics, 
                      "Metrics should include average inference time.")
        self.assertIn("total_requests", metrics,
                      "Metrics should include total request count.")
        # Adjust this if your code increments once per sample vs once per request
        self.assertGreaterEqual(metrics.get("total_requests", 0), 1, 
                                "Total requests should increment for predictions.")

    def test_shutdown_changes_state(self):
        """
        Test that calling `shutdown` transitions the engine state correctly.
        """
        self.assertEqual(self.engine.state, EngineState.READY)
        self.engine.shutdown()
        self.assertEqual(self.engine.state, EngineState.STOPPED, 
                         "Engine state should be STOPPED after shutdown.")

    def test_context_manager(self):
        """
        Test using the engine as a context manager, verifying proper shutdown.
        """
        with InferenceEngine(InferenceEngineConfig(model_version="test-context")) as eng:
            self.assertIn(eng.state, [EngineState.READY, EngineState.RUNNING],
                          "Engine should be in READY or RUNNING inside context manager.")
        self.assertEqual(eng.state, EngineState.STOPPED,
                         "Engine should be STOPPED after exiting context manager.")

    def test_save_and_model_info(self):
        """
        Test the save method and verify model info is correctly saved.
        """
        model_path = os.path.join(self.temp_dir, "model_to_save.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(SimpleModelSum(), f)
        
        self.engine.load_model(model_path, ModelType.CUSTOM)
        
        save_dir = os.path.join(self.temp_dir, "saved_engine")
        saved = self.engine.save(save_dir, include_config=True, include_metrics=True)
        self.assertTrue(saved, "Save method should return True on success")
        
        # Adjust if your InferenceEngine saves under a different file name
        saved_model_path = os.path.join(save_dir, "model.pkl")
        self.assertTrue(os.path.exists(saved_model_path),
                        "Model file should be saved as model.pkl")
        
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")),
                        "Config file should be saved")
        self.assertTrue(os.path.exists(os.path.join(save_dir, "model_info.json")),
                        "Model info file should be saved")
        
        model_info = self.engine.get_model_info()
        self.assertIsInstance(model_info, dict, "Model info should be a dictionary")
        self.assertIn("model_info", model_info, "Model info should contain 'model_info' key")

    @patch("psutil.Process")
    def test_memory_monitoring(self, mock_process):
        """
        Test memory monitoring with mocked psutil.
        """
        memory_info_mock = MagicMock()
        memory_info_mock.rss = (self.config.memory_high_watermark_mb + 100) * 1024 * 1024
        process_mock = MagicMock()
        process_mock.memory_info.return_value = memory_info_mock
        mock_process.return_value = process_mock
        
        with patch('gc.collect') as mock_gc:
            self.engine._check_memory_usage()
            mock_gc.assert_called()  # Ensure garbage collector is triggered

if __name__ == "__main__":
    unittest.main()
