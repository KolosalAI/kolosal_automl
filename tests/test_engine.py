import os
import sys
import tempfile
import unittest
import pickle
import numpy as np
from concurrent.futures import TimeoutError

# Import the inference engine and required configuration and enums from the corresponding path.
from modules.engine.inference_engine import InferenceEngine
from modules.configs import CPUAcceleratedModelConfig, EngineState, ModelType

# Import a basic model from scikit-learn
from sklearn.linear_model import LinearRegression

class TestInferenceEngineSklearn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory to store the scikit-learn model file.
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.temp_dir.name, "linear_regression_model.pkl")
        
        # Create and train a simple LinearRegression model.
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([3, 5, 7, 9])
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Optionally set feature names if available.
        if hasattr(model, "feature_names_in_"):
            model.feature_names_in_ = np.array(["feature1", "feature2"])
        
        # Save the model to disk.
        with open(cls.model_path, 'wb') as f:
            pickle.dump(model, f)
    
    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
    
    def setUp(self):
        # Create a CPUAcceleratedModelConfig with deduplication and batching enabled for testing.
        self.config = CPUAcceleratedModelConfig(
            model_version="vtest",
            enable_request_deduplication=True,
            max_cache_entries=10,
            cache_ttl_seconds=60,
            enable_batching=True,
            initial_batch_size=2,
            min_batch_size=1,
            max_batch_size=4,
            batch_timeout=0.1,
            max_concurrent_requests=5,
            enable_monitoring=True,
            monitoring_window=50,
            debug_mode=True,
            num_threads=2,
            enable_quantization=False,
            enable_input_quantization=False,
            enable_model_quantization=False,
            enable_feature_scaling=False,
            enable_warmup=False,
            set_cpu_affinity=False,
            throttle_on_high_cpu=False,
            cpu_threshold_percent=90,
            memory_high_watermark_mb=500,
            memory_limit_gb=2,
            enable_intel_optimization=False,
            enable_adaptive_batching=False
        )
        self.engine = InferenceEngine(config=self.config)
    
    def tearDown(self):
        self.engine.shutdown()

    def test_load_model_success(self):
        # Test that the scikit-learn model loads successfully.
        load_success = self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN)
        self.assertTrue(load_success)
        self.assertIsNotNone(self.engine.model)
        self.assertEqual(self.engine.model_type, ModelType.SKLEARN)
    
    def test_predict_without_model(self):
        # Without loading a model, prediction should return an error.
        X = np.random.rand(3, 2)
        success, preds, meta = self.engine.predict(X)
        self.assertFalse(success)
        self.assertIsNone(preds)
        self.assertIn("error", meta)
    
    def test_predict_with_model(self):
        # Load the scikit-learn model.
        self.assertTrue(self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN))
        
        # Create some test input data.
        X = np.array([[5, 6], [7, 8]])
        # For our linear regression model trained on [1,2],[2,3],[3,4],[4,5] with y = 2*x1 + 1,
        # predictions should follow the learned linear relationship.
        success, preds, meta = self.engine.predict(X)
        self.assertTrue(success)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(preds.shape[0], X.shape[0])
        self.assertIn("inference_time_ms", meta)
        self.assertEqual(meta["batch_size"], X.shape[0])
    
    def test_predict_batch(self):
        # Load the model and test batch predictions.
        self.assertTrue(self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN))
        X = np.array([[1, 2], [3, 4]])
        # Using predict_batch should return a Future.
        future = self.engine.predict_batch(X)
        try:
            result, meta = future.result(timeout=2)
        except TimeoutError:
            self.fail("Batch prediction timed out")
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape[0], X.shape[0])
    
    def test_get_state_and_shutdown(self):
        # After initialization, state should be READY.
        self.assertEqual(self.engine.get_state(), EngineState.READY)
        # Shutdown and then check state.
        self.engine.shutdown()
        self.assertEqual(self.engine.get_state(), EngineState.STOPPED)
    
    def test_get_metrics(self):
        # Load model and perform a prediction to update metrics.
        self.assertTrue(self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN))
        X = np.random.rand(3, 2)
        self.engine.predict(X)
        metrics = self.engine.get_metrics()
        self.assertIn("total_requests", metrics)
        self.assertIn("engine_state", metrics)
    
    def test_get_model_info(self):
        # Test get_model_info before and after model loading.
        info_before = self.engine.get_model_info()
        self.assertIn("error", info_before)
        self.assertTrue(self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN))
        info_after = self.engine.get_model_info()
        self.assertIn("model_info", info_after)
        self.assertIn("config", info_after)
    
    def test_clear_cache(self):
        # Load model, make a prediction to populate cache, then clear it.
        self.assertTrue(self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN))
        X = np.random.rand(2, 2)
        # Make two predictions with the same input so that deduplication may cache.
        self.engine.predict(X)
        cache_stats_before = self.engine.get_metrics().get("cache_stats", {})
        self.engine.clear_cache()
        cache_stats_after = self.engine.get_metrics().get("cache_stats", {})
        # If cache is enabled, size should be 0 after clearing.
        if cache_stats_before:
            self.assertEqual(cache_stats_after.get("size", 0), 0)
    
    def test_set_get_feature_names(self):
        # Test setting and getting feature names.
        feature_names = ["feature1", "feature2"]
        self.engine.set_feature_names(feature_names)
        retrieved = self.engine.get_feature_names()
        self.assertEqual(retrieved, feature_names)
    
    def test_get_memory_usage(self):
        # Test that memory usage returns a dict with expected keys.
        mem_usage = self.engine.get_memory_usage()
        self.assertIn("rss_mb", mem_usage)
        self.assertIn("vms_mb", mem_usage)
        self.assertIn("percent", mem_usage)
        self.assertIn("cpu_percent", mem_usage)
    
    def test_set_config_updates(self):
        # Load model then update configuration.
        self.assertTrue(self.engine.load_model(self.model_path, model_type=ModelType.SKLEARN))
        new_config = CPUAcceleratedModelConfig(
            model_version="vtest_updated",
            enable_request_deduplication=False,
            max_cache_entries=5,
            cache_ttl_seconds=30,
            enable_batching=True,
            initial_batch_size=3,
            min_batch_size=1,
            max_batch_size=6,
            batch_timeout=0.2,
            max_concurrent_requests=3,
            enable_monitoring=True,
            monitoring_window=20,
            debug_mode=False,
            num_threads=2,
            enable_quantization=False,
            enable_input_quantization=False,
            enable_model_quantization=False,
            enable_feature_scaling=False,
            enable_warmup=False,
            set_cpu_affinity=False,
            throttle_on_high_cpu=False,
            cpu_threshold_percent=80,
            memory_high_watermark_mb=300,
            memory_limit_gb=1,
            enable_intel_optimization=False,
            enable_adaptive_batching=False
        )
        self.engine.set_config(new_config)
        # Ensure the configuration was updated.
        self.assertEqual(self.engine.config.model_version, "vtest_updated")
        # If deduplication is disabled, result_cache should be None.
        self.assertIsNone(self.engine.result_cache)

if __name__ == '__main__':
    unittest.main()
