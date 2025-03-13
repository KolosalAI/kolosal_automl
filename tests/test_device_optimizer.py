import unittest
import os
import tempfile
import shutil
import json
from pathlib import Path

# Import the fixed DeviceOptimizer class and related modules
from modules.device_optimizer import (
    DeviceOptimizer, OptimizationMode, create_optimized_configs,
    create_configs_for_all_modes, load_config, safe_dict_serializer
)

# Import the configuration classes that are used by DeviceOptimizer
from modules.configs import (
    QuantizationConfig, BatchProcessorConfig, PreprocessorConfig,
    InferenceEngineConfig, MLTrainingEngineConfig
)


class TestDeviceOptimizer(unittest.TestCase):
    """Test suite for the DeviceOptimizer class and related utilities."""
    
    def setUp(self):
        """Create temporary directories for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "configs")
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoints")
        self.model_registry_path = os.path.join(self.temp_dir, "model_registry")
        
        # Create a test optimizer instance
        self.optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the DeviceOptimizer initializes correctly."""
        # Check that directories were created
        self.assertTrue(os.path.exists(self.config_path))
        self.assertTrue(os.path.exists(self.checkpoint_path))
        self.assertTrue(os.path.exists(self.model_registry_path))
        
        # Check default optimization mode
        self.assertEqual(self.optimizer.optimization_mode, OptimizationMode.BALANCED)
        
        # Test with explicit optimization mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        self.assertEqual(optimizer.optimization_mode, OptimizationMode.PERFORMANCE)

    def test_serialize_config_dict(self):
        """Test the fixed _serialize_config_dict method."""
        # Create a test dictionary with various types including Enum values
        from enum import Enum
        
        class TestEnum(Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"
        
        test_dict = {
            "string": "test",
            "int": 123,
            "float": 45.67,
            "bool": True,
            "enum": TestEnum.VALUE1,
            "list": [1, 2, TestEnum.VALUE2],
            "nested_dict": {
                "key": "value",
                "enum_key": TestEnum.VALUE2
            }
        }
        
        # Serialize the dictionary
        result = self.optimizer._serialize_config_dict(test_dict)
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that all keys are preserved
        self.assertEqual(set(result.keys()), set(test_dict.keys()))
        
        # Check that Enum values are converted to strings
        self.assertEqual(result["enum"], "value1")
        self.assertEqual(result["list"][2], "value2")
        self.assertEqual(result["nested_dict"]["enum_key"], "value2")
        
        # Check that other values remain unchanged
        self.assertEqual(result["string"], "test")
        self.assertEqual(result["int"], 123)
        self.assertEqual(result["float"], 45.67)
        self.assertEqual(result["bool"], True)

    def test_optimization_modes(self):
        """Test that different optimization modes produce different configurations."""
        configs = {}
        
        # Generate a config for each optimization mode
        for mode in OptimizationMode:
            optimizer = DeviceOptimizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path,
                optimization_mode=mode
            )
            
            # Get quantization config for this mode
            quant_config = optimizer.get_optimal_quantization_config()
            configs[mode] = quant_config
        
        # Verify that configs are different for different modes
        # For example, MEMORY_SAVING should have smaller cache size than FULL_UTILIZATION
        self.assertLess(
            configs[OptimizationMode.MEMORY_SAVING].cache_size,
            configs[OptimizationMode.FULL_UTILIZATION].cache_size
        )
        
        # PERFORMANCE should have more CPU usage than CONSERVATIVE
        batch_config_perf = DeviceOptimizer(
            optimization_mode=OptimizationMode.PERFORMANCE
        ).get_optimal_batch_processor_config()
        
        batch_config_cons = DeviceOptimizer(
            optimization_mode=OptimizationMode.CONSERVATIVE
        ).get_optimal_batch_processor_config()
        
        self.assertGreater(
            batch_config_perf.max_workers,
            batch_config_cons.max_workers
        )

    def test_save_configs(self):
        """Test saving configurations to disk."""
        # Generate and save configurations
        config_id = "test_config"
        result = self.optimizer.save_configs(config_id)
        
        # Check that result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("config_id", result)
        self.assertIn("quantization_config", result)
        self.assertIn("batch_processor_config", result)
        self.assertIn("preprocessor_config", result)
        self.assertIn("inference_engine_config", result)
        self.assertIn("training_engine_config", result)
        
        # Check that config files were created
        config_dir = os.path.join(self.config_path, config_id)
        self.assertTrue(os.path.exists(config_dir))
        self.assertTrue(os.path.exists(os.path.join(config_dir, "quantization_config.json")))
        self.assertTrue(os.path.exists(os.path.join(config_dir, "batch_processor_config.json")))
        
        # Load a config and check it's a valid JSON
        quant_config_path = os.path.join(config_dir, "quantization_config.json")
        with open(quant_config_path, "r") as f:
            config_data = json.load(f)
            
        self.assertIsInstance(config_data, dict)
        self.assertIn("quantization_type", config_data)
        self.assertIn("cache_size", config_data)

    def test_create_optimized_configs_helper(self):
        """Test the create_optimized_configs helper function."""
        # Generate configs with a specific mode
        result = create_optimized_configs(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            config_id="test_helper",
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        
        # Check result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["config_id"], "test_helper")
        
        # Verify optimization mode was saved in system info
        system_info_path = Path(result["system_info"])
        self.assertTrue(system_info_path.exists())
        
        with open(system_info_path, "r") as f:
            system_info = json.load(f)
            self.assertEqual(system_info["optimization_mode"], OptimizationMode.PERFORMANCE.value)

    def test_create_configs_for_all_modes(self):
        """Test creating configurations for all optimization modes."""
        # Generate configs for all modes
        result = create_configs_for_all_modes(
            base_config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check result
        self.assertIsInstance(result, dict)
        
        # Check that we have an entry for each mode
        for mode in OptimizationMode:
            self.assertIn(mode.value, result)
            
            # Check that each mode has a valid config
            mode_result = result[mode.value]
            self.assertIsInstance(mode_result, dict)
            self.assertIn("config_id", mode_result)
            
            # Check that the mode-specific directory was created
            mode_dir = os.path.join(self.config_path, mode.value)
            self.assertTrue(os.path.exists(mode_dir))

    def test_load_config(self):
        """Test loading configurations."""
        # Save a configuration
        config_id = "test_load"
        self.optimizer.save_configs(config_id)
        
        # Get path to a config file
        config_dir = os.path.join(self.config_path, config_id)
        quant_config_path = os.path.join(config_dir, "quantization_config.json")
        
        # Load the config
        config = load_config(quant_config_path)
        
        # Check the loaded config
        self.assertIsInstance(config, dict)
        self.assertIn("quantization_type", config)
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_file.json")

    def test_safe_dict_serializer(self):
        """Test the safe_dict_serializer function."""
        import numpy as np
        from datetime import datetime
        
        # Create a complex object with various types
        complex_obj = {
            "int": 123,
            "float": 45.67,
            "str": "test",
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "np_array": np.array([1, 2, 3]),
            "np_int": np.int32(42),
            "np_float": np.float32(3.14),
            "date": datetime(2025, 3, 15),
            "path": Path("/tmp/test"),
            "function": lambda x: x * 2,  # This should be converted to string
            "nested": {
                "key": "value",
                "array": np.array([4, 5, 6])
            }
        }
        
        # Serialize the object
        result = safe_dict_serializer(complex_obj)
        
        # Check that the result is serializable to JSON
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)
        
        # Deserialize the JSON and check values
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["int"], 123)
        self.assertEqual(deserialized["float"], 45.67)
        self.assertEqual(deserialized["str"], "test")
        self.assertEqual(deserialized["bool"], True)
        self.assertIsNone(deserialized["none"])
        self.assertEqual(deserialized["list"], [1, 2, 3])
        self.assertEqual(deserialized["np_array"], [1, 2, 3])
        self.assertEqual(deserialized["np_int"], 42)
        self.assertAlmostEqual(deserialized["np_float"], 3.14, places=5)
        self.assertEqual(deserialized["date"], "2025-03-15T00:00:00")
        self.assertEqual(deserialized["path"], "/tmp/test")
        self.assertIsInstance(deserialized["function"], str)  # Function converted to string
        self.assertEqual(deserialized["nested"]["key"], "value")
        self.assertEqual(deserialized["nested"]["array"], [4, 5, 6])

    def test_quantization_config(self):
        """Test generation of quantization configurations."""
        # Test with different optimization modes
        config_balanced = DeviceOptimizer(
            optimization_mode=OptimizationMode.BALANCED
        ).get_optimal_quantization_config()
        
        config_memory = DeviceOptimizer(
            optimization_mode=OptimizationMode.MEMORY_SAVING
        ).get_optimal_quantization_config()
        
        config_performance = DeviceOptimizer(
            optimization_mode=OptimizationMode.PERFORMANCE
        ).get_optimal_quantization_config()
        
        # Check that configurations are different
        self.assertLess(config_memory.cache_size, config_balanced.cache_size)
        self.assertGreaterEqual(config_performance.buffer_size, config_memory.buffer_size)
        
        # Check type and attributes
        self.assertIsInstance(config_balanced, QuantizationConfig)
        self.assertTrue(config_balanced.enable_cache)
        self.assertEqual(config_balanced.num_bits, 8)

    def test_batch_processor_config(self):
        """Test generation of batch processor configurations."""
        # Test with different optimization modes
        config_balanced = DeviceOptimizer(
            optimization_mode=OptimizationMode.BALANCED
        ).get_optimal_batch_processor_config()
        
        config_full = DeviceOptimizer(
            optimization_mode=OptimizationMode.FULL_UTILIZATION
        ).get_optimal_batch_processor_config()
        
        # Check that configurations are different
        self.assertLess(config_balanced.max_batch_size, config_full.max_batch_size)
        self.assertTrue(config_balanced.enable_adaptive_batching)
        self.assertFalse(config_full.enable_adaptive_batching)
        
        # Check type and attributes
        self.assertIsInstance(config_balanced, BatchProcessorConfig)
        self.assertTrue(config_balanced.enable_priority_queue)
        self.assertTrue(config_balanced.enable_monitoring)

    def test_preprocessor_config(self):
        """Test generation of preprocessor configurations."""
        # Test with different optimization modes
        config_balanced = DeviceOptimizer(
            optimization_mode=OptimizationMode.BALANCED
        ).get_optimal_preprocessor_config()
        
        config_performance = DeviceOptimizer(
            optimization_mode=OptimizationMode.PERFORMANCE
        ).get_optimal_preprocessor_config()
        
        # Check that configurations are different
        self.assertTrue(config_balanced.detect_outliers)
        self.assertFalse(config_performance.detect_outliers)
        
        # Check type and attributes
        self.assertIsInstance(config_balanced, PreprocessorConfig)
        self.assertTrue(config_balanced.handle_nan)
        self.assertTrue(config_balanced.handle_inf)

    def test_inference_engine_config(self):
        """Test generation of inference engine configurations."""
        # Test with different optimization modes
        config_balanced = DeviceOptimizer(
            optimization_mode=OptimizationMode.BALANCED
        ).get_optimal_inference_engine_config()
        
        config_full = DeviceOptimizer(
            optimization_mode=OptimizationMode.FULL_UTILIZATION
        ).get_optimal_inference_engine_config()
        
        # Check that configurations are different
        self.assertEqual(config_balanced.batch_processing_strategy, "adaptive")
        self.assertEqual(config_full.batch_processing_strategy, "greedy")
        
        self.assertTrue(config_balanced.throttle_on_high_cpu)
        self.assertFalse(config_full.throttle_on_high_cpu)
        
        # Check type and attributes
        self.assertIsInstance(config_balanced, InferenceEngineConfig)
        self.assertTrue(config_balanced.enable_quantization)
        self.assertTrue(config_balanced.enable_model_quantization)

    def test_training_engine_config(self):
        """Test generation of training engine configurations."""
        # Test with different optimization modes
        config_conservative = DeviceOptimizer(
            optimization_mode=OptimizationMode.CONSERVATIVE
        ).get_optimal_training_engine_config()
        
        config_performance = DeviceOptimizer(
            optimization_mode=OptimizationMode.PERFORMANCE
        ).get_optimal_training_engine_config()
        
        # Check that configurations are different
        self.assertLess(config_conservative.n_jobs, config_performance.n_jobs)
        self.assertLess(config_conservative.optimization_iterations, config_performance.optimization_iterations)
        
        # Check type and attributes
        self.assertIsInstance(config_conservative, MLTrainingEngineConfig)
        self.assertEqual(config_conservative.random_state, 42)
        self.assertEqual(config_conservative.test_size, 0.2)
        self.assertTrue(config_conservative.stratify)


if __name__ == "__main__":
    unittest.main()