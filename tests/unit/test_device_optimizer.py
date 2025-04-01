import unittest
from unittest import mock
import os
import tempfile
import shutil
import json
import sys
from pathlib import Path

# Import the modules to be tested
from modules.configs import (
    OptimizationMode, TaskType, BatchProcessingStrategy, 
    NormalizationType, QuantizationType, QuantizationMode,
    OptimizationStrategy, ModelSelectionCriteria, AutoMLMode
)

# Import the class to be tested
from modules.device_optimizer import (
    DeviceOptimizer, HardwareAccelerator, GPUInfo,
    create_optimized_configs, load_saved_configs
)


class TestDeviceOptimizer(unittest.TestCase):
    """Tests for the DeviceOptimizer class"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create temporary directories for configs, checkpoints, and model registry
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "configs")
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoints")
        self.model_registry_path = os.path.join(self.temp_dir, "model_registry")
        
        # Create the common patcher for hardware detection methods
        # This allows tests to run in any environment by mocking hardware detection
        
        # CPU patchers
        self.cpu_count_patcher = mock.patch('psutil.cpu_count', return_value=4)
        self.cpu_freq_patcher = mock.patch.object(
            DeviceOptimizer, '_get_cpu_frequency', 
            return_value={"current": 2.5, "min": 1.0, "max": 3.5}
        )
        
        # Memory patchers
        self.virtual_memory_patcher = mock.patch(
            'psutil.virtual_memory', 
            return_value=mock.MagicMock(
                total=16 * (1024 ** 3),  # 16 GB
                available=8 * (1024 ** 3)  # 8 GB
            )
        )
        
        # Disk patchers
        self.disk_usage_patcher = mock.patch(
            'psutil.disk_usage',
            return_value=mock.MagicMock(
                total=500 * (1024 ** 3),  # 500 GB
                free=250 * (1024 ** 3)  # 250 GB
            )
        )
        
        # GPU detection patchers
        self.detect_gpu_patcher = mock.patch.object(
            DeviceOptimizer, '_detect_gpu_capabilities',
            side_effect=self._mock_gpu_detection
        )
        
        # Start all patchers
        self.cpu_count_patcher.start()
        self.cpu_freq_patcher.start()
        self.virtual_memory_patcher.start()
        self.disk_usage_patcher.start()
        self.detect_gpu_patcher.start()
        
        # Additional mock attributes for the optimizer
        self.cpu_patcher = mock.patch.multiple(
            DeviceOptimizer,
            has_avx=True,
            has_avx2=True,
            has_avx512=False,
            has_sse4=True,
            has_fma=True,
            is_intel_cpu=True,
            is_amd_cpu=False,
            is_arm_cpu=False,
            has_neon=False,
            is_ssd=True
        )
        self.cpu_patcher.start()

    def tearDown(self):
        """Clean up test environment after each test"""
        # Stop all patchers
        self.cpu_count_patcher.stop()
        self.cpu_freq_patcher.stop()
        self.virtual_memory_patcher.stop()
        self.disk_usage_patcher.stop()
        self.detect_gpu_patcher.stop()
        self.cpu_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def _mock_gpu_detection(self):
        """Mock GPU detection to simulate a system with a GPU"""
        optimizer = self._get_current_instance()
        if optimizer:
            optimizer.gpus = [
                GPUInfo(
                    gpu_id=0,
                    name="NVIDIA GeForce RTX 3080",
                    memory_mb=10240,  # 10 GB
                    compute_capability="8.6"
                )
            ]
            optimizer.has_cuda = True
            optimizer.has_rocm = False
    
    def _get_current_instance(self):
        """Helper method to get the current DeviceOptimizer instance from the mock"""
        for mock_call in self.detect_gpu_patcher.mock.mock_calls:
            if mock_call[0] == '()':
                return mock_call[1][0]  # Return the 'self' argument
        return None

    def test_initialization(self):
        """Test that the DeviceOptimizer can be initialized with default parameters"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check that the optimizer was initialized with expected values
        self.assertEqual(optimizer.optimization_mode, OptimizationMode.BALANCED)
        self.assertEqual(optimizer.workload_type, "mixed")
        self.assertTrue(optimizer.auto_tune)
        
        # Check that the directories were created
        self.assertTrue(os.path.exists(self.config_path))
        self.assertTrue(os.path.exists(self.checkpoint_path))
        self.assertTrue(os.path.exists(self.model_registry_path))
        
        # Check that hardware detection methods were called
        self.assertEqual(len(optimizer.gpus), 1)
        self.assertEqual(optimizer.gpus[0].name, "NVIDIA GeForce RTX 3080")

    def test_system_info_generation(self):
        """Test that system information can be generated correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        system_info = optimizer.get_system_info()
        
        # Check the structure of the system info dictionary
        self.assertIn("system", system_info)
        self.assertIn("cpu", system_info)
        self.assertIn("memory", system_info)
        self.assertIn("disk", system_info)
        self.assertIn("gpu", system_info)
        
        # Check specific values
        self.assertEqual(system_info["cpu"]["physical_cores"], 4)
        self.assertEqual(system_info["cpu"]["logical_cores"], 4)
        self.assertEqual(system_info["gpu"]["count"], 1)
        self.assertTrue(system_info["gpu"]["has_cuda"])
        self.assertFalse(system_info["gpu"]["has_rocm"])

    def test_quantization_config_generation(self):
        """Test that quantization configuration can be generated correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        quant_config = optimizer.get_optimal_quantization_config()
        
        # Check that the configuration has the expected attributes
        self.assertIsNotNone(quant_config.quantization_type)
        self.assertIsNotNone(quant_config.quantization_mode)
        self.assertIsInstance(quant_config.per_channel, bool)
        self.assertIsInstance(quant_config.cache_size, int)
        self.assertIsInstance(quant_config.buffer_size, int)

    def test_batch_processor_config_generation(self):
        """Test that batch processor configuration can be generated correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        batch_config = optimizer.get_optimal_batch_processor_config()
        
        # Check that the configuration has the expected attributes
        self.assertIsInstance(batch_config.initial_batch_size, int)
        self.assertIsInstance(batch_config.min_batch_size, int)
        self.assertIsInstance(batch_config.max_batch_size, int)
        self.assertIsInstance(batch_config.num_workers, int)
        
        # Check that batch sizes are reasonable
        self.assertGreater(batch_config.max_batch_size, batch_config.initial_batch_size)
        self.assertGreater(batch_config.initial_batch_size, batch_config.min_batch_size)

    def test_preprocessor_config_generation(self):
        """Test that preprocessor configuration can be generated correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        preproc_config = optimizer.get_optimal_preprocessor_config()
        
        # Check that the configuration has the expected attributes
        self.assertIsNotNone(preproc_config.normalization)
        self.assertIsInstance(preproc_config.handle_nan, bool)
        self.assertIsInstance(preproc_config.handle_inf, bool)
        self.assertIsInstance(preproc_config.n_jobs, int)

    def test_inference_engine_config_generation(self):
        """Test that inference engine configuration can be generated correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        inference_config = optimizer.get_optimal_inference_engine_config()
        
        # Check that the configuration has the expected attributes
        self.assertIsInstance(inference_config.enable_batching, bool)
        self.assertIsInstance(inference_config.max_batch_size, int)
        self.assertIsNotNone(inference_config.model_precision)
        self.assertIsInstance(inference_config.thread_count, int)

    def test_training_engine_config_generation(self):
        """Test that training engine configuration can be generated correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        training_config = optimizer.get_optimal_training_engine_config()
        
        # Check that the configuration has the expected attributes
        self.assertEqual(training_config.task_type, TaskType.CLASSIFICATION)
        self.assertIsInstance(training_config.n_jobs, int)
        self.assertIsInstance(training_config.cv_folds, int)
        self.assertIsInstance(training_config.test_size, float)
        
        # Check nested configurations
        self.assertIsNotNone(training_config.preprocessing_config)
        self.assertIsNotNone(training_config.batch_processing_config)
        self.assertIsNotNone(training_config.inference_config)
        self.assertIsNotNone(training_config.quantization_config)

    def test_save_configs(self):
        """Test that configurations can be saved to files"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Save configurations with a specific ID
        config_id = "test_config"
        master_config = optimizer.save_configs(config_id)
        
        # Check that the master config contains the expected paths
        self.assertIn("config_id", master_config)
        self.assertEqual(master_config["config_id"], config_id)
        self.assertIn("optimization_mode", master_config)
        
        # Check that the config files were created
        configs_dir = Path(self.config_path) / config_id
        self.assertTrue(configs_dir.exists())
        
        expected_files = [
            "master_config.json",
            "system_info.json",
            "quantization_config.json",
            "batch_processor_config.json",
            "preprocessor_config.json",
            "inference_engine_config.json",
            "training_engine_config.json"
        ]
        
        for filename in expected_files:
            file_path = configs_dir / filename
            self.assertTrue(file_path.exists(), f"Missing expected file: {filename}")
            
            # Check that the file contains valid JSON
            with open(file_path, "r") as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)

    def test_load_configs(self):
        """Test that saved configurations can be loaded correctly"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Save configurations with a specific ID
        config_id = "test_load_config"
        optimizer.save_configs(config_id)
        
        # Load the saved configurations
        loaded_configs = optimizer.load_configs(config_id)
        
        # Check that the loaded configs contain the expected sections
        self.assertIn("master_config", loaded_configs)
        self.assertIn("system_info", loaded_configs)
        self.assertIn("quantization_config", loaded_configs)
        self.assertIn("batch_processor_config", loaded_configs)
        self.assertIn("preprocessor_config", loaded_configs)
        self.assertIn("inference_engine_config", loaded_configs)
        self.assertIn("training_engine_config", loaded_configs)
        
        # Check that the config ID matches
        self.assertEqual(loaded_configs["master_config"]["config_id"], config_id)

    def test_optimization_modes(self):
        """Test that different optimization modes produce different configurations"""
        # Create optimizers with different optimization modes
        balanced_optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.BALANCED
        )
        
        performance_optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        
        memory_optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.MEMORY_SAVING
        )
        
        # Get batch configs for comparison
        balanced_batch = balanced_optimizer.get_optimal_batch_processor_config()
        performance_batch = performance_optimizer.get_optimal_batch_processor_config()
        memory_batch = memory_optimizer.get_optimal_batch_processor_config()
        
        # Performance mode should have larger batch sizes than balanced
        self.assertGreaterEqual(performance_batch.max_batch_size, balanced_batch.max_batch_size)
        
        # Memory saving mode should have smaller batch sizes than balanced
        self.assertLessEqual(memory_batch.max_batch_size, balanced_batch.max_batch_size)

    def test_create_configs_for_all_modes(self):
        """Test that configurations for all optimization modes can be created"""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        all_configs = optimizer.create_configs_for_all_modes()
        
        # Check that configs for all modes were created
        for mode in OptimizationMode:
            self.assertIn(mode.value, all_configs)
            self.assertIsInstance(all_configs[mode.value], dict)
            self.assertIn("optimization_mode", all_configs[mode.value])
            self.assertEqual(all_configs[mode.value]["optimization_mode"], mode.value)

    def test_hardware_dependent_configs(self):
        """Test that configurations adapt to available hardware"""
        # Configure with GPU
        with mock.patch.object(DeviceOptimizer, '_detect_gpu_capabilities', 
                              side_effect=self._mock_gpu_detection):
            gpu_optimizer = DeviceOptimizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path
            )
            
            # Check that GPU-specific options are enabled
            inference_config = gpu_optimizer.get_optimal_inference_engine_config()
            self.assertTrue(len(gpu_optimizer.gpus) > 0)
            self.assertTrue(gpu_optimizer.has_cuda)
            
            # Mock a system without GPU
            with mock.patch.object(DeviceOptimizer, '_detect_gpu_capabilities', 
                                  side_effect=lambda: None):
                cpu_optimizer = DeviceOptimizer(
                    config_path=self.config_path,
                    checkpoint_path=self.checkpoint_path,
                    model_registry_path=self.model_registry_path
                )
                
                # Check that GPU-specific options are not enabled
                cpu_inference_config = cpu_optimizer.get_optimal_inference_engine_config()
                
                # CPU optimizer should have its GPU list cleared by the mock
                self.assertEqual(len(cpu_optimizer.gpus), 0)
                
                # We cannot directly compare inference configs as the mocks don't properly reset
                # the internal state between tests, but we can check that training configs adapt
                gpu_training = gpu_optimizer.get_optimal_training_engine_config()
                cpu_training = cpu_optimizer.get_optimal_training_engine_config()
                
                # GPU-enabled system should have different training config
                self.assertNotEqual(gpu_training.use_gpu, cpu_training.use_gpu)

    def test_helper_functions(self):
        """Test that the helper functions work correctly"""
        # Test create_optimized_configs
        with mock.patch('paste.DeviceOptimizer') as mock_optimizer:
            mock_instance = mock.MagicMock()
            mock_optimizer.return_value = mock_instance
            mock_instance.save_configs.return_value = {"test": "config"}
            
            # Call the helper function
            result = create_optimized_configs(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path,
                auto_tune=False
            )
            
            # Check that the optimizer was created with the right parameters
            mock_optimizer.assert_called_once()
            self.assertEqual(result, {"test": "config"})
        
        # Test load_saved_configs
        with mock.patch('paste.DeviceOptimizer') as mock_optimizer:
            mock_instance = mock.MagicMock()
            mock_optimizer.return_value = mock_instance
            mock_instance.load_configs.return_value = {"loaded": "config"}
            
            # Call the helper function
            result = load_saved_configs(
                config_path=self.config_path,
                config_id="test_id"
            )
            
            # Check that load_configs was called with the right parameters
            mock_instance.load_configs.assert_called_once_with("test_id")
            self.assertEqual(result, {"loaded": "config"})


if __name__ == '__main__':
    unittest.main()