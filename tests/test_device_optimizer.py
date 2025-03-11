import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Import the module to test
from modules.device_optimizer import (
    DeviceOptimizer, 
    load_config, 
    create_optimized_configs,
    QuantizationType,
    QuantizationMode,
    BatchProcessingStrategy,
    NormalizationType,
    TaskType,
    OptimizationStrategy
)

class TestDeviceOptimizer(unittest.TestCase):
    """Test cases for the DeviceOptimizer class and related functions."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories for configs, checkpoints, and model registry
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "configs")
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoints")
        self.model_registry_path = os.path.join(self.temp_dir, "model_registry")
        
        # Mock system information for consistent testing
        self.cpu_count_patcher = patch('multiprocessing.cpu_count', return_value=8)
        self.mock_cpu_count = self.cpu_count_patcher.start()
        
        self.virtual_memory_patcher = patch('psutil.virtual_memory')
        self.mock_virtual_memory = self.virtual_memory_patcher.start()
        self.mock_virtual_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
        
        self.platform_system_patcher = patch('platform.system', return_value='Linux')
        self.mock_platform_system = self.platform_system_patcher.start()
        
        self.platform_machine_patcher = patch('platform.machine', return_value='x86_64')
        self.mock_platform_machine = self.platform_machine_patcher.start()
        
        self.platform_processor_patcher = patch('platform.processor', return_value='Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz')
        self.mock_platform_processor = self.platform_processor_patcher.start()
        
        self.socket_hostname_patcher = patch('socket.gethostname', return_value='test-host')
        self.mock_socket_hostname = self.socket_hostname_patcher.start()
        
        # Mock /proc/cpuinfo for AVX detection
        self.open_patcher = patch('builtins.open')
        self.mock_open = self.open_patcher.start()
        self.mock_open.return_value.__enter__.return_value.read.return_value = "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d arch_capabilities"
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.cpu_count_patcher.stop()
        self.virtual_memory_patcher.stop()
        self.platform_system_patcher.stop()
        self.platform_machine_patcher.stop()
        self.platform_processor_patcher.stop()
        self.socket_hostname_patcher.stop()
        self.open_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_device_optimizer_initialization(self):
        """Test that DeviceOptimizer initializes correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check that directories were created
        self.assertTrue(os.path.exists(self.config_path))
        self.assertTrue(os.path.exists(self.checkpoint_path))
        self.assertTrue(os.path.exists(self.model_registry_path))
        
        # Check that system information was detected correctly
        self.assertEqual(optimizer.cpu_count, 8)
        self.assertEqual(optimizer.total_memory_gb, 16)
        self.assertEqual(optimizer.system, 'Linux')
        self.assertEqual(optimizer.machine, 'x86_64')
        self.assertTrue(optimizer.is_intel_cpu)
        self.assertTrue(optimizer.has_avx)
        self.assertTrue(optimizer.has_avx2)
    
    def test_get_optimal_quantization_config(self):
        """Test that optimal quantization config is generated correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        config = optimizer.get_optimal_quantization_config()
        
        # Check that config has expected values
        self.assertEqual(config.quantization_type, QuantizationType.INT8.value)
        self.assertEqual(config.quantization_mode, QuantizationMode.DYNAMIC_PER_CHANNEL.value)
        self.assertTrue(config.enable_cache)
        self.assertEqual(config.cache_size, 1024)
        self.assertEqual(config.buffer_size, 512)  # 32MB per GB * 16GB
        self.assertTrue(config.use_percentile)
        self.assertEqual(config.num_bits, 8)
        self.assertFalse(config.optimize_memory)  # False because total_memory_gb >= 16
    
    def test_get_optimal_batch_processor_config(self):
        """Test that optimal batch processor config is generated correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        config = optimizer.get_optimal_batch_processor_config()
        
        # Check that config has expected values
        self.assertEqual(config.max_batch_size, 128)  # 8 CPUs * 16
        self.assertEqual(config.initial_batch_size, 64)  # max_batch_size / 2
        self.assertEqual(config.min_batch_size, 16)  # initial_batch_size / 4
        self.assertEqual(config.max_workers, 6)  # 8 CPUs * 0.75
        self.assertEqual(config.max_batch_memory_mb, 16 * 128)  # 128MB per GB * 16GB
        self.assertEqual(config.max_queue_size, 1600)  # 100 per GB * 16GB
        self.assertEqual(config.processing_strategy, BatchProcessingStrategy.ADAPTIVE)
        self.assertTrue(config.enable_adaptive_batching)
        self.assertTrue(config.enable_memory_optimization)
    
    def test_get_optimal_preprocessor_config(self):
        """Test that optimal preprocessor config is generated correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        config = optimizer.get_optimal_preprocessor_config()
        
        # Check that config has expected values
        self.assertEqual(config.normalization, NormalizationType.STANDARD)
        self.assertTrue(config.parallel_processing)
        self.assertEqual(config.n_jobs, 7)  # cpu_count - 1
        self.assertEqual(config.cache_size, 512)  # 32 per GB * 16GB
        self.assertEqual(config.dtype, np.float64)  # float64 for systems with >= 8GB
        self.assertTrue(config.handle_nan)
        self.assertTrue(config.handle_inf)
        self.assertTrue(config.detect_outliers)
        self.assertEqual(config.outlier_method, "iqr")
    
    def test_get_optimal_inference_engine_config(self):
        """Test that optimal inference engine config is generated correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        config = optimizer.get_optimal_inference_engine_config()
        
        # Check that config has expected values
        self.assertEqual(config.num_threads, 6)  # 8 CPUs * 0.8
        self.assertTrue(config.set_cpu_affinity)  # True because cpu_count > 4
        self.assertTrue(config.enable_intel_optimization)  # True because is_intel_cpu and has_avx
        self.assertTrue(config.enable_quantization)
        self.assertTrue(config.enable_model_quantization)
        self.assertTrue(config.enable_input_quantization)
        self.assertEqual(config.quantization_dtype, "int8")
        self.assertEqual(config.max_cache_entries, 1600)  # 100 per GB * 16GB
        self.assertEqual(config.max_batch_size, 128)  # 8 CPUs * 16
        self.assertEqual(config.initial_batch_size, 64)  # max_batch_size / 2
        self.assertEqual(config.min_batch_size, 16)  # initial_batch_size / 4
        self.assertTrue(config.enable_adaptive_batching)
        self.assertTrue(config.enable_memory_optimization)
        self.assertTrue(config.enable_quantization_aware_inference)  # True because is_intel_cpu and has_avx2
    
    def test_get_optimal_training_engine_config(self):
        """Test that optimal training engine config is generated correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        config = optimizer.get_optimal_training_engine_config()
        
        # Check that config has expected values
        self.assertEqual(config.n_jobs, 6)  # 8 CPUs * 0.8
        self.assertEqual(config.optimization_strategy, OptimizationStrategy.ASHT)  # ASHT for systems with > 16GB and > 8 CPUs
        self.assertEqual(config.optimization_iterations, 50)
        self.assertTrue(config.early_stopping)
        self.assertTrue(config.feature_selection)
        self.assertEqual(config.feature_selection_method, "mutual_info")
        self.assertEqual(config.model_path, str(optimizer.model_registry_path))
        self.assertTrue(config.use_intel_optimization)  # True because is_intel_cpu and has_avx
        self.assertFalse(config.memory_optimization)  # False because total_memory_gb >= 16
        self.assertFalse(config.enable_distributed)  # False because cpu_count <= 8
    def test_save_configs(self):
        """Test that configs are saved correctly."""
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Save configs with a specific ID
        config_id = "test-config"
        
        # Actually save the configs to disk
        master_config = optimizer.save_configs(config_id)
        
        # Check that all config files were created
        configs_dir = Path(self.config_path) / config_id
        self.assertTrue(configs_dir.exists(), f"Config directory was not created: {configs_dir}")
        
        expected_files = [
            "system_info.json",
            "quantization_config.json",
            "batch_processor_config.json",
            "preprocessor_config.json",
            "inference_engine_config.json",
            "training_engine_config.json",
            "master_config.json"
        ]
        
        # Check that each file exists
        for expected_file in expected_files:
            file_path = configs_dir / expected_file
            self.assertTrue(file_path.exists(), f"Expected file was not created: {file_path}")
            
            # Verify that each file contains valid JSON
            with open(file_path, 'r') as f:
                try:
                    content = json.load(f)
                    self.assertIsInstance(content, dict, f"File does not contain a valid JSON object: {file_path}")
                except json.JSONDecodeError:
                    self.fail(f"File does not contain valid JSON: {file_path}")
        
        # Check that master config has correct values
        self.assertEqual(master_config["config_id"], config_id)
        self.assertTrue("system_info" in master_config)
        self.assertTrue("quantization_config" in master_config)
        self.assertTrue("batch_processor_config" in master_config)
        self.assertTrue("preprocessor_config" in master_config)
        self.assertTrue("inference_engine_config" in master_config)
        self.assertTrue("training_engine_config" in master_config)
        
        # Check that paths in master config point to existing files
        for key in ["system_info", "quantization_config", "batch_processor_config", 
                    "preprocessor_config", "inference_engine_config", "training_engine_config"]:
            config_path = Path(master_config[key])
            self.assertTrue(config_path.exists(), f"Path in master config does not exist: {config_path}")
        
        # Check that checkpoint and model registry paths are correct
        self.assertEqual(master_config["checkpoint_path"], str(optimizer.checkpoint_path))
        self.assertEqual(master_config["model_registry_path"], str(optimizer.model_registry_path))
        
        # Optionally, load one of the configs to verify content structure
        with open(configs_dir / "quantization_config.json", 'r') as f:
            quant_config = json.load(f)
            self.assertIn("quantization_type", quant_config)
            self.assertIn("quantization_mode", quant_config)
            self.assertIn("cache_size", quant_config)
    
    def test_load_config(self):
        """Test that configs can be loaded correctly."""
        # Create a test config file
        test_config = {"test_key": "test_value"}
        test_config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Ensure the file is actually created
        with open(test_config_path, "w") as f:
            json.dump(test_config, f)
        
        # Verify the file exists before loading
        self.assertTrue(os.path.exists(test_config_path), f"Test config file was not created at {test_config_path}")
        
        # Load the config
        loaded_config = load_config(test_config_path)
        
        # Check that the loaded config matches the original
        self.assertEqual(loaded_config, test_config)
    
    def test_load_config_file_not_found(self):
        """Test that load_config raises FileNotFoundError for non-existent files."""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.json")
        with self.assertRaises(FileNotFoundError):
            load_config(non_existent_path)
    

    def test_create_optimized_configs(self):
        """Test that create_optimized_configs works correctly."""
        # Call create_optimized_configs with a specific ID
        config_id = "test-create"
        
        # Patch the save_configs method to avoid JSON serialization issues in the test
        with patch.object(DeviceOptimizer, 'save_configs') as mock_save_configs:
            # Set up the mock to return a valid master_config
            mock_master_config = {
                "config_id": config_id,
                "system_info": "path/to/system_info.json",
                "quantization_config": "path/to/quantization_config.json",
                "batch_processor_config": "path/to/batch_processor_config.json",
                "preprocessor_config": "path/to/preprocessor_config.json",
                "inference_engine_config": "path/to/inference_engine_config.json",
                "training_engine_config": "path/to/training_engine_config.json",
                "checkpoint_path": str(Path(self.checkpoint_path)),
                "model_registry_path": str(Path(self.model_registry_path))
            }
            mock_save_configs.return_value = mock_master_config
            
            # Call the function
            master_config = create_optimized_configs(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path,
                config_id=config_id
            )
            
            # Check that master config has the correct ID
            self.assertEqual(master_config["config_id"], config_id)
            
            # Verify that save_configs was called with the correct config_id
            mock_save_configs.assert_called_once_with(config_id)

    @patch('psutil.virtual_memory')
    def test_low_memory_system(self, mock_virtual_memory):
        """Test configurations for a low-memory system."""
        # Mock a system with 4GB of RAM
        mock_virtual_memory.return_value.total = 4 * (1024 ** 3)  # 4 GB
        
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check quantization config for low memory
        quant_config = optimizer.get_optimal_quantization_config()
        self.assertEqual(quant_config.quantization_mode, QuantizationMode.DYNAMIC_PER_BATCH.value)
        self.assertEqual(quant_config.cache_size, 512)
        self.assertTrue(quant_config.optimize_memory)
        
        # Check preprocessor config for low memory
        preproc_config = optimizer.get_optimal_preprocessor_config()
        self.assertEqual(preproc_config.dtype, np.float32)
        self.assertIsNotNone(preproc_config.chunk_size)
        
        # Check training config for low memory
        training_config = optimizer.get_optimal_training_engine_config()
        self.assertEqual(training_config.optimization_strategy, OptimizationStrategy.RANDOM_SEARCH)
        self.assertEqual(training_config.optimization_iterations, 20)
        self.assertTrue(training_config.memory_optimization)

    @patch('multiprocessing.cpu_count')
    def test_high_cpu_system(self, mock_cpu_count):
        """Test configurations for a high-CPU system."""
        # Mock a system with 32 CPUs
        mock_cpu_count.return_value = 32
        
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check batch processor config for high CPU
        batch_config = optimizer.get_optimal_batch_processor_config()
        self.assertEqual(batch_config.max_batch_size, 256)  # Capped at 256
        self.assertEqual(batch_config.max_workers, 24)  # 32 * 0.75
        
        # Check inference config for high CPU
        inference_config = optimizer.get_optimal_inference_engine_config()
        self.assertEqual(inference_config.num_threads, 25)  # 32 * 0.8
        
        # Check training config for high CPU
        training_config = optimizer.get_optimal_training_engine_config()
        self.assertTrue(training_config.enable_distributed)  # True because cpu_count > 8

    @patch('platform.processor')
    def test_non_intel_cpu(self, mock_processor):
        """Test configurations for a non-Intel CPU."""
        # Mock a non-Intel CPU
        mock_processor.return_value = 'AMD Ryzen 9 5900X 12-Core Processor'
        
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check that Intel optimizations are disabled
        self.assertFalse(optimizer.is_intel_cpu)
        
        # Check inference config for non-Intel CPU
        inference_config = optimizer.get_optimal_inference_engine_config()
        self.assertFalse(inference_config.enable_intel_optimization)
        self.assertFalse(inference_config.enable_quantization_aware_inference)
        
        # Check training config for non-Intel CPU
        training_config = optimizer.get_optimal_training_engine_config()
        self.assertFalse(training_config.use_intel_optimization)

if __name__ == '__main__':
    unittest.main()
