import unittest
from unittest.mock import patch, Mock, mock_open, call
import os
import json
import tempfile
import shutil
import platform
import sys
from pathlib import Path

# Import the module to test
from modules.configs import (
    QuantizationType, QuantizationMode, OptimizationMode, BatchProcessingStrategy,
    NormalizationType
)
# The main class we're testing
from modules.device_optimizer import (
    DeviceOptimizer, HardwareAccelerator, create_optimized_configs,
    create_configs_for_all_modes, load_saved_configs, get_system_information,
    optimize_for_environment, optimize_for_workload, apply_configs_to_pipeline
)


class TestDeviceOptimizer(unittest.TestCase):
    """Test the DeviceOptimizer class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories for configs, checkpoints, and model registry
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "configs")
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoints")
        self.model_registry_path = os.path.join(self.temp_dir, "model_registry")
        
        # Mock system info for consistent testing
        self.psutil_virtual_memory_patcher = patch('psutil.virtual_memory')
        self.mock_virtual_memory = self.psutil_virtual_memory_patcher.start()
        
        memory_mock = Mock()
        memory_mock.total = 8 * (1024 ** 3)  # 8 GB
        memory_mock.available = 4 * (1024 ** 3)  # 4 GB
        memory_mock.percent = 50  # 50% memory usage
        self.mock_virtual_memory.return_value = memory_mock
        
        self.psutil_cpu_count_patcher = patch('psutil.cpu_count')
        self.mock_cpu_count = self.psutil_cpu_count_patcher.start()
        self.mock_cpu_count.side_effect = lambda logical=True: 4 if logical else 2
        
        self.psutil_cpu_freq_patcher = patch('psutil.cpu_freq')
        self.mock_cpu_freq = self.psutil_cpu_freq_patcher.start()
        
        cpu_freq_mock = Mock()
        cpu_freq_mock.current = 2500.0
        cpu_freq_mock.min = 800.0
        cpu_freq_mock.max = 3500.0
        self.mock_cpu_freq.return_value = cpu_freq_mock
        
        self.psutil_disk_usage_patcher = patch('psutil.disk_usage')
        self.mock_disk_usage = self.psutil_disk_usage_patcher.start()
        
        disk_usage_mock = Mock()
        disk_usage_mock.total = 500 * (1024 ** 3)  # 500 GB
        disk_usage_mock.free = 200 * (1024 ** 3)  # 200 GB
        self.mock_disk_usage.return_value = disk_usage_mock
        
        self.psutil_swap_memory_patcher = patch('psutil.swap_memory')
        self.mock_swap_memory = self.psutil_swap_memory_patcher.start()
        
        swap_memory_mock = Mock()
        swap_memory_mock.total = 2 * (1024 ** 3)  # 2 GB
        self.mock_swap_memory.return_value = swap_memory_mock
        
        # Mock platform info
        self.platform_system_patcher = patch('platform.system')
        self.mock_platform_system = self.platform_system_patcher.start()
        self.mock_platform_system.return_value = "Linux"
        
        self.platform_release_patcher = patch('platform.release')
        self.mock_platform_release = self.platform_release_patcher.start()
        self.mock_platform_release.return_value = "5.4.0"
        
        self.platform_machine_patcher = patch('platform.machine')
        self.mock_platform_machine = self.platform_machine_patcher.start()
        self.mock_platform_machine.return_value = "x86_64"
        
        self.platform_processor_patcher = patch('platform.processor')
        self.mock_platform_processor = self.platform_processor_patcher.start()
        self.mock_platform_processor.return_value = "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"
        
        self.socket_hostname_patcher = patch('socket.gethostname')
        self.mock_socket_hostname = self.socket_hostname_patcher.start()
        self.mock_socket_hostname.return_value = "test-host"
        
        self.platform_python_version_patcher = patch('platform.python_version')
        self.mock_platform_python_version = self.platform_python_version_patcher.start()
        self.mock_platform_python_version.return_value = "3.9.0"
        
        # Mock open for CPU info
        self.mock_open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_file = self.mock_open_patcher.start()
        
        cpu_info = """
        processor   : 0
        vendor_id   : GenuineIntel
        cpu family  : 6
        model       : 158
        model name  : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
        stepping    : 10
        microcode   : 0xde
        cpu MHz     : 800.049
        cache size  : 12288 KB
        physical id : 0
        siblings    : 12
        core id     : 0
        cpu cores   : 6
        apicid      : 0
        initial apicid  : 0
        fpu     : yes
        fpu_exception   : yes
        cpuid level : 22
        wp      : yes
        flags       : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d arch_capabilities
        bugs        : cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs taa itlb_multihit
        bogomips    : 5199.98
        clflush size    : 64
        cache_alignment : 64
        address sizes   : 39 bits physical, 48 bits virtual
        power management:
        """
        self.mock_file.return_value.__enter__.return_value.read.return_value = cpu_info
        
        # Mock os.path methods
        self.os_path_exists_patcher = patch('os.path.exists')
        self.mock_os_path_exists = self.os_path_exists_patcher.start()
        self.mock_os_path_exists.return_value = False
        
        self.os_path_ismount_patcher = patch('os.path.ismount')
        self.mock_os_path_ismount = self.os_path_ismount_patcher.start()
        self.mock_os_path_ismount.return_value = True
        
        # Fix the path_mkdir_patcher setup to use a proper mock
        self.mkdir_mock = Mock()
        self.path_mkdir_patcher = patch('pathlib.Path.mkdir', self.mkdir_mock)
        self.path_mkdir_patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.psutil_virtual_memory_patcher.stop()
        self.psutil_cpu_count_patcher.stop()
        self.psutil_cpu_freq_patcher.stop()
        self.psutil_disk_usage_patcher.stop()
        self.psutil_swap_memory_patcher.stop()
        self.platform_system_patcher.stop()
        self.platform_release_patcher.stop()
        self.platform_machine_patcher.stop()
        self.platform_processor_patcher.stop()
        self.socket_hostname_patcher.stop()
        self.platform_python_version_patcher.stop()
        self.mock_open_patcher.stop()
        self.os_path_exists_patcher.stop()
        self.os_path_ismount_patcher.stop()
        self.path_mkdir_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the DeviceOptimizer initializes correctly."""
        # Create the optimizer
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Check that mkdir was called for the three paths
        self.assertEqual(self.mkdir_mock.call_count, 3)
        
        # Check basic attributes
        self.assertEqual(optimizer.config_path, Path(self.config_path))
        self.assertEqual(optimizer.checkpoint_path, Path(self.checkpoint_path))
        self.assertEqual(optimizer.model_registry_path, Path(self.model_registry_path))
        self.assertEqual(optimizer.optimization_mode, OptimizationMode.BALANCED)
        self.assertEqual(optimizer.workload_type, "mixed")
        self.assertFalse(optimizer.power_efficiency)
        
        # Check system detection
        self.assertEqual(optimizer.system, "Linux")
        self.assertEqual(optimizer.release, "5.4.0")
        self.assertEqual(optimizer.machine, "x86_64")
        self.assertEqual(optimizer.processor, "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz")
        self.assertEqual(optimizer.hostname, "test-host")
        self.assertEqual(optimizer.python_version, "3.9.0")
        
        # Check CPU capabilities
        self.assertEqual(optimizer.cpu_count_physical, 2)
        self.assertEqual(optimizer.cpu_count_logical, 4)
        self.assertDictEqual(optimizer.cpu_freq, {"current": 2500.0, "min": 800.0, "max": 3500.0})
        self.assertTrue(optimizer.has_avx)
        self.assertTrue(optimizer.has_avx2)
        
        # Check memory info
        self.assertAlmostEqual(optimizer.total_memory_gb, 8.0)
        self.assertAlmostEqual(optimizer.available_memory_gb, 4.0)
        self.assertAlmostEqual(optimizer.usable_memory_gb, 7.2)  # 8.0 * (1 - 0.1)
        
        # Check disk info
        self.assertAlmostEqual(optimizer.disk_total_gb, 500.0)
        self.assertAlmostEqual(optimizer.disk_free_gb, 200.0)

    @patch('builtins.open', new_callable=mock_open)
    def test_detect_specialized_accelerators(self, mock_file):
        """Test detection of specialized hardware accelerators."""
        # Mock the ctypes CDLL to simulate Intel MKL being available
        with patch('ctypes.CDLL') as mock_cdll:
            # Make ctypes.CDLL succeed to simulate MKL availability
            mock_cdll.return_value = Mock()
            
            optimizer = DeviceOptimizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path,
                enable_specialized_accelerators=True
            )
            
            # Now let's make sure Intel MKL was detected
            self.assertIn(HardwareAccelerator.INTEL_MKL, optimizer.accelerators)
    
    def test_detect_environment(self):
        """Test environment detection."""
        # Test auto-detection for different scenarios
        
        # 1. Desktop (default for our mock settings)
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            environment="auto"
        )
        self.assertEqual(optimizer.environment, "desktop")
        
        # 2. Edge (low resources)
        with patch('psutil.virtual_memory') as mock_vm:
            vm_mock = Mock()
            vm_mock.total = 2 * (1024 ** 3)  # 2 GB
            vm_mock.available = 1 * (1024 ** 3)  # 1 GB
            mock_vm.return_value = vm_mock
            
            with patch('psutil.cpu_count') as mock_cpu:
                mock_cpu.side_effect = lambda logical=True: 2 if logical else 1
                
                optimizer = DeviceOptimizer(
                    config_path=self.config_path,
                    checkpoint_path=self.checkpoint_path,
                    model_registry_path=self.model_registry_path,
                    environment="auto"
                )
                self.assertEqual(optimizer.environment, "edge")
        
        # 3. Cloud (high resources)
        with patch('psutil.virtual_memory') as mock_vm:
            vm_mock = Mock()
            vm_mock.total = 64 * (1024 ** 3)  # 64 GB
            vm_mock.available = 32 * (1024 ** 3)  # 32 GB
            mock_vm.return_value = vm_mock
            
            with patch('psutil.cpu_count') as mock_cpu:
                mock_cpu.side_effect = lambda logical=True: 32 if logical else 16
                
                optimizer = DeviceOptimizer(
                    config_path=self.config_path,
                    checkpoint_path=self.checkpoint_path,
                    model_registry_path=self.model_registry_path,
                    environment="auto"
                )
                self.assertEqual(optimizer.environment, "cloud")
        
        # 4. Manual override
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            environment="edge"
        )
        self.assertEqual(optimizer.environment, "edge")

    def test_get_optimal_quantization_config(self):
        """Test generation of quantization configuration."""
        # Test with different optimization modes
        
        # 1. Performance mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        config = optimizer.get_optimal_quantization_config()
        
        # Check key settings for performance mode
        self.assertEqual(config.quantization_type, QuantizationType.FLOAT16.value)
        self.assertEqual(config.quantization_mode, QuantizationMode.STATIC.value)
        self.assertTrue(config.per_channel)
        self.assertFalse(config.symmetric)
        self.assertTrue(config.enable_cache)
        self.assertEqual(config.weight_bits, 16)
        self.assertEqual(config.activation_bits, 16)
        
        # 2. Memory saving mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.MEMORY_SAVING
        )
        config = optimizer.get_optimal_quantization_config()
        
        # Check key settings for memory saving mode
        self.assertEqual(config.quantization_type, QuantizationType.INT8.value)
        self.assertEqual(config.quantization_mode, QuantizationMode.DYNAMIC.value)
        self.assertEqual(config.weight_bits, 8)
        self.assertEqual(config.activation_bits, 8)
        self.assertTrue(config.optimize_memory)

    def test_get_optimal_preprocessor_config(self):
        """Test generation of preprocessor configuration."""
        # Test with different optimization modes
        
        # 1. Performance mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        config = optimizer.get_optimal_preprocessor_config()
        
        # Check key settings for performance mode
        self.assertEqual(config.normalization, NormalizationType.STANDARD)
        self.assertFalse(config.auto_feature_selection)
        self.assertEqual(config.n_jobs, 4)  # Using all logical cores
        
        # 2. Memory saving mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.MEMORY_SAVING
        )
        config = optimizer.get_optimal_preprocessor_config()
        
        # Check key settings for memory saving mode
        self.assertEqual(config.normalization, NormalizationType.MINMAX)
        self.assertTrue(config.auto_feature_selection)
        self.assertEqual(config.text_max_features, 1000)  # Lower feature count
        self.assertEqual(config.dimension_reduction, "pca")  # Using dimension reduction

    def test_get_optimal_batch_processor_config(self):
        """Test generation of batch processor configuration."""
        # Test with different optimization modes
        
        # 1. Performance mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        config = optimizer.get_optimal_batch_processor_config()
        
        # Check key settings for performance mode
        self.assertEqual(config.batch_timeout, 0.5)  # Shorter timeout
        self.assertEqual(config.processing_strategy, BatchProcessingStrategy.GREEDY)
        self.assertTrue(config.enable_prefetching)
        
        # 2. Memory saving mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.MEMORY_SAVING
        )
        config = optimizer.get_optimal_batch_processor_config()
        
        # Check key settings for memory saving mode
        self.assertEqual(config.batch_timeout, 2.0)  # Longer timeout
        self.assertEqual(config.processing_strategy, BatchProcessingStrategy.FIXED)
        self.assertEqual(config.prefetch_batches, 1)  # Less prefetching

    def test_get_optimal_inference_engine_config(self):
        """Test generation of inference engine configuration."""
        # Test with different optimization modes
        
        # 1. Performance mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.PERFORMANCE
        )
        config = optimizer.get_optimal_inference_engine_config()
        
        # Check key settings for performance mode
        self.assertEqual(config.model_precision, "fp32")
        self.assertTrue(config.enable_intel_optimization)
        self.assertEqual(config.batch_timeout, 0.05)  # Shorter timeout
        self.assertFalse(config.enable_quantization)
        
        # 2. Memory saving mode
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            optimization_mode=OptimizationMode.MEMORY_SAVING
        )
        config = optimizer.get_optimal_inference_engine_config()
        
        # Check key settings for memory saving mode
        self.assertEqual(config.model_precision, "fp16")
        self.assertEqual(config.batch_timeout, 0.1)  # Longer timeout
        self.assertTrue(config.enable_quantization)
        self.assertTrue(config.enable_model_quantization)

    def test_save_configs(self):
        """Test saving configurations to files."""
        # Mock necessary methods and file operations
        with patch('json.dump') as mock_json_dump, \
             patch('builtins.open', mock_open()) as mock_file_open, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Create optimizer with some test settings
            optimizer = DeviceOptimizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path
            )
            
            # Mock the missing method
            optimizer.get_optimal_training_engine_config = Mock()
            optimizer.get_optimal_training_engine_config.return_value = {}
            
            # Set up return values for other methods
            optimizer.get_optimal_quantization_config = Mock(return_value={})
            optimizer.get_optimal_batch_processor_config = Mock(return_value={})
            optimizer.get_optimal_preprocessor_config = Mock(return_value={})
            optimizer.get_optimal_inference_engine_config = Mock(return_value={})
            optimizer.get_system_info = Mock(return_value={})
            
            # Call save_configs with a test config_id
            config_id = "test_config"
            master_config = optimizer.save_configs(config_id)
            
            # Check that the methods were called
            optimizer.get_optimal_quantization_config.assert_called_once()
            optimizer.get_optimal_batch_processor_config.assert_called_once()
            optimizer.get_optimal_preprocessor_config.assert_called_once()
            optimizer.get_optimal_inference_engine_config.assert_called_once()
            optimizer.get_optimal_training_engine_config.assert_called_once()
            optimizer.get_system_info.assert_called_once()
            
            # Check that json.dump was called the expected number of times (7 files)
            # 1 for system_info + 5 for configs + 1 for master config = 7
            self.assertEqual(mock_json_dump.call_count, 7)
            
            # Check that the master config contains expected keys
            self.assertIn("config_id", master_config)
            self.assertEqual(master_config["config_id"], config_id)

    @patch('json.load')
    def test_load_configs(self, mock_json_load):
        """Test loading configurations from files."""
        # Create a DeviceOptimizer instance
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Mock return values for json.load
        mock_master_config = {
            "config_id": "test_config",
            "optimization_mode": "balanced",
            "system_info_path": "path/to/system_info.json",
            "quantization_config_path": "path/to/quantization_config.json",
            "batch_processor_config_path": "path/to/batch_processor_config.json",
            "preprocessor_config_path": "path/to/preprocessor_config.json",
            "inference_engine_config_path": "path/to/inference_engine_config.json",
            "training_engine_config_path": "path/to/training_engine_config.json"
        }
        mock_system_info = {"system": "Linux", "cpu_cores": 4}
        mock_configs = {
            "quantization_config": {"type": "int8"},
            "batch_processor_config": {"batch_size": 32},
            "preprocessor_config": {"normalization": "standard"},
            "inference_engine_config": {"threads": 4},
            "training_engine_config": {"epochs": 10}
        }
        
        # Set up mock_json_load to return different values for different files
        mock_json_load.side_effect = [
            mock_master_config,  # For master_config.json
            mock_system_info,    # For system_info.json
            mock_configs["quantization_config"],
            mock_configs["batch_processor_config"],
            mock_configs["preprocessor_config"],
            mock_configs["inference_engine_config"],
            mock_configs["training_engine_config"]
        ]
        
        # Mock Path.exists to return True for all paths
        with patch.object(Path, 'exists', return_value=True):
            # Call load_configs
            config_id = "test_config"
            loaded_configs = optimizer.load_configs(config_id)
            
            # Check that the correct configs were loaded
            self.assertEqual(loaded_configs["master_config"], mock_master_config)
            self.assertEqual(loaded_configs["system_info"], mock_system_info)
            for config_name in mock_configs:
                self.assertEqual(loaded_configs[config_name], mock_configs[config_name])

    def test_create_configs_for_all_modes(self):
        """Test creating configurations for all optimization modes."""
        # Create a DeviceOptimizer instance
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        # Mock save_configs to avoid actual file operations
        with patch.object(DeviceOptimizer, 'save_configs') as mock_save_configs:
            # Set up mock_save_configs to return a simple dictionary
            mock_save_configs.return_value = {"config_id": "test"}
            
            # Call create_configs_for_all_modes
            configs = optimizer.create_configs_for_all_modes()
            
            # Check that save_configs was called for each optimization mode
            self.assertEqual(mock_save_configs.call_count, len(OptimizationMode))
            
            # Check that the result contains an entry for each mode
            for mode in OptimizationMode:
                self.assertIn(mode.value, configs)

    def test_helper_functions(self):
        """Test the helper functions."""
        # Create proper mocks for all the methods being patched
        with patch.object(DeviceOptimizer, 'save_configs') as mock_save_configs, \
             patch.object(DeviceOptimizer, 'auto_tune_configs') as mock_auto_tune, \
             patch.object(DeviceOptimizer, 'create_configs_for_all_modes') as mock_create_all, \
             patch.object(DeviceOptimizer, 'load_configs') as mock_load, \
             patch.object(DeviceOptimizer, 'get_system_info') as mock_get_system_info:
            
            # Set return values for the mocks
            mock_save_configs.return_value = {"master_config": "path/to/config"}
            mock_auto_tune.return_value = {"master_config": "path/to/auto_tuned_config"}
            mock_create_all.return_value = {"balanced": {"master_config": "path/to/balanced_config"}}
            mock_load.return_value = {"master_config": {}, "quantization_config": {}}
            mock_get_system_info.return_value = {"system": "test", "cpu": {}}
            
            # Call the helper functions
            result1 = create_optimized_configs()
            result2 = create_optimized_configs(auto_tune=True)
            result3 = create_configs_for_all_modes()
            result4 = load_saved_configs("path", "config_id")
            result5 = get_system_information()
            
            # Check that the right methods were called
            mock_save_configs.assert_called_once()
            mock_auto_tune.assert_called_once()
            mock_create_all.assert_called_once()
            mock_load.assert_called_once_with("config_id")
            mock_get_system_info.assert_called_once()
            
            # Check the results
            self.assertEqual(result1, {"master_config": "path/to/config"})
            self.assertEqual(result2, {"master_config": "path/to/auto_tuned_config"})
            self.assertEqual(result3, {"balanced": {"master_config": "path/to/balanced_config"}})
            self.assertEqual(result4, {"master_config": {}, "quantization_config": {}})
            self.assertEqual(result5, {"system": "test", "cpu": {}})

    def test_auto_tune_configs(self):
        """Test auto-tuning of configurations."""
        # 1. Test with auto_tune enabled
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            auto_tune=True
        )
        
        # Mock CPU and memory loads
        with patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('psutil.virtual_memory') as mock_virtual_memory:
            
            mock_cpu_percent.return_value = 50.0  # 50% CPU load
            
            vm_mock = Mock()
            vm_mock.percent = 60.0  # 60% memory load
            mock_virtual_memory.return_value = vm_mock
            
            # Mock save_configs to avoid actual file operations
            with patch.object(DeviceOptimizer, 'save_configs') as mock_save_configs:
                mock_save_configs.return_value = {"config_id": "auto_tuned"}
                
                # Call auto_tune_configs
                result = optimizer.auto_tune_configs()
                
                # Check that save_configs was called with the expected parameter
                mock_save_configs.assert_called_once_with("auto_tuned")
                self.assertEqual(result, {"config_id": "auto_tuned"})
                
                # Verify that CPU load affected the tuning process
                self.assertEqual(mock_cpu_percent.call_count, 1)
                self.assertEqual(mock_virtual_memory.call_count, 1)
        
        # 2. Test with auto_tune disabled
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            auto_tune=False
        )
        
        with patch.object(DeviceOptimizer, 'save_configs') as mock_save_configs:
            mock_save_configs.return_value = {"config_id": "default"}
            
            # Call auto_tune_configs - should fall back to regular save_configs
            result = optimizer.auto_tune_configs()
            
            # Verify save_configs was called with no specific config_id
            mock_save_configs.assert_called_once()
            self.assertEqual(result, {"config_id": "default"})
    
    def test_different_workload_types(self):
        """Test behavior with different workload types."""
        # 1. Test inference workload
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            workload_type="inference"
        )
        
        batch_config = optimizer.get_optimal_batch_processor_config()
        
        # Inference workloads typically have larger batch sizes
        self.assertTrue(batch_config.initial_batch_size >= 32)
        
        # 2. Test training workload
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path,
            workload_type="training"
        )
        
        # Training configs should optimize for memory differently
        quant_config = optimizer.get_optimal_quantization_config()
        self.assertTrue(quant_config.enable_cache)
    
    def test_error_handling(self):
        """Test error handling in DeviceOptimizer."""
        # 1. Test loading non-existent config
        optimizer = DeviceOptimizer(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model_registry_path=self.model_registry_path
        )
        
        with patch.object(Path, 'exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                optimizer.load_configs("non_existent_config")
        
        # 2. Test handling of disk detection errors
        with patch('psutil.disk_usage', side_effect=Exception("Disk error")):
            optimizer = DeviceOptimizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path
            )
            
            # Should default to 0 when detection fails
            self.assertEqual(optimizer.disk_total_gb, 0)
            self.assertEqual(optimizer.disk_free_gb, 0)
        
        # 3. Test handling of CPU frequency errors
        with patch('psutil.cpu_freq', side_effect=Exception("CPU freq error")):
            optimizer = DeviceOptimizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                model_registry_path=self.model_registry_path
            )
            
            # Should default to 0 when detection fails
            self.assertEqual(optimizer.cpu_freq["current"], 0)
            self.assertEqual(optimizer.cpu_freq["min"], 0)
            self.assertEqual(optimizer.cpu_freq["max"], 0)
    
    def test_optimize_for_environment_helper(self):
        """Test the optimize_for_environment helper function."""
        with patch('modules.device_optimizer.create_optimized_configs') as mock_create_configs:
            mock_create_configs.return_value = {"config_id": "cloud_optimized"}
            
            # 1. Test cloud environment
            result = optimize_for_environment("cloud")
            mock_create_configs.assert_called_with(
                optimization_mode=OptimizationMode.PERFORMANCE,
                environment="cloud",
                workload_type="mixed",
                config_id="cloud_optimized"
            )
            self.assertEqual(result, {"config_id": "cloud_optimized"})
            
            # 2. Test desktop environment
            mock_create_configs.reset_mock()
            mock_create_configs.return_value = {"config_id": "desktop_optimized"}
            
            result = optimize_for_environment("desktop")
            mock_create_configs.assert_called_with(
                optimization_mode=OptimizationMode.BALANCED,
                environment="desktop",
                workload_type="mixed",
                config_id="desktop_optimized"
            )
            self.assertEqual(result, {"config_id": "desktop_optimized"})
            
            # 3. Test edge environment
            mock_create_configs.reset_mock()
            mock_create_configs.return_value = {"config_id": "edge_optimized"}
            
            result = optimize_for_environment("edge")
            mock_create_configs.assert_called_with(
                optimization_mode=OptimizationMode.MEMORY_SAVING,
                environment="edge",
                workload_type="mixed",
                config_id="edge_optimized"
            )
            self.assertEqual(result, {"config_id": "edge_optimized"})
            
            # 4. Test invalid environment
            with self.assertRaises(ValueError):
                optimize_for_environment("invalid_environment")
    
    def test_optimize_for_workload_helper(self):
        """Test the optimize_for_workload helper function."""
        with patch('modules.device_optimizer.create_optimized_configs') as mock_create_configs:
            mock_create_configs.return_value = {"config_id": "inference_optimized"}
            
            # 1. Test inference workload
            result = optimize_for_workload("inference")
            mock_create_configs.assert_called_with(
                workload_type="inference",
                config_id="inference_optimized"
            )
            self.assertEqual(result, {"config_id": "inference_optimized"})
            
            # 2. Test training workload
            mock_create_configs.reset_mock()
            mock_create_configs.return_value = {"config_id": "training_optimized"}
            
            result = optimize_for_workload("training")
            mock_create_configs.assert_called_with(
                workload_type="training",
                config_id="training_optimized"
            )
            self.assertEqual(result, {"config_id": "training_optimized"})
            
            # 3. Test invalid workload
            with self.assertRaises(ValueError):
                optimize_for_workload("invalid_workload")
    
    def test_apply_configs_to_pipeline(self):
        """Test applying configurations to a pipeline."""
        # 1. Test successful application
        configs_dict = {
            "master_config": {},
            "quantization_config": {},
            "batch_processor_config": {},
            "preprocessor_config": {},
            "inference_engine_config": {},
            "training_engine_config": {}
        }
        
        # Using a simple mock to avoid logger dependency
        with patch('logging.info') as mock_logging:
            result = apply_configs_to_pipeline(configs_dict)
            self.assertTrue(result)
            self.assertTrue(mock_logging.called)
        
        # 2. Test failure due to missing configurations
        incomplete_configs = {
            "master_config": {},
            "quantization_config": {}
            # Missing other required configs
        }
        
        with patch('logging.error') as mock_logging_error:
            result = apply_configs_to_pipeline(incomplete_configs)
            self.assertFalse(result)
            self.assertTrue(mock_logging_error.called)
        
        # 3. Test failure due to exception
        with patch('logging.error') as mock_logging_error:
            with patch('logging.info', side_effect=Exception("Pipeline error")):
                result = apply_configs_to_pipeline(configs_dict)
                self.assertFalse(result)
                self.assertTrue(mock_logging_error.called)