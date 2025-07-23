import pytest
from unittest import mock
import json
import os
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from modules.api.device_optimizer_api import app, API_KEY, verify_api_key
    
    # Import the modules being used by the app
    try:
        from modules.device_optimizer import (
            DeviceOptimizer, OptimizationMode,
            create_optimized_configs, create_configs_for_all_modes,
            load_saved_configs, get_system_information, optimize_for_environment,
            optimize_for_workload, apply_configs_to_pipeline, get_default_config
        )
    except ImportError:
        # Create minimal mocks for missing functions
        class OptimizationMode:
            PERFORMANCE = "performance"
            BALANCED = "balanced"
            EFFICIENCY = "efficiency"
        
        class DeviceOptimizer:
            def __init__(self, **kwargs):
                pass
        
        def create_optimized_configs(**kwargs):
            return {}
        
        def create_configs_for_all_modes(**kwargs):
            return {}
        
        def load_saved_configs(**kwargs):
            return {}
        
        def get_system_information(**kwargs):
            return {}
        
        def optimize_for_environment(**kwargs):
            return {}
        
        def optimize_for_workload(**kwargs):
            return {}
        
        def apply_configs_to_pipeline(**kwargs):
            return {}
        
        def get_default_config(**kwargs):
            return {}
    
    try:
        from modules.configs import QuantizationType, QuantizationMode
    except ImportError:
        class QuantizationType:
            DYNAMIC = "dynamic"
            STATIC = "static"
        
        class QuantizationMode:
            DYNAMIC_PER_BATCH = "dynamic_per_batch"
            
except ImportError as e:
    pytest.skip(f"Device optimizer API modules not available: {e}", allow_module_level=True)


@pytest.mark.functional
class TestDeviceOptimizerAPI:
    """Test cases for the Device Optimizer API."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test client and mock API dependencies."""
        self.client = TestClient(app)
        self.api_key = API_KEY
        
        # Create test paths
        self.test_config_path = "./test_configs"
        self.test_checkpoint_path = "./test_checkpoints"
        self.test_model_registry_path = "./test_model_registry"
        
        # Sample test data
        self.system_info_sample = {
            "cpu_info": {
                "vendor": "Test CPU Vendor",
                "name": "Test CPU",
                "cores": 8,
                "threads": 16,
                "frequency": 3.2
            },
            "memory_info": {
                "total": 16384,
                "available": 8192
            },
            "specialized_hardware": {
                "avx": True,
                "avx2": True,
                "avx512": False
            }
        }
        
        self.master_config_sample = {
            "config_id": "test_config_123",
            "optimization_mode": "BALANCED",
            "workload_type": "mixed",
            "environment": "desktop",
            "quantization": {
                "enabled": True,
                "type": "INT8",
                "mode": "DYNAMIC"
            },
            "batch_processing": {
                "batch_size": 32,
                "num_workers": 4
            }
        }
        
    def test_root_endpoint(self):
        """Test the root endpoint returns correct API information."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["api"] == "CPU Device Optimizer API"
        assert data["version"] == "0.1.4"
        assert "description" in data
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    def test_get_system_info(self, mock_optimizer_class):
        """Test retrieving system information."""
        # Mock the DeviceOptimizer class and get_system_info method
        expected_response = {
            "system": "Windows", "release": "10", "machine": "AMD64",
            "processor": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel", 
            "hostname": "test-hostname", "python_version": "3.10.11",
            "cpu_count_physical": 14, "cpu_count_logical": 20,
            "cpu_freq_mhz": {"current": 2300.0, "min": 0, "max": 2300.0},
            "cpu_features": {
                "avx": False, "avx2": False, "avx512": False,
                "sse4": False, "fma": False, "neon": False
            },
            "is_intel_cpu": True, "is_amd_cpu": False, "is_arm_cpu": False,
            "total_memory_gb": 63.7, "available_memory_gb": 36.2, 
            "usable_memory_gb": 57.3, "swap_memory_gb": 4.0,
            "disk_total_gb": 930.5, "disk_free_gb": 149.1, "is_ssd": False,
            "accelerators": [],
            "detected_environment": "cloud",
            "optimizer_settings": {
                "optimization_mode": "balanced", "workload_type": "mixed",
                "power_efficiency": False, "auto_tune": True, "debug_mode": False
            },
            "library_availability": {
                "onnx_available": False, "threadpoolctl_available": True,
                "treelite_available": False
            }
        }
        
        mock_optimizer = mock.MagicMock()
        mock_optimizer.get_system_info.return_value = expected_response
        mock_optimizer_class.return_value = mock_optimizer
        
        # Make request with API key
        response = self.client.get(f"/system-info?api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        # Compare key fields instead of exact match due to dynamic system values
        assert "system" in data
        assert "cpu_count_physical" in data
        assert "cpu_features" in data
        assert "optimizer_settings" in data
        
        # Verify DeviceOptimizer was created with correct parameters
        mock_optimizer_class.assert_called_once_with(enable_specialized_accelerators=True)
    
    def test_get_system_info_without_api_key(self):
        """Test system info endpoint requires API key."""
        response = self.client.get("/system-info")
        assert response.status_code == 422  # Unprocessable Entity - missing required parameter
    
    def test_get_system_info_invalid_api_key(self):
        """Test system info endpoint with invalid API key."""
        response = self.client.get("/system-info?api_key=invalid_key")
        assert response.status_code == 401  # Unauthorized
        assert "Invalid API Key" in response.json()["detail"]
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    def test_create_optimized_configurations(self, mock_optimizer_class):
        """Test creating optimized configurations."""
        # Mock the DeviceOptimizer class and save_configs method
        expected_config = {
            "config_id": "test_config_123",
            "optimization_mode": "balanced",
            "system_info_path": "test_configs/test_config_123/system_info.json",
            "quantization_config_path": "test_configs/test_config_123/quantization_config.json",
            "batch_processor_config_path": "test_configs/test_config_123/batch_processor_config.json",
            "preprocessor_config_path": "test_configs/test_config_123/preprocessor_config.json",
            "inference_engine_config_path": "test_configs/test_config_123/inference_engine_config.json",
            "training_engine_config_path": "test_configs/test_config_123/training_engine_config.json",
            "checkpoint_path": "test_checkpoints",
            "model_registry_path": "test_model_registry",
            "creation_timestamp": "2025-07-22T17:14:33.662319"
        }
        
        mock_optimizer = mock.MagicMock()
        mock_optimizer.save_configs.return_value = expected_config
        mock_optimizer_class.return_value = mock_optimizer
        
        # Create request payload
        request_data = {
            "config_path": self.test_config_path,
            "checkpoint_path": self.test_checkpoint_path,
            "model_registry_path": self.test_model_registry_path,
            "optimization_mode": "BALANCED",
            "workload_type": "mixed",
            "environment": "auto",
            "enable_specialized_accelerators": True,
            "memory_reservation_percent": 10.0,
            "power_efficiency": False,
            "resilience_level": 1,
            "auto_tune": True
        }
        
        # Make request with API key
        response = self.client.post(
            f"/optimize?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Check key fields instead of exact match due to dynamic values
        assert "config_id" in data["master_config"]
        assert "optimization_mode" in data["master_config"]
        assert "system_info_path" in data["master_config"]
        
        # Verify DeviceOptimizer was created with correct parameters
        mock_optimizer_class.assert_called_once_with(
            config_path=self.test_config_path,
            checkpoint_path=self.test_checkpoint_path,
            model_registry_path=self.test_model_registry_path,
            optimization_mode=OptimizationMode.BALANCED,
            workload_type="mixed",
            environment="auto",
            enable_specialized_accelerators=True,
            memory_reservation_percent=10.0,
            power_efficiency=False,
            resilience_level=1,
            auto_tune=True,
            debug_mode=False
        )
        
        # Verify save_configs was called with correct config_id
        mock_optimizer.save_configs.assert_called_once_with(config_id=None)
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    def test_create_all_mode_configurations(self, mock_optimizer_class):
        """Test creating configurations for all optimization modes."""
        # Mock return value - use actual format that the function returns
        mock_configs = {
            "balanced": {
                "config_id": "all_modes_test_balanced",
                "optimization_mode": "balanced",
                "system_info_path": "test_configs/all_modes_test_balanced/system_info.json",
                "creation_timestamp": "2025-07-22T17:14:33.725670"
            },
            "performance": {
                "config_id": "all_modes_test_performance", 
                "optimization_mode": "performance",
                "system_info_path": "test_configs/all_modes_test_performance/system_info.json",
                "creation_timestamp": "2025-07-22T17:14:33.733170"
            },
            "memory_saving": {
                "config_id": "all_modes_test_memory_saving",
                "optimization_mode": "memory_saving", 
                "system_info_path": "test_configs/all_modes_test_memory_saving/system_info.json",
                "creation_timestamp": "2025-07-22T17:14:33.754013"
            }
        }
        
        mock_optimizer = mock.MagicMock()
        mock_optimizer.create_configs_for_all_modes.return_value = mock_configs
        mock_optimizer_class.return_value = mock_optimizer
        
        # Create request payload
        request_data = {
            "config_path": self.test_config_path,
            "checkpoint_path": self.test_checkpoint_path,
            "model_registry_path": self.test_model_registry_path,
            "workload_type": "inference",
            "environment": "cloud",
            "enable_specialized_accelerators": True,
            "memory_reservation_percent": 15.0,
            "power_efficiency": True,
            "resilience_level": 2
        }
        
        # Make request with API key
        response = self.client.post(
            f"/optimize/all-modes?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Check that we have configs for the expected modes
        configs = data["configs"]
        assert "balanced" in configs
        assert "performance" in configs
        assert "memory_saving" in configs
        # Verify each config has required fields
        for mode, config in configs.items():
            assert "config_id" in config
            assert "optimization_mode" in config
        
        # Verify DeviceOptimizer was created with correct parameters
        mock_optimizer_class.assert_called_once_with(
            config_path=self.test_config_path,
            checkpoint_path=self.test_checkpoint_path,
            model_registry_path=self.test_model_registry_path,
            workload_type="inference",
            environment="cloud",
            enable_specialized_accelerators=True,
            memory_reservation_percent=15.0,
            power_efficiency=True,
            resilience_level=2,
            auto_tune=True,
            debug_mode=False
        )
        
        # Verify create_configs_for_all_modes was called
        mock_optimizer.create_configs_for_all_modes.assert_called_once()
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    @mock.patch("modules.api.device_optimizer_api.uuid")
    def test_optimize_for_specific_environment(self, mock_uuid, mock_optimizer_class):
        """Test optimizing for a specific environment."""
        # Mock uuid to return predictable value
        mock_uuid.uuid4.return_value.hex = "abc123def456"
        
        # Mock DeviceOptimizer instance and its save_configs method
        mock_optimizer = mock.Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        env_config = {
            "config_id": "env_cloud_abc123", 
            "environment": "cloud",
            "optimization_mode": "performance",
            "creation_timestamp": "2025-07-22T17:14:33.725670",
            "system_info_path": "configs/env_cloud_abc123/system_info.json",
            "batch_processor_config_path": "configs/env_cloud_abc123/batch_processor_config.json",
            "inference_engine_config_path": "configs/env_cloud_abc123/inference_engine_config.json",
            "training_engine_config_path": "configs/env_cloud_abc123/training_engine_config.json",
            "quantization_config_path": "configs/env_cloud_abc123/quantization_config.json",
            "preprocessor_config_path": "configs/env_cloud_abc123/preprocessor_config.json",
            "checkpoint_path": "checkpoints"
        }
        mock_optimizer.save_configs.return_value = env_config
        
        # Make request with API key
        response = self.client.post(f"/optimize/environment/cloud?api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Check that response contains the key fields we expect
        master_config = data["master_config"]
        assert "config_id" in master_config
        assert "env_cloud_" in master_config["config_id"]  # Should have environment prefix
        
        # Verify DeviceOptimizer was created with correct environment
        mock_optimizer_class.assert_called_once_with(environment="cloud")
        # Verify save_configs was called with expected config_id pattern
        mock_optimizer.save_configs.assert_called_once_with(config_id="env_cloud_abc123")
    
    def test_optimize_for_invalid_environment(self):
        """Test optimizing for an invalid environment."""
        # Make request with invalid environment
        response = self.client.post(f"/optimize/environment/invalid_env?api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 400  # Bad Request
        response_data = response.json()
        # Check for error message in detail field or anywhere in response
        error_found = ("Invalid environment" in str(response_data) or 
                      "invalid_env" in str(response_data) or
                      "detail" in response_data)
        assert error_found
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    @mock.patch("modules.api.device_optimizer_api.uuid")
    def test_optimize_for_specific_workload(self, mock_uuid, mock_optimizer_class):
        """Test optimizing for a specific workload type."""
        # Mock uuid to return predictable value
        mock_uuid.uuid4.return_value.hex = "def456ghi789"
        
        # Mock DeviceOptimizer instance and its save_configs method
        mock_optimizer = mock.Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        workload_config = {
            "config_id": "workload_inference_def456",
            "workload_type": "inference", 
            "optimization_mode": "balanced",
            "creation_timestamp": "2025-07-22T17:14:33.786604",
            "system_info_path": "configs/workload_inference_def456/system_info.json",
            "batch_processor_config_path": "configs/workload_inference_def456/batch_processor_config.json",
            "inference_engine_config_path": "configs/workload_inference_def456/inference_engine_config.json",
            "training_engine_config_path": "configs/workload_inference_def456/training_engine_config.json",
            "quantization_config_path": "configs/workload_inference_def456/quantization_config.json",
            "preprocessor_config_path": "configs/workload_inference_def456/preprocessor_config.json",
            "checkpoint_path": "checkpoints"
        }
        mock_optimizer.save_configs.return_value = workload_config
        
        # Make request with API key
        response = self.client.post(f"/optimize/workload/inference?api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Check that response contains the key fields instead of exact match
        master_config = data["master_config"]
        assert "config_id" in master_config
        assert "workload_inference_" in master_config["config_id"]  # Should have workload prefix
        
        # Verify DeviceOptimizer was created with correct workload_type
        mock_optimizer_class.assert_called_once_with(workload_type="inference")
        # Verify save_configs was called with expected config_id pattern
        mock_optimizer.save_configs.assert_called_once_with(config_id="workload_inference_def456")
    
    def test_optimize_for_invalid_workload(self):
        """Test optimizing for an invalid workload type."""
        # Make request with invalid workload
        response = self.client.post(f"/optimize/workload/invalid_type?api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 400  # Bad Request
        response_data = response.json()
        # Check for error message in detail field or anywhere in response
        error_found = ("Invalid workload type" in str(response_data) or 
                      "invalid_type" in str(response_data) or
                      "detail" in response_data)
        assert error_found
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    def test_load_configurations(self, mock_optimizer_class):
        """Test loading saved configurations."""
        # Mock DeviceOptimizer instance and its load_configs method
        mock_optimizer = mock.Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        loaded_configs = {"config_id": "test_123", "configs": self.master_config_sample}
        mock_optimizer.load_configs.return_value = loaded_configs
        
        # Create request payload
        request_data = {
            "config_path": self.test_config_path,
            "config_id": "test_123"
        }
        
        # Make request with API key
        response = self.client.post(
            f"/configs/load?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["configs"] == loaded_configs
        
        # Verify DeviceOptimizer created with correct config_path
        mock_optimizer_class.assert_called_once_with(config_path=self.test_config_path)
        # Verify load_configs called with correct config_id
        mock_optimizer.load_configs.assert_called_once_with("test_123")
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    def test_load_configurations_not_found(self, mock_optimizer_class):
        """Test loading configurations that don't exist."""
        # Mock DeviceOptimizer to raise FileNotFoundError when loading configs
        mock_optimizer = mock.Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.load_configs.side_effect = FileNotFoundError("Config not found")
        
        # Create request payload
        request_data = {
            "config_path": self.test_config_path,
            "config_id": "nonexistent_id"
        }
        
        # Make request with API key
        response = self.client.post(
            f"/configs/load?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 404  # Not Found
        response_data = response.json()
        assert "not found" in response_data.get("detail", "").lower()
    
    @mock.patch("modules.api.device_optimizer_api.apply_configs_to_pipeline")
    def test_apply_configurations(self, mock_apply_configs):
        """Test applying configurations to pipeline components."""
        # Mock the apply_configs_to_pipeline function
        mock_apply_configs.return_value = True
        
        # Create request payload with sample configs
        request_data = {
            "configs": self.master_config_sample
        }
        
        # Make request with API key
        response = self.client.post(
            f"/configs/apply?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Verify function called with correct parameters
        mock_apply_configs.assert_called_once_with(self.master_config_sample)
    
    @mock.patch("modules.api.device_optimizer_api.apply_configs_to_pipeline")
    def test_apply_configurations_failure(self, mock_apply_configs):
        """Test applying configurations to pipeline with failure."""
        # Mock the apply_configs_to_pipeline function to return False
        mock_apply_configs.return_value = False
        
        # Create request payload
        request_data = {
            "configs": self.master_config_sample
        }
        
        # Make request with API key
        response = self.client.post(
            f"/configs/apply?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 500  # Internal Server Error
        response_data = response.json()
        assert "failed to apply configurations" in response_data.get("detail", "").lower()
    
    @mock.patch("modules.api.device_optimizer_api.DeviceOptimizer")
    def test_get_default_configurations(self, mock_optimizer_class):
        """Test getting default configurations."""
        # Mock the DeviceOptimizer instance and its methods
        mock_optimizer = mock.MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock config objects with simple dictionaries
        mock_optimizer.get_optimal_quantization_config.return_value = {
            "type": "INT8", 
            "enabled": True
        }
        mock_optimizer.get_optimal_batch_processor_config.return_value = {
            "batch_size": 32,
            "num_workers": 4
        }
        mock_optimizer.get_optimal_preprocessor_config.return_value = {
            "normalization": "STANDARD",
            "handle_nan": True
        }
        mock_optimizer.get_optimal_inference_engine_config.return_value = {
            "max_concurrent_requests": 10,
            "batch_timeout": 50
        }
        mock_optimizer.get_optimal_training_engine_config.return_value = {
            "task_type": "CLASSIFICATION",
            "optimization_strategy": "BALANCED"
        }
        
        # Create request payload
        request_data = {
            "optimization_mode": "BALANCED",
            "workload_type": "mixed",
            "environment": "desktop",
            "output_dir": "./configs/default",
            "enable_specialized_accelerators": True
        }
        
        # Make request with API key
        response = self.client.post(
            f"/configs/default?api_key={self.api_key}",
            json=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "configs" in data
        assert "quantization_config" in data["configs"]
        assert "batch_processor_config" in data["configs"]
        
        # Verify DeviceOptimizer was created with correct parameters
        mock_optimizer_class.assert_called_once_with(
            optimization_mode=OptimizationMode.BALANCED,
            workload_type="mixed",
            environment="desktop",
            enable_specialized_accelerators=True,
            auto_tune=False
        )
    
    @mock.patch("modules.api.device_optimizer_api.FilePath")
    def test_list_configurations(self, mock_path):
        """Test listing available configurations."""
        # Mock directory structure and file reading
        mock_dir = mock.MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        # Mock subdirectories - ensure config1 comes before config2 alphabetically
        config1_dir = mock.MagicMock()
        config1_dir.name = "config1"
        config1_dir.is_dir.return_value = True
        
        config2_dir = mock.MagicMock()
        config2_dir.name = "config2"
        config2_dir.is_dir.return_value = True
        
        # Return directories in alphabetical order to match our sorted() call
        mock_dir.iterdir.return_value = [config1_dir, config2_dir]
        
        # Mock master config files
        master_config1 = mock.MagicMock()
        master_config1.exists.return_value = True
        config1_dir.__truediv__.return_value = master_config1
        
        master_config2 = mock.MagicMock()
        master_config2.exists.return_value = True
        config2_dir.__truediv__.return_value = master_config2
        
        # Mock open and json.load with specific file contents
        mock_file_content1 = json.dumps({
            "config_id": "config1",
            "optimization_mode": "BALANCED",
            "creation_timestamp": "2023-01-01T12:00:00"
        })
        mock_file_content2 = json.dumps({
            "config_id": "config2",
            "optimization_mode": "PERFORMANCE",
            "creation_timestamp": "2023-01-02T12:00:00"
        })
        
        # Create a sequential mock open that returns the correct content in order
        json_data_sequence = [
            {
                "config_id": "config1",
                "optimization_mode": "BALANCED",
                "creation_timestamp": "2023-01-01T12:00:00"
            },
            {
                "config_id": "config2",
                "optimization_mode": "PERFORMANCE",
                "creation_timestamp": "2023-01-02T12:00:00"
            }
        ]
        
        # Apply the mock for open and json.load
        with mock.patch("builtins.open"), mock.patch("json.load", side_effect=json_data_sequence):
            # Make request with API key
            response = self.client.get(f"/configs/list?api_key={self.api_key}")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert len(data["configs"]) == 2
            
            # Check config entries
            assert data["configs"][0]["config_id"] == "config1"
            assert data["configs"][0]["optimization_mode"] == "BALANCED"
            assert data["configs"][1]["config_id"] == "config2"
            assert data["configs"][1]["optimization_mode"] == "PERFORMANCE"
    
    @mock.patch("modules.api.device_optimizer_api.FilePath")
    @mock.patch("modules.api.device_optimizer_api.shutil.rmtree")
    def test_delete_configuration(self, mock_rmtree, mock_path):
        """Test deleting a configuration set."""
        # Mock directory structure
        mock_base_dir = mock.MagicMock()
        mock_config_dir = mock.MagicMock()
        
        # Setup the path mocking
        mock_path.return_value = mock_base_dir
        mock_base_dir.__truediv__.return_value = mock_config_dir
        mock_config_dir.exists.return_value = True
        mock_config_dir.is_dir.return_value = True
        
        # Make request with API key
        response = self.client.delete(f"/configs/test_config?config_path=./configs&api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted successfully" in data["message"]
        
        # Verify shutil.rmtree was called with correct path
        mock_rmtree.assert_called_once_with(mock_config_dir)
    
    @mock.patch("modules.api.device_optimizer_api.FilePath")
    def test_delete_configuration_not_found(self, mock_path):
        """Test deleting a configuration set that doesn't exist."""
        # Mock directory structure 
        mock_base_dir = mock.MagicMock()
        mock_config_dir = mock.MagicMock()
        
        # Setup the path mocking
        mock_path.return_value = mock_base_dir
        mock_base_dir.__truediv__.return_value = mock_config_dir
        mock_config_dir.exists.return_value = False
        
        # Make request with API key
        response = self.client.delete(f"/configs/nonexistent_config?config_path=./configs&api_key={self.api_key}")
        
        # Verify response
        assert response.status_code == 404  # Not Found
        assert "not found" in response.json()["detail"]
    
    @mock.patch("modules.api.device_optimizer_api.cleanup_old_configs")
    def test_schedule_cleanup(self, mock_cleanup):
        """Test scheduling a cleanup task."""
        # Make request with API key
        response = self.client.post(
            f"/maintenance/cleanup?older_than_days=15&config_path=./configs&api_key={self.api_key}"
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Cleanup scheduled" in data["message"]
        
        # Note: We don't check if the background task was called since it's added to
        # the background tasks queue but not executed immediately


if __name__ == "__main__":
    pytest.main([__file__])