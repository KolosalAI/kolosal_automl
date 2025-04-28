import unittest
from unittest import mock
import json
import os
from fastapi.testclient import TestClient
from pathlib import Path

# Import the FastAPI app
from modules.api.device_optimizer_api import app, API_KEY, verify_api_key

# Import the modules being used by the app
from modules.device_optimizer import (
    DeviceOptimizer, OptimizationMode,
    create_optimized_configs, create_configs_for_all_modes,
    load_saved_configs, get_system_information, optimize_for_environment,
    optimize_for_workload, apply_configs_to_pipeline, get_default_config
)
from modules.configs import QuantizationType, QuantizationMode

class TestDeviceOptimizerAPI(unittest.TestCase):
    """Test cases for the Device Optimizer API."""
    
    def setUp(self):
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["api"], "CPU Device Optimizer API")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("description", data)
    
    @mock.patch("modules.api.device_optimizer.get_system_information")
    def test_get_system_info(self, mock_get_system_info):
        """Test retrieving system information."""
        # Mock the get_system_information function
        mock_get_system_info.return_value = self.system_info_sample
        
        # Make request with API key
        response = self.client.get(f"/system-info?api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, self.system_info_sample)
        
        # Verify function called with correct parameters
        mock_get_system_info.assert_called_once_with(enable_specialized_accelerators=True)
    
    def test_get_system_info_without_api_key(self):
        """Test system info endpoint requires API key."""
        response = self.client.get("/system-info")
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity - missing required parameter
    
    def test_get_system_info_invalid_api_key(self):
        """Test system info endpoint with invalid API key."""
        response = self.client.get("/system-info?api_key=invalid_key")
        self.assertEqual(response.status_code, 401)  # Unauthorized
        self.assertIn("Invalid API Key", response.json()["detail"])
    
    @mock.patch("modules.api.device_optimizer.create_optimized_configs")
    def test_create_optimized_configurations(self, mock_create_configs):
        """Test creating optimized configurations."""
        # Mock the create_optimized_configs function
        mock_create_configs.return_value = self.master_config_sample
        
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["master_config"], self.master_config_sample)
        
        # Verify function called with correct parameters
        mock_create_configs.assert_called_once_with(
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
            config_id=None
        )
    
    @mock.patch("modules.api.device_optimizer.create_configs_for_all_modes")
    def test_create_all_mode_configurations(self, mock_create_all_modes):
        """Test creating configurations for all optimization modes."""
        # Mock return value - sample configs for different modes
        mock_configs = {
            "BALANCED": {"mode": "BALANCED", "config_id": "balanced_123"},
            "PERFORMANCE": {"mode": "PERFORMANCE", "config_id": "perf_123"},
            "MEMORY_SAVING": {"mode": "MEMORY_SAVING", "config_id": "mem_123"}
        }
        mock_create_all_modes.return_value = mock_configs
        
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["configs"], mock_configs)
        
        # Verify function called with correct parameters
        mock_create_all_modes.assert_called_once_with(
            config_path=self.test_config_path,
            checkpoint_path=self.test_checkpoint_path,
            model_registry_path=self.test_model_registry_path,
            workload_type="inference",
            environment="cloud",
            enable_specialized_accelerators=True,
            memory_reservation_percent=15.0,
            power_efficiency=True,
            resilience_level=2
        )
    
    @mock.patch("modules.api.device_optimizer.optimize_for_environment")
    def test_optimize_for_specific_environment(self, mock_optimize_env):
        """Test optimizing for a specific environment."""
        # Mock the optimize_for_environment function
        env_config = {"environment": "cloud", "config_id": "cloud_123"}
        mock_optimize_env.return_value = env_config
        
        # Make request with API key
        response = self.client.post(f"/optimize/environment/cloud?api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["master_config"], env_config)
        
        # Verify function called with correct parameter
        mock_optimize_env.assert_called_once_with("cloud")
    
    def test_optimize_for_invalid_environment(self):
        """Test optimizing for an invalid environment."""
        # Make request with invalid environment
        response = self.client.post(f"/optimize/environment/invalid_env?api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 400)  # Bad Request
        self.assertIn("Invalid environment", response.json()["detail"])
    
    @mock.patch("modules.api.device_optimizer.optimize_for_workload")
    def test_optimize_for_specific_workload(self, mock_optimize_workload):
        """Test optimizing for a specific workload type."""
        # Mock the optimize_for_workload function
        workload_config = {"workload_type": "inference", "config_id": "inference_123"}
        mock_optimize_workload.return_value = workload_config
        
        # Make request with API key
        response = self.client.post(f"/optimize/workload/inference?api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["master_config"], workload_config)
        
        # Verify function called with correct parameter
        mock_optimize_workload.assert_called_once_with("inference")
    
    def test_optimize_for_invalid_workload(self):
        """Test optimizing for an invalid workload type."""
        # Make request with invalid workload
        response = self.client.post(f"/optimize/workload/invalid_type?api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 400)  # Bad Request
        self.assertIn("Invalid workload type", response.json()["detail"])
    
    @mock.patch("modules.api.device_optimizer.load_saved_configs")
    def test_load_configurations(self, mock_load_configs):
        """Test loading saved configurations."""
        # Mock the load_saved_configs function
        loaded_configs = {"config_id": "test_123", "configs": self.master_config_sample}
        mock_load_configs.return_value = loaded_configs
        
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["configs"], loaded_configs)
        
        # Verify function called with correct parameters
        mock_load_configs.assert_called_once_with(
            config_path=self.test_config_path,
            config_id="test_123"
        )
    
    @mock.patch("modules.api.device_optimizer.load_saved_configs")
    def test_load_configurations_not_found(self, mock_load_configs):
        """Test loading configurations that don't exist."""
        # Mock the load_saved_configs to raise FileNotFoundError
        mock_load_configs.side_effect = FileNotFoundError("Config not found")
        
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
        self.assertEqual(response.status_code, 404)  # Not Found
        self.assertIn("Config not found", response.json()["detail"])
    
    @mock.patch("modules.api.device_optimizer.apply_configs_to_pipeline")
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        
        # Verify function called with correct parameters
        mock_apply_configs.assert_called_once_with(self.master_config_sample)
    
    @mock.patch("modules.api.device_optimizer.apply_configs_to_pipeline")
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
        self.assertEqual(response.status_code, 500)  # Internal Server Error
        self.assertIn("Failed to apply configurations", response.json()["detail"])
    
    @mock.patch("modules.api.device_optimizer.get_default_config")
    def test_get_default_configurations(self, mock_default_config):
        """Test getting default configurations."""
        # Mock the get_default_config function
        default_configs = {
            "optimization_mode": "BALANCED",
            "quantization": {"type": "INT8", "enabled": True},
            "batch_processing": {"batch_size": 32}
        }
        mock_default_config.return_value = default_configs
        
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["configs"], default_configs)
        
        # Verify function called with correct parameters
        mock_default_config.assert_called_once_with(
            optimization_mode=OptimizationMode.BALANCED,
            workload_type="mixed",
            environment="desktop",
            output_dir="./configs/default",
            enable_specialized_accelerators=True
        )
    
    @mock.patch("modules.api.device_optimizer.FilePath")
    def test_list_configurations(self, mock_path):
        """Test listing available configurations."""
        # Mock directory structure and file reading
        mock_dir = mock.MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        # Mock subdirectories
        config1_dir = mock.MagicMock()
        config1_dir.name = "config1"
        config1_dir.is_dir.return_value = True
        
        config2_dir = mock.MagicMock()
        config2_dir.name = "config2"
        config2_dir.is_dir.return_value = True
        
        mock_dir.iterdir.return_value = [config1_dir, config2_dir]
        
        # Mock master config files
        master_config1 = mock.MagicMock()
        master_config1.exists.return_value = True
        config1_dir.__truediv__.return_value = master_config1
        
        master_config2 = mock.MagicMock()
        master_config2.exists.return_value = True
        config2_dir.__truediv__.return_value = master_config2
        
        # Mock open and json.load
        mock_open = mock.mock_open()
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
        
        # Set up mock open to return different content based on path
        def mock_open_func(file, mode):
            if "config1" in str(file):
                return mock.mock_open(read_data=mock_file_content1)()
            else:
                return mock.mock_open(read_data=mock_file_content2)()
        
        # Apply the mock for open
        with mock.patch("builtins.open", mock_open_func):
            # Make request with API key
            response = self.client.get(f"/configs/list?api_key={self.api_key}")
            
            # Verify response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(len(data["configs"]), 2)
            
            # Check config entries
            self.assertEqual(data["configs"][0]["config_id"], "config1")
            self.assertEqual(data["configs"][0]["optimization_mode"], "BALANCED")
            self.assertEqual(data["configs"][1]["config_id"], "config2")
            self.assertEqual(data["configs"][1]["optimization_mode"], "PERFORMANCE")
    
    @mock.patch("modules.api.device_optimizer.FilePath")
    @mock.patch("modules.api.device_optimizer.shutil.rmtree")
    def test_delete_configuration(self, mock_rmtree, mock_path):
        """Test deleting a configuration set."""
        # Mock directory checks
        mock_dir = mock.MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        
        # Make request with API key
        response = self.client.delete(f"/configs/test_config?config_path=./configs&api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("deleted successfully", data["message"])
        
        # Verify shutil.rmtree was called with correct path
        mock_rmtree.assert_called_once_with(mock_dir)
    
    @mock.patch("modules.api.device_optimizer.FilePath")
    def test_delete_configuration_not_found(self, mock_path):
        """Test deleting a configuration set that doesn't exist."""
        # Mock directory doesn't exist
        mock_dir = mock.MagicMock()
        mock_path.return_value = mock_dir
        mock_dir.exists.return_value = False
        
        # Make request with API key
        response = self.client.delete(f"/configs/nonexistent_config?config_path=./configs&api_key={self.api_key}")
        
        # Verify response
        self.assertEqual(response.status_code, 404)  # Not Found
        self.assertIn("not found", response.json()["detail"])
    
    @mock.patch("modules.api.device_optimizer.cleanup_old_configs")
    def test_schedule_cleanup(self, mock_cleanup):
        """Test scheduling a cleanup task."""
        # Make request with API key
        response = self.client.post(
            f"/maintenance/cleanup?older_than_days=15&config_path=./configs&api_key={self.api_key}"
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Cleanup scheduled", data["message"])
        
        # Note: We don't check if the background task was called since it's added to
        # the background tasks queue but not executed immediately


if __name__ == "__main__":
    unittest.main()