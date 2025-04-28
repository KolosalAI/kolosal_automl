import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import numpy as np
from fastapi.testclient import TestClient
from contextlib import contextmanager

# Import the FastAPI app and related modules
from modules.api.quantizer_api import app, get_quantizer, numpy_to_list, list_to_numpy
from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
from modules.engine.quantizer import Quantizer


class TestQuantizerAPI(unittest.TestCase):
    """Test suite for the Quantizer API endpoints."""

    def setUp(self):
        """Set up the test client and mock dependencies before each test."""
        self.client = TestClient(app)
        
        # Create a mock quantizer instance for testing
        self.mock_quantizer = MagicMock(spec=Quantizer)
        
        # Mock configuration for the quantizer
        self.mock_config = MagicMock(spec=QuantizationConfig)
        self.mock_config.quantization_type = QuantizationType.INT8
        self.mock_config.quantization_mode = QuantizationMode.DYNAMIC
        
        # Set up the mock quantizer to return appropriate values
        self.mock_quantizer.get_config.return_value = {
            "quantization_type": QuantizationType.INT8,
            "quantization_mode": QuantizationMode.DYNAMIC,
            "num_bits": 8
        }
        self.mock_quantizer.config = self.mock_config
        self.mock_quantizer.scale = 0.1
        self.mock_quantizer.zero_point = 128
        
        # Set up mock stats
        self.mock_quantizer.get_stats.return_value = {
            "clipped_values": 0,
            "total_values": 100,
            "quantize_calls": 5,
            "dequantize_calls": 3,
            "cache_hits": 2,
            "cache_misses": 3,
            "last_scale": 0.1,
            "last_zero_point": 128,
            "processing_time_ms": 1.5
        }
        
        # Mock the _lock attribute required by dequantize endpoint
        self.mock_quantizer._lock = MagicMock()
        self.mock_quantizer._qtype = np.int8

    @patch("modules.api.quantizer.quantizer_instances")
    def test_root_endpoint(self, mock_instances):
        """Test the root endpoint returns the expected health check response."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "service": "Quantizer API"})

    @patch("modules.api.quantizer.quantizer_instances")
    def test_get_instances(self, mock_instances):
        """Test retrieving all quantizer instances."""
        mock_instances.keys.return_value = ["default", "test_instance"]
        
        response = self.client.get("/instances")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["default", "test_instance"])

    @patch("modules.api.quantizer.quantizer_instances")
    def test_create_instance(self, mock_instances):
        """Test creating a new quantizer instance."""
        # Set up the mock to check if the instance exists
        mock_instances.__contains__.return_value = False
        
        config_data = {
            "quantization_type": "INT8",
            "quantization_mode": "DYNAMIC",
            "num_bits": 8,
            "symmetric": False,
            "enable_cache": True
        }
        
        response = self.client.post("/instances/test_instance", json=config_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Quantizer instance test_instance created successfully"})
        
        # Test duplicate instance creation
        mock_instances.__contains__.return_value = True
        response = self.client.post("/instances/test_instance", json=config_data)
        self.assertEqual(response.status_code, 409)

    @patch("modules.api.quantizer.quantizer_instances")
    def test_delete_instance(self, mock_instances):
        """Test deleting a quantizer instance."""
        # Mock the get_quantizer function to return our mock
        with patch("modules.api.quantizer.get_quantizer", return_value=self.mock_quantizer):
            # Set up the mock for checking existence
            def mock_contains(key):
                return key in ["default", "test_instance"]
            
            mock_instances.__contains__.side_effect = mock_contains
            
            # Test deleting default instance (not allowed)
            response = self.client.delete("/instances/default")
            self.assertEqual(response.status_code, 403)
            
            # Test deleting non-default instance
            response = self.client.delete("/instances/test_instance")
            self.assertEqual(response.status_code, 200)
            mock_instances.__delitem__.assert_called_once_with("test_instance")
            
            # Test deleting non-existent instance
            response = self.client.delete("/instances/nonexistent")
            self.assertEqual(response.status_code, 404)

    @patch("modules.api.quantizer.get_quantizer")
    def test_get_config(self, mock_get_quantizer):
        """Test retrieving the configuration of a quantizer instance."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        response = self.client.get("/instances/default/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["quantization_type"], "INT8")
        self.assertEqual(response.json()["quantization_mode"], "DYNAMIC")
        self.assertEqual(response.json()["num_bits"], 8)

    @patch("modules.api.quantizer.get_quantizer")
    def test_update_config(self, mock_get_quantizer):
        """Test updating the configuration of a quantizer instance."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        config_data = {
            "quantization_type": "INT16",
            "quantization_mode": "CALIBRATED",
            "num_bits": 16,
            "symmetric": True
        }
        
        response = self.client.put("/instances/default/config", json=config_data)
        self.assertEqual(response.status_code, 200)
        self.mock_quantizer.update_config.assert_called_once()

    @patch("modules.api.quantizer.get_quantizer")
    def test_quantize_data(self, mock_get_quantizer):
        """Test quantizing input data."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Set up the quantize method to return a numpy array
        quantized_data = np.array([[1, 2], [3, 4]], dtype=np.int8)
        self.mock_quantizer.quantize.return_value = quantized_data
        
        request_data = {
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "validate": True
        }
        
        response = self.client.post("/instances/default/quantize", json=request_data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["data"], [[1, 2], [3, 4]])
        self.assertEqual(result["quantization_type"], "INT8")
        self.assertEqual(result["original_shape"], [2, 2])

    @patch("modules.api.quantizer.get_quantizer")
    def test_dequantize_data(self, mock_get_quantizer):
        """Test dequantizing input data."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Set up the dequantize method to return a numpy array
        dequantized_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.mock_quantizer.dequantize.return_value = dequantized_data
        
        request_data = {
            "data": [[1, 2], [3, 4]],
            "channel_dim": None
        }
        
        response = self.client.post("/instances/default/dequantize", json=request_data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["data"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(result["original_shape"], [2, 2])

    @patch("modules.api.quantizer.get_quantizer")
    def test_quantize_dequantize_data(self, mock_get_quantizer):
        """Test quantize and dequantize in one operation."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Set up the quantize_dequantize method to return a numpy array
        result_data = np.array([[0.9, 1.9], [2.9, 3.9]], dtype=np.float32)
        self.mock_quantizer.quantize_dequantize.return_value = result_data
        
        request_data = {
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "validate": True
        }
        
        response = self.client.post("/instances/default/quantize_dequantize", json=request_data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["data"], [[0.9, 1.9], [2.9, 3.9]])
        self.assertEqual(result["original_shape"], [2, 2])

    @patch("modules.api.quantizer.get_quantizer")
    def test_calibrate_quantizer(self, mock_get_quantizer):
        """Test calibrating the quantizer with example data."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        request_data = {
            "data": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        }
        
        response = self.client.post("/instances/default/calibrate", json=request_data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["message"], "Calibration successful")
        self.assertEqual(result["scale"], 0.1)
        self.assertEqual(result["zero_point"], 128)
        self.mock_quantizer.calibrate.assert_called_once()

    @patch("modules.api.quantizer.get_quantizer")
    def test_compute_scale_zero_point(self, mock_get_quantizer):
        """Test computing scale and zero point for input data."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Mock the compute_scale_and_zero_point method
        self.mock_quantizer.compute_scale_and_zero_point.return_value = (0.1, 128)
        
        request_data = {
            "data": [[1.0, 2.0], [3.0, 4.0]]
        }
        
        response = self.client.post("/instances/default/compute_params", json=request_data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["scale"], 0.1)
        self.assertEqual(result["zero_point"], 128)

    @patch("modules.api.quantizer.get_quantizer")
    def test_get_stats(self, mock_get_quantizer):
        """Test retrieving quantization statistics."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        response = self.client.get("/instances/default/stats")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["quantize_calls"], 5)
        self.assertEqual(result["dequantize_calls"], 3)
        self.assertEqual(result["cache_hits"], 2)

    @patch("modules.api.quantizer.get_quantizer")
    def test_clear_cache(self, mock_get_quantizer):
        """Test clearing the dequantization cache."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        response = self.client.post("/instances/default/clear_cache")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Cache cleared successfully"})
        self.mock_quantizer.clear_cache.assert_called_once()

    @patch("modules.api.quantizer.get_quantizer")
    def test_reset_stats(self, mock_get_quantizer):
        """Test resetting quantization statistics."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        response = self.client.post("/instances/default/reset_stats")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Statistics reset successfully"})
        self.mock_quantizer.reset_stats.assert_called_once()

    @patch("modules.api.quantizer.get_quantizer")
    def test_export_parameters(self, mock_get_quantizer):
        """Test exporting quantization parameters."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Set up the export_import_parameters method
        self.mock_quantizer.export_import_parameters.return_value = {
            "scale": np.float32(0.1),
            "zero_point": np.int8(128),
            "quantization_type": QuantizationType.INT8,
            "quantization_mode": QuantizationMode.DYNAMIC
        }
        
        response = self.client.get("/instances/default/parameters")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["scale"], 0.1)
        self.assertEqual(result["zero_point"], 128)
        self.assertEqual(result["quantization_type"], "INT8")
        self.assertEqual(result["quantization_mode"], "DYNAMIC")

    @patch("modules.api.quantizer.get_quantizer")
    def test_import_parameters(self, mock_get_quantizer):
        """Test importing quantization parameters."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        request_data = {
            "parameters": {
                "scale": 0.1,
                "zero_point": 128,
                "quantization_type": "INT8",
                "quantization_mode": "DYNAMIC"
            }
        }
        
        response = self.client.post("/instances/default/parameters", json=request_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Parameters imported successfully"})
        self.mock_quantizer.load_parameters.assert_called_once()

    @patch("modules.api.quantizer.get_quantizer")
    def test_get_layer_params(self, mock_get_quantizer):
        """Test getting quantization parameters for a specific layer."""
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Mock the get_layer_quantization_params method
        self.mock_quantizer.get_layer_quantization_params.return_value = {
            "scale": 0.1,
            "zero_point": 128,
            "type": QuantizationType.INT8
        }
        
        response = self.client.get("/instances/default/layer_params/conv1")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["scale"], 0.1)
        self.assertEqual(result["zero_point"], 128)
        self.assertEqual(result["type"], "INT8")

    @patch("modules.api.quantizer.get_quantizer")
    @patch("builtins.open")
    def test_upload_numpy(self, mock_open, mock_get_quantizer):
        """Test uploading and processing a NumPy array file."""
        # This test requires more complex mocking for file uploads and async operations
        # For simplicity, we're not actually testing the file upload functionality
        # but rather mocking the necessary components
        mock_get_quantizer.return_value = self.mock_quantizer
        
        # Since this endpoint uses UploadFile which is complex to mock in unit tests,
        # this test is intentionally simplified
        pass

    def test_documentation(self):
        """Test retrieving API documentation."""
        response = self.client.get("/documentation")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("title", result)
        self.assertIn("description", result)
        self.assertIn("version", result)
        self.assertIn("endpoints", result)
        self.assertIn("examples", result)

    def test_helper_functions(self):
        """Test the helper functions for converting between numpy arrays and Python lists."""
        # Test list_to_numpy
        test_list = [[1.0, 2.0], [3.0, 4.0]]
        arr = list_to_numpy(test_list)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float32)
        
        # Test numpy_to_list
        test_arr = np.array([[1, 2], [3, 4]], dtype=np.int8)
        result_list = numpy_to_list(test_arr)
        self.assertIsInstance(result_list, list)
        self.assertEqual(result_list, [[1, 2], [3, 4]])
        
        # Test handling non-numpy inputs
        regular_list = [[1, 2], [3, 4]]
        self.assertEqual(numpy_to_list(regular_list), regular_list)


if __name__ == "__main__":
    unittest.main()