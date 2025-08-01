import unittest
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import pytest
from io import BytesIO, StringIO
import uuid
import os
from datetime import datetime

# Try to import FastAPI dependencies with error handling
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError as e:
    FASTAPI_AVAILABLE = False
    FASTAPI_ERROR = str(e)
    # Create a mock TestClient for when FastAPI is not available
    class MockTestClient:
        def __init__(self, *args, **kwargs):
            pass
    TestClient = MockTestClient

# Try to import API modules with fallback
try:
    from modules.api.data_preprocessor_api import (
        app, 
        DataPreprocessor,
        PreprocessorConfig,
        NormalizationType,
        active_preprocessors,
        clean_config_for_creation,
        create_preprocessor_config,
        parse_csv_data,
        get_preprocessor
    )
    API_MODULES_AVAILABLE = True
except ImportError as e:
    API_MODULES_AVAILABLE = False
    API_ERROR = str(e)

# Skip all tests if dependencies are not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE or not API_MODULES_AVAILABLE,
    reason=f"FastAPI or API modules not available: FastAPI={FASTAPI_AVAILABLE}, API={API_MODULES_AVAILABLE}"
)

class TestDataPreprocessorAPI(unittest.TestCase):
    def setUp(self):
        """Set up the test client and clean up any active preprocessors."""
        if not FASTAPI_AVAILABLE or not API_MODULES_AVAILABLE:
            self.skipTest("FastAPI or API modules not available")
        
        self.client = TestClient(app)
        # Clear any active preprocessors
        active_preprocessors.clear()

    def tearDown(self):
        """Clean up after tests."""
        active_preprocessors.clear()

    def test_health_check_endpoint(self):
        """Test the health check endpoint returns successful response."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["message"], "Data Preprocessor API is operational")

    def test_create_preprocessor(self):
        """Test creating a new preprocessor instance."""
        # Prepare test data
        config_data = {
            "normalization": "STANDARD",
            "handle_nan": True,
            "handle_inf": True,
            "detect_outliers": False,
            "nan_strategy": "MEAN",
            "inf_strategy": "MAX_VALUE"
        }

        # Make request
        response = self.client.post("/preprocessors", json=config_data)
        
        # Check response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("preprocessor_id", data)
        self.assertIn("config", data)
        self.assertIn("created_at", data)
        
        # Verify preprocessor was created and stored
        preprocessor_id = data["preprocessor_id"]
        self.assertIn(preprocessor_id, active_preprocessors)
        self.assertIsInstance(active_preprocessors[preprocessor_id], DataPreprocessor)

    def test_list_preprocessors(self):
        """Test listing all active preprocessors."""
        # Create a couple of preprocessors
        config_data = {"normalization": "STANDARD"}
        response1 = self.client.post("/preprocessors", json=config_data)
        response2 = self.client.post("/preprocessors", json=config_data)
        
        # Get list of preprocessors
        response = self.client.get("/preprocessors")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("preprocessors", data)
        self.assertEqual(len(data["preprocessors"]), 2)

    def test_get_preprocessor_info(self):
        """Test getting detailed information about a specific preprocessor."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Get preprocessor info
        response = self.client.get(f"/preprocessors/{preprocessor_id}")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["preprocessor_id"], preprocessor_id)
        self.assertFalse(data["fitted"])
        self.assertIn("config", data)

    def test_get_nonexistent_preprocessor(self):
        """Test getting a preprocessor that doesn't exist returns 404."""
        response = self.client.get("/preprocessors/nonexistent-id")
        self.assertEqual(response.status_code, 404)

    def test_delete_preprocessor(self):
        """Test deleting a preprocessor."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Delete preprocessor
        response = self.client.delete(f"/preprocessors/{preprocessor_id}")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify preprocessor was removed
        self.assertNotIn(preprocessor_id, active_preprocessors)

    def test_delete_nonexistent_preprocessor(self):
        """Test deleting a preprocessor that doesn't exist returns 404."""
        response = self.client.delete("/preprocessors/nonexistent-id")
        self.assertEqual(response.status_code, 404)

    @patch.object(DataPreprocessor, 'fit')
    def test_fit_preprocessor_with_json_data(self, mock_fit):
        """Test fitting a preprocessor with JSON data."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Prepare test data
        data = {
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "feature_names": ["feature1", "feature2"]
        }
        
        # Mock fit method
        mock_fit.return_value = None
        
        # Fit preprocessor
        response = self.client.post(f"/preprocessors/{preprocessor_id}/fit", json=data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify fit was called
        mock_fit.assert_called_once()
        args, _ = mock_fit.call_args
        np.testing.assert_array_equal(args[0], np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        self.assertEqual(args[1], ["feature1", "feature2"])

    @patch.object(DataPreprocessor, 'fit')
    def test_fit_preprocessor_with_csv_file(self, mock_fit):
        """Test fitting a preprocessor with a CSV file."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Prepare CSV data
        csv_content = "feature1,feature2\n1.0,2.0\n3.0,4.0\n5.0,6.0"
        csv_file = BytesIO(csv_content.encode())
        
        # Mock fit method and parse_csv_data
        mock_fit.return_value = None
        
        # Patch parse_csv_data to return expected values
        with patch('modules.api.data_preprocessor_api.parse_csv_data', return_value=(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 
            ["feature1", "feature2"]
        )):
            # Fit preprocessor
            response = self.client.post(
                f"/preprocessors/{preprocessor_id}/fit",
                files={"csv_file": ("test.csv", csv_file, "text/csv")}
            )
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            
            # Verify fit was called
            mock_fit.assert_called_once()

    @patch.object(DataPreprocessor, 'transform')
    def test_transform_data_json_input_json_output(self, mock_transform):
        """Test transforming data with JSON input and JSON output."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Set the preprocessor as fitted
        preprocessor = active_preprocessors[preprocessor_id]
        preprocessor._fitted = True
        preprocessor._feature_names = ["feature1", "feature2"]
        
        # Prepare test data
        data = {
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        }
        
        # Mock transform method
        transformed_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_transform.return_value = transformed_data
        
        # Transform data
        response = self.client.post(
            f"/preprocessors/{preprocessor_id}/transform",
            json={"data": data["data"]},
            params={"output_format": "json"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("transformed_data", result)
        self.assertEqual(result["shape"], [3, 2])
        np.testing.assert_array_equal(
            np.array(result["transformed_data"]), 
            transformed_data
        )

    @patch.object(DataPreprocessor, 'transform')
    def test_transform_data_not_fitted(self, mock_transform):
        """Test transforming data with a preprocessor that is not fitted."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Prepare test data
        data = {
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        }
        
        # Transform data
        response = self.client.post(
            f"/preprocessors/{preprocessor_id}/transform",
            json={"data": data["data"]}
        )
        
        # Check response
        self.assertEqual(response.status_code, 400)
        self.assertIn("Preprocessor is not fitted", response.json()["detail"])

    @patch.object(DataPreprocessor, 'fit_transform')
    def test_fit_transform_data(self, mock_fit_transform):
        """Test fit and transform data in one operation."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Prepare test data
        data = {
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "feature_names": ["feature1", "feature2"]
        }
        
        # Mock fit_transform method
        transformed_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_fit_transform.return_value = transformed_data
        
        # Fit and transform data
        response = self.client.post(
            f"/preprocessors/{preprocessor_id}/fit-transform",
            json=data,
            params={"output_format": "json"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("transformed_data", result)
        np.testing.assert_array_equal(
            np.array(result["transformed_data"]), 
            transformed_data
        )

    @patch.object(DataPreprocessor, 'partial_fit')
    def test_partial_fit_preprocessor(self, mock_partial_fit):
        """Test partially fitting a preprocessor with new data."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Prepare test data
        data = {
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "feature_names": ["feature1", "feature2"]
        }
        
        # Mock partial_fit method
        mock_partial_fit.return_value = None
        
        # Partial fit preprocessor
        response = self.client.post(f"/preprocessors/{preprocessor_id}/partial-fit", json=data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify partial_fit was called
        mock_partial_fit.assert_called_once()

    @patch.object(DataPreprocessor, 'reverse_transform')
    def test_reverse_transform_data(self, mock_reverse_transform):
        """Test reverse transforming data."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Set the preprocessor as fitted
        preprocessor = active_preprocessors[preprocessor_id]
        preprocessor._fitted = True
        preprocessor._feature_names = ["feature1", "feature2"]
        
        # Prepare test data
        data = {
            "data": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        }
        
        # Mock reverse_transform method
        reverse_transformed_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mock_reverse_transform.return_value = reverse_transformed_data
        
        # Reverse transform data
        response = self.client.post(
            f"/preprocessors/{preprocessor_id}/reverse-transform",
            json={"data": data["data"]},
            params={"output_format": "json"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("reverse_transformed_data", result)
        np.testing.assert_array_equal(
            np.array(result["reverse_transformed_data"]), 
            reverse_transformed_data
        )

    @patch.object(DataPreprocessor, 'reset')
    def test_reset_preprocessor(self, mock_reset):
        """Test resetting a preprocessor to its initial state."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Mock reset method
        mock_reset.return_value = None
        
        # Reset preprocessor
        response = self.client.post(f"/preprocessors/{preprocessor_id}/reset")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify reset was called
        mock_reset.assert_called_once()

    @patch.object(DataPreprocessor, 'update_config')
    def test_update_preprocessor_config(self, mock_update_config):
        """Test updating the configuration of an existing preprocessor."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # New configuration
        new_config = {
            "normalization": "MINMAX",
            "handle_nan": False,
            "handle_inf": False
        }
        
        # Mock update_config method
        mock_update_config.return_value = None
        
        # Update preprocessor config
        response = self.client.post(f"/preprocessors/{preprocessor_id}/update-config", json=new_config)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify update_config was called
        mock_update_config.assert_called_once()

    @patch.object(DataPreprocessor, 'get_statistics')
    def test_get_preprocessor_statistics(self, mock_get_statistics):
        """Test getting statistics computed by a fitted preprocessor."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Set the preprocessor as fitted
        preprocessor = active_preprocessors[preprocessor_id]
        preprocessor._fitted = True
        
        # Mock get_statistics method
        mock_statistics = {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.5, 0.7])
        }
        mock_get_statistics.return_value = mock_statistics
        
        # Get statistics
        response = self.client.get(f"/preprocessors/{preprocessor_id}/statistics")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["preprocessor_id"], preprocessor_id)
        self.assertIn("statistics", data)
        self.assertIn("mean", data["statistics"])
        self.assertIn("std", data["statistics"])
        
        # Verify get_statistics was called
        mock_get_statistics.assert_called_once()

    @patch.object(DataPreprocessor, 'get_performance_metrics')
    def test_get_preprocessor_metrics(self, mock_get_metrics):
        """Test getting performance metrics collected during preprocessing operations."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Mock get_performance_metrics method
        mock_metrics = {
            "fit_time": [0.01, 0.02],
            "transform_time": [0.005, 0.007]
        }
        mock_get_metrics.return_value = mock_metrics
        
        # Get metrics
        response = self.client.get(f"/preprocessors/{preprocessor_id}/metrics")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["preprocessor_id"], preprocessor_id)
        self.assertIn("metrics", data)
        self.assertIn("fit_time", data["metrics"])
        self.assertIn("transform_time", data["metrics"])
        
        # Verify get_performance_metrics was called
        mock_get_metrics.assert_called_once()

    @patch.object(DataPreprocessor, 'serialize')
    def test_serialize_preprocessor(self, mock_serialize):
        """Test serializing (saving) a preprocessor to disk."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Mock serialize method
        mock_serialize.return_value = True
        
        # Serialize preprocessor
        response = self.client.post(f"/preprocessors/{preprocessor_id}/serialize")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        # Verify serialize was called
        mock_serialize.assert_called_once()

    @patch.object(DataPreprocessor, 'deserialize')
    def test_deserialize_preprocessor(self, mock_deserialize):
        """Test deserializing (loading) a preprocessor from a file."""
        # Mock deserialize method
        mock_preprocessor = MagicMock()
        mock_preprocessor.config = PreprocessorConfig()
        mock_deserialize.return_value = mock_preprocessor
        
        # Create a dummy file
        file_content = b"dummy serialized data"
        
        # Deserialize preprocessor
        with patch('builtins.open', mock_open(read_data=file_content)):
            response = self.client.post(
                "/preprocessors/deserialize",
                files={"file": ("preprocessor.pkl", BytesIO(file_content), "application/octet-stream")}
            )
        
        # Check response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("preprocessor_id", data)
        self.assertIn("config", data)
        
        # Verify deserialize was called
        mock_deserialize.assert_called_once()
        
    @patch.object(DataPreprocessor, 'serialize')
    def test_download_preprocessor(self, mock_serialize):
        """Test serializing and downloading a preprocessor."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Mock serialize method
        mock_serialize.return_value = True
        
        # Patch open to avoid actual file operations
        with patch('builtins.open', mock_open()):
            # Download preprocessor
            response = self.client.get(f"/preprocessors/{preprocessor_id}/download")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/octet-stream")
        self.assertEqual(
            response.headers["Content-Disposition"], 
            f"attachment; filename=preprocessor_{preprocessor_id}.pkl"
        )
        
        # Verify serialize was called
        mock_serialize.assert_called_once()

    def test_parse_csv_data(self):
        """Test parsing CSV data from uploaded file content."""
        # Prepare CSV data
        csv_content = "feature1,feature2\n1.0,2.0\n3.0,4.0\n5.0,6.0"
        
        # Parse CSV data
        data, feature_names = parse_csv_data(csv_content.encode(), has_header=True)
        
        # Check results
        np.testing.assert_array_equal(data, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        self.assertEqual(feature_names, ["feature1", "feature2"])

    def test_clean_config_for_creation(self):
        """Test cleaning the configuration dictionary for preprocessor creation."""
        from enum import Enum
        
        # Create a mock Enum for testing
        class TestEnum(Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"
        
        # Prepare config with Enum
        config = {
            "enum_key": TestEnum.VALUE1,
            "normal_key": "normal_value"
        }
        
        # Clean config
        cleaned_config = clean_config_for_creation(config)
        
        # Check results
        self.assertEqual(cleaned_config["enum_key"], "value1")
        self.assertEqual(cleaned_config["normal_key"], "normal_value")

    def test_create_preprocessor_config(self):
        """Test creating a PreprocessorConfig instance from a dictionary."""
        # Prepare config dictionary
        config_dict = {
            "normalization": "STANDARD",
            "handle_nan": True,
            "handle_inf": False
        }
        
        # Create config
        config = create_preprocessor_config(config_dict)
        
        # Check results
        self.assertIsInstance(config, PreprocessorConfig)
        self.assertEqual(config.normalization, NormalizationType.STANDARD)
        self.assertEqual(config.handle_nan, True)
        self.assertEqual(config.handle_inf, False)

    def test_get_preprocessor(self):
        """Test retrieving a preprocessor instance by ID."""
        # Create a preprocessor
        config_data = {"normalization": "STANDARD"}
        response = self.client.post("/preprocessors", json=config_data)
        preprocessor_id = response.json()["preprocessor_id"]
        
        # Get preprocessor
        preprocessor = get_preprocessor(preprocessor_id)
        
        # Check result
        self.assertIsInstance(preprocessor, DataPreprocessor)
        self.assertEqual(preprocessor, active_preprocessors[preprocessor_id])

    def test_get_preprocessor_not_found(self):
        """Test that get_preprocessor raises HTTPException when preprocessor not found."""
        from fastapi import HTTPException
        
        with self.assertRaises(HTTPException) as context:
            get_preprocessor("nonexistent-id")
        
        self.assertIn("Preprocessor with ID nonexistent-id not found", str(context.exception.detail))


if __name__ == "__main__":
    unittest.main()