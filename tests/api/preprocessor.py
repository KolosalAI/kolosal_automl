"""
Tests for the Data Preprocessor API endpoints.
"""
import pytest
import os
import json
from fastapi import status

def test_analyze_data(client, sample_classification_data):
    """Test data analysis endpoint."""
    data_file_path = sample_classification_data["file_path"]
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/preprocessor/analyze",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            params={"sample_rows": "5", "include_stats": "true", "include_correlation": "true"}
        )
    
    assert response.status_code == 200
    assert "shape" in response.json()
    assert "columns" in response.json()
    assert "dtypes" in response.json()
    assert "missing_values" in response.json()
    assert "sample_data" in response.json()
    assert "numeric_stats" in response.json()
    assert "correlation" in response.json()
    
    # Check correct dimensions
    assert response.json()["shape"]["rows"] == len(sample_classification_data["dataframe"])
    assert response.json()["shape"]["columns"] == len(sample_classification_data["dataframe"].columns)

def test_preprocess_data(client, sample_classification_data):
    """Test data preprocessing endpoint."""
    data_file_path = sample_classification_data["file_path"]
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/preprocessor/preprocess",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            params={
                "normalization": "standard",
                "handle_outliers": "true",
                "handle_missing": "true"
            }
        )
    
    assert response.status_code == 200
    assert "statistics" in response.json()
    assert "output_path" in response.json()
    
    # Verify the output file exists
    assert os.path.exists(response.json()["output_path"])
    
    # Verify statistics
    stats = response.json()["statistics"]
    assert stats["original_shape"]["rows"] == len(sample_classification_data["dataframe"])
    assert stats["original_shape"]["columns"] == len(sample_classification_data["dataframe"].columns)
    assert stats["normalization"] == "standard"

def test_feature_selection(client, sample_classification_data):
    """Test feature selection endpoint."""
    data_file_path = sample_classification_data["file_path"]
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/preprocessor/feature-selection",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={
                "target_column": "target",
                "method": "mutual_info",
                "top_k": "2"  # Select top 2 features
            }
        )
    
    assert response.status_code == 200
    assert "selected_features" in response.json()
    assert "feature_importance" in response.json()
    assert "original_feature_count" in response.json()
    assert "selected_feature_count" in response.json()
    assert "output_path" in response.json()
    
    # Verify output file exists
    assert os.path.exists(response.json()["output_path"])
    
    # Verify correct number of features selected
    assert len(response.json()["selected_features"]) == 2
    assert response.json()["selected_feature_count"] == 2
    assert response.json()["original_feature_count"] == 2  # Original dataset had 2 features

def test_feature_selection_invalid_target(client, sample_classification_data):
    """Test feature selection with invalid target column."""
    data_file_path = sample_classification_data["file_path"]
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/preprocessor/feature-selection",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={
                "target_column": "nonexistent_column",  # Invalid column name
                "method": "mutual_info",
                "top_k": "2"
            }
        )
    
    assert response.status_code == 400
    assert "detail" in response.json()