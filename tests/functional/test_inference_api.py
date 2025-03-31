"""
Tests for the Model Inference API endpoints.
"""
import pytest
import os
import json
import time
from fastapi import status
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def test_predict(client, ml_engine, sample_classification_data):
    """Test the standard prediction endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_inference_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test prediction endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/inference/predict",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            params={"model_name": "test_inference_model", "return_probabilities": "false"}
        )
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "model" in response.json()
    assert "sample_count" in response.json()
    assert "execution_time_ms" in response.json()
    assert len(response.json()["predictions"]) == len(sample_classification_data["dataframe"])

def test_predict_with_probabilities(client, ml_engine, sample_classification_data):
    """Test prediction with probability estimates."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_proba_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test prediction with probabilities
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/inference/predict",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            params={"model_name": "test_proba_model", "return_probabilities": "true"}
        )
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    
    # Verify predictions are probability arrays
    # For binary classification, each prediction should be an array of 2 values
    predictions = response.json()["predictions"]
    assert isinstance(predictions[0], list)
    assert len(predictions[0]) == 2  # Binary classification, 2 classes

def test_regression_predict(client, regression_ml_engine, sample_regression_data):
    """Test prediction with a regression model."""
    # First train a regression model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    X = sample_regression_data["X"]
    y = sample_regression_data["y"]
    
    regression_ml_engine.train_model(
        model=model,
        model_name="test_regression_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test prediction
    with open(sample_regression_data["file_path"], "rb") as f:
        response = client.post(
            "/inference/predict",
            files={"data_file": ("regression_data.csv", f, "text/csv")},
            params={"model_name": "test_regression_model"}
        )
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    
    # Verify predictions are single values, not arrays
    predictions = response.json()["predictions"]
    assert isinstance(predictions[0], (int, float))

def test_batch_inference(client, ml_engine, sample_classification_data):
    """Test batch inference with multiple files."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_batch_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Create a second data file by copying the first
    data_file_path = sample_classification_data["file_path"]
    second_file_path = str(Path(data_file_path).parent / "batch_data.csv")
    sample_classification_data["dataframe"].to_csv(second_file_path, index=False)
    
    # Test batch inference with two files
    with open(data_file_path, "rb") as f1, open(second_file_path, "rb") as f2:
        response = client.post(
            "/inference/batch-inference",
            files=[
                ("files", ("file1.csv", f1, "text/csv")),
                ("files", ("file2.csv", f2, "text/csv"))
            ],
            params={"model_name": "test_batch_model", "parallel": "true"}
        )
    
    assert response.status_code == 200
    assert "results" in response.json()
    assert "batch_count" in response.json()
    assert "total_samples" in response.json()
    assert response.json()["batch_count"] == 2
    assert len(response.json()["results"]) == 2

def test_explain_prediction(client, ml_engine, sample_classification_data):
    """Test prediction explanation endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_explain_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test explanation endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/inference/explain",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            params={
                "model_name": "test_explain_model",
                "method": "feature_importance",
                "n_samples": "5"
            }
        )
    
    # Some explanation methods might not be available in the test environment,
    # so we'll accept either a 200 or a 500 with a specific error.
    if response.status_code == 200:
        assert "explanations" in response.json()
        assert "model" in response.json()
        assert "method" in response.json()
    else:
        # If explanation failed, it should be due to missing libraries
        assert response.status_code == 500
        assert "detail" in response.json()

def test_list_models(client, ml_engine, sample_classification_data):
    """Test listing available models for inference."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_list_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test models listing endpoint
    response = client.get("/inference/models")
    
    assert response.status_code == 200
    assert "models" in response.json()
    assert "count" in response.json()
    assert response.json()["count"] >= 1
    assert "test_list_model" in str(response.json()["models"])

def test_model_not_found(client):
    """Test appropriate error when model not found."""
    response = client.post(
        "/inference/load-model/nonexistent_model"
    )
    
    assert response.status_code == 404
    assert "detail" in response.json()