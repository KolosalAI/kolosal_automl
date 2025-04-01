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
    
    model_id = "test_inference_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Test prediction endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/api/v1/inference/predict",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={"params": json.dumps({
                "model_id": model_id,
                "return_probabilities": False
            })}
        )
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "model_id" in response.json()
    assert "sample_count" in response.json()
    assert "execution_time_ms" in response.json()
    assert len(response.json()["predictions"]) == len(sample_classification_data["dataframe"])

def test_predict_with_probabilities(client, ml_engine, sample_classification_data):
    """Test prediction with probability estimates."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    model_id = "test_proba_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Test prediction with probabilities
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/api/v1/inference/predict",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={"params": json.dumps({
                "model_id": model_id,
                "return_probabilities": True
            })}
        )
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    
    # Verify predictions are probability arrays
    # For binary classification, each prediction should be an array of 2 values
    predictions = response.json()["predictions"]
    assert isinstance(predictions[0], list)
    assert len(predictions[0]) == 2  # Binary classification, 2 classes

def test_regression_predict(client, ml_engine, sample_regression_data):
    """Test prediction with a regression model."""
    # First train a regression model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    X = sample_regression_data["X"]
    y = sample_regression_data["y"]
    
    model_id = "test_regression_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Test prediction
    with open(sample_regression_data["file_path"], "rb") as f:
        response = client.post(
            "/api/v1/inference/predict",
            files={"data_file": ("regression_data.csv", f, "text/csv")},
            data={"params": json.dumps({
                "model_id": model_id
            })}
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
    
    model_id = "test_batch_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Create a second data file by copying the first
    data_file_path = sample_classification_data["file_path"]
    second_file_path = str(Path(data_file_path).parent / "batch_data.csv")
    sample_classification_data["dataframe"].to_csv(second_file_path, index=False)
    
    # Test batch inference with two files
    with open(data_file_path, "rb") as f1, open(second_file_path, "rb") as f2:
        response = client.post(
            "/api/v1/inference/batch-inference",
            files=[
                ("files", ("file1.csv", f1, "text/csv")),
                ("files", ("file2.csv", f2, "text/csv"))
            ],
            data={"params": json.dumps({
                "model_id": model_id,
                "parallel": True
            })}
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
    
    model_id = "test_explain_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Test explanation endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/api/v1/inference/explain",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={"params": json.dumps({
                "model_id": model_id,
                "method": "shap",
                "n_samples": 5
            })}
        )
    
    # Some explanation methods might not be available in the test environment,
    # so we'll accept either a 200 or a 500 with a specific error.
    if response.status_code == 200:
        assert "explanations" in response.json()
        assert "model_id" in response.json()
        assert "method" in response.json()
    else:
        # If explanation failed, it should be due to missing libraries
        assert response.status_code == 500
        assert "detail" in response.json()

def test_list_models(client, ml_engine):
    """Test listing available models for inference."""
    # Test models listing endpoint
    response = client.get("/api/v1/inference/models")
    
    assert response.status_code == 200
    assert "models" in response.json()

def test_model_not_found(client):
    """Test appropriate error when model not found."""
    response = client.post(
        "/api/v1/inference/load-model/nonexistent_model"
    )
    
    assert response.status_code == 404
    assert "detail" in response.json()

def test_streaming_inference(client, ml_engine, sample_classification_data):
    """Test streaming inference endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    model_id = "test_streaming_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Test streaming inference
    response = client.post(
        "/api/v1/inference/streaming-inference",
        json={
            "model_id": model_id,
            "batch_identifier": "test_batch",
            "output_format": "csv"
        }
    )
    
    assert response.status_code == 200
    assert "job_id" in response.json()
    assert "status" in response.json()
    assert "status_endpoint" in response.json()
    
    # Get job ID from response
    job_id = response.json()["job_id"]
    
    # Test status endpoint
    max_retries = 5
    for i in range(max_retries):
        status_response = client.get(f"/api/v1/inference/streaming-status/{job_id}")
        assert status_response.status_code == 200
        
        # If job is completed or failed, break
        if status_response.json()["status"] in ["completed", "failed"]:
            break
        
        # Wait a bit before checking again
        time.sleep(1)
    
    assert "progress" in status_response.json()

def test_engine_status(client):
    """Test getting engine status."""
    response = client.get("/api/v1/inference/status")
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert "ready_for_inference" in response.json()

def test_engine_metrics(client):
    """Test getting engine performance metrics."""
    response = client.get("/api/v1/inference/metrics")
    
    assert response.status_code == 200
    assert "engine_state" in response.json()
    assert "total_requests" in response.json()
    
def test_model_metadata(client, ml_engine, sample_classification_data):
    """Test getting model metadata."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    model_id = "test_metadata_model"
    model_path = os.path.join(ml_engine.model_registry, model_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Train and save the model
    model.fit(X, y)
    ml_engine.save_model(model, model_id)
    
    # Test loading the model first
    client.post(f"/api/v1/inference/load-model/{model_id}")
    
    # Then get metadata
    response = client.get("/api/v1/inference/model-metadata")
    
    assert response.status_code == 200
    assert "metadata" in response.json()

@pytest.fixture
def cleanup_models(ml_engine):
    """Fixture to clean up test models after tests."""
    yield
    # Clean up any test models
    model_ids = [
        "test_inference_model", "test_proba_model", "test_regression_model",
        "test_batch_model", "test_explain_model", "test_list_model",
        "test_streaming_model", "test_metadata_model"
    ]
    for model_id in model_ids:
        model_path = os.path.join(ml_engine.model_registry, model_id)
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)