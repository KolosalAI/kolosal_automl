"""
Tests for the ML Training Engine API endpoints.
"""
import pytest
import os
import json
from fastapi import status
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def test_list_models_empty(client):
    """Test listing models when none exist."""
    response = client.get("/ml-engine/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert "count" in response.json()
    assert response.json()["count"] == 0

def test_train_model(client, sample_classification_data):
    """Test training a model endpoint."""
    data_file_path = sample_classification_data["file_path"]
    
    # Define model parameters
    model_params = {
        "model_type": "random_forest",
        "model_name": "test_rf_model",
        "param_grid": {
            "n_estimators": [10],
            "max_depth": [3]
        },
        "test_size": 0.2,
        "stratify": True
    }
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/ml-engine/train-model",
            data={"params": json.dumps(model_params)},
            files={"data_file": ("classification_data.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "training_started"
    assert response.json()["model_name"] == "test_rf_model"

def test_evaluate_model(client, ml_engine, sample_classification_data):
    """Test evaluating a model."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_eval_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Now test evaluation endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/ml-engine/evaluate/test_eval_model",
            files={"test_data": ("classification_data.csv", f, "text/csv")},
            params={"detailed": "true"}
        )
    
    assert response.status_code == 200
    assert "metrics" in response.json()
    assert "model_name" in response.json()
    assert "test_samples" in response.json()

def test_predict(client, ml_engine, sample_classification_data):
    """Test making predictions with a model."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_pred_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Now test prediction endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/ml-engine/predict",
            files={"batch_data": ("classification_data.csv", f, "text/csv")},
            params={"model_name": "test_pred_model", "return_proba": "false"}
        )
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "model_used" in response.json()
    assert "row_count" in response.json()
    assert len(response.json()["predictions"]) == len(sample_classification_data["dataframe"])

def test_feature_importance(client, ml_engine, sample_classification_data):
    """Test generating feature importance report."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_feat_imp_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test feature importance endpoint
    response = client.post(
        "/ml-engine/feature-importance",
        json={"model_name": "test_feat_imp_model", "top_n": 10, "include_plot": True}
    )
    
    assert response.status_code == 200
    assert "feature_importance" in response.json()
    assert "top_features" in response.json()
    assert len(response.json()["top_features"]) <= 10

def test_compare_models(client, ml_engine, sample_classification_data):
    """Test comparing multiple models."""
    # Train two models
    models = [
        ("rf_model_1", RandomForestClassifier(n_estimators=10, random_state=42)),
        ("rf_model_2", RandomForestClassifier(n_estimators=20, random_state=42))
    ]
    
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    for name, model in models:
        ml_engine.train_model(
            model=model,
            model_name=name,
            param_grid={"n_estimators": [10]},
            X=X,
            y=y
        )
    
    # Test model comparison endpoint
    response = client.post(
        "/ml-engine/compare-models",
        json={"model_names": ["rf_model_1", "rf_model_2"], "include_plot": True}
    )
    
    assert response.status_code == 200
    assert "models" in response.json()
    assert "data" in response.json()
    assert "rf_model_1" in response.json()["data"]
    assert "rf_model_2" in response.json()["data"]

def test_delete_model(client, ml_engine, sample_classification_data):
    """Test deleting a model."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_delete_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test deletion endpoint
    response = client.delete("/ml-engine/models/test_delete_model")
    
    assert response.status_code == 200
    assert "message" in response.json()
    
    # Verify model is deleted
    response = client.get("/ml-engine/models")
    assert "test_delete_model" not in response.json()["models"]

def test_model_not_found(client):
    """Test appropriate error when model not found."""
    response = client.get("/ml-engine/models/nonexistent_model")
    assert response.status_code == 404
    assert "detail" in response.json()