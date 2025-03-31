"""
Tests for the Model Monitoring API endpoints.
"""
import pytest
import os
import json
import pandas as pd
import numpy as np
from fastapi import status
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def test_data_drift(client, ml_engine, sample_classification_data):
    """Test data drift detection endpoint."""
    # First train a model to have reference data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_drift_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Create slightly drifted data
    df = sample_classification_data["dataframe"].copy()
    df["feature1"] = df["feature1"] + 0.5  # Add a constant shift
    
    # Save drifted data to file
    drift_file_path = str(Path("./test_data") / "drift_data.csv")
    df.to_csv(drift_file_path, index=False)
    
    # Test drift detection
    with open(drift_file_path, "rb") as new_f, open(sample_classification_data["file_path"], "rb") as ref_f:
        response = client.post(
            "/monitoring/data-drift",
            files={
                "new_data": ("drift_data.csv", new_f, "text/csv"),
                "reference_data": ("reference_data.csv", ref_f, "text/csv")
            },
            params={"drift_threshold": "0.1"}
        )
    
    assert response.status_code == 200
    assert "feature_drift" in response.json()
    assert "dataset_drift" in response.json()
    assert "drifted_features" in response.json()
    
    # Feature1 should be detected as drifted
    assert "feature1" in response.json()["drifted_features"]

def test_feature_importance(client, ml_engine, sample_classification_data):
    """Test feature importance analysis endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_feature_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test feature importance endpoint
    response = client.post(
        "/monitoring/feature-importance",
        params={
            "model_name": "test_feature_model",
            "top_n": "10",
            "include_plots": "true"
        }
    )
    
    assert response.status_code == 200
    assert "feature_importance" in response.json()
    assert "top_features" in response.json()
    
    # Check that all features are ranked
    assert len(response.json()["feature_importance"]) >= 2  # At least the 2 features we have

def test_error_analysis(client, ml_engine, sample_classification_data):
    """Test error analysis endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_error_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test error analysis endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/monitoring/error-analysis",
            files={"test_data": ("classification_data.csv", f, "text/csv")},
            params={
                "model_name": "test_error_model",
                "n_samples": "10",
                "include_plots": "true"
            }
        )
    
    # The model might be too accurate for the simple dataset, resulting in no errors
    # So accept either a successful analysis or a message about no errors
    assert response.status_code == 200
    
    if "error_count" in response.json():
        if response.json()["error_count"] > 0:
            assert "detailed_samples" in response.json()
        else:
            assert response.json()["error_rate"] == 0.0

def test_model_health_check(client, ml_engine, sample_classification_data):
    """Test model health check endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_health_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test health check endpoint
    with open(sample_classification_data["file_path"], "rb") as f:
        response = client.post(
            "/monitoring/model-health-check",
            files={"test_data": ("classification_data.csv", f, "text/csv")},
            params={
                "model_name": "test_health_model",
                "check_drift": "true",
                "check_performance": "true",
                "check_resources": "true"
            }
        )
    
    assert response.status_code == 200
    assert "status" in response.json()
    assert "checks" in response.json()
    assert "model_name" in response.json()
    assert "timestamp" in response.json()
    
    # Verify model_exists check passed
    model_exists_check = next((check for check in response.json()["checks"] if check["name"] == "model_exists"), None)
    assert model_exists_check is not None
    assert model_exists_check["status"] == "passed"

def test_compare_models(client, ml_engine, sample_classification_data):
    """Test model comparison endpoint."""
    # First train two models
    models = [
        ("rf_compare_1", RandomForestClassifier(n_estimators=10, random_state=42)),
        ("rf_compare_2", RandomForestClassifier(n_estimators=20, random_state=42))
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
        "/monitoring/compare-models",
        json={
            "model_names": ["rf_compare_1", "rf_compare_2"],
            "metrics": ["accuracy", "f1"],
            "include_plots": True
        }
    )
    
    assert response.status_code == 200
    assert "models" in response.json()
    assert "data" in response.json()
    
    # Verify both models are in the comparison
    assert "rf_compare_1" in response.json()["data"]
    assert "rf_compare_2" in response.json()["data"]

def test_generate_report(client, ml_engine, sample_classification_data):
    """Test report generation endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_report_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test report generation endpoint
    response = client.post(
        "/monitoring/generate-report",
        params={"include_plots": "true"}
    )
    
    assert response.status_code == 200
    assert "report_path" in response.json()
    assert "model_count" in response.json()
    assert "report_content" in response.json()
    
    # Verify report file exists
    assert os.path.exists(response.json()["report_path"])

def test_performance_history(client, ml_engine, sample_classification_data):
    """Test performance history endpoint."""
    # First train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    ml_engine.train_model(
        model=model,
        model_name="test_history_model",
        param_grid={"n_estimators": [10]},
        X=X,
        y=y
    )
    
    # Test performance history endpoint
    response = client.get(
        "/monitoring/performance-history",
        params={
            "model_name": "test_history_model",
            "metric": "accuracy",
            "period": "week"
        }
    )
    
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "metric" in response.json()
    assert "dates" in response.json()
    assert "values" in response.json()
    assert "trend" in response.json()
    
    # Should have 7 data points for a week
    assert len(response.json()["dates"]) == 7
    assert len(response.json()["values"]) == 7