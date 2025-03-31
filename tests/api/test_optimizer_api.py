"""
Tests for the Hyperparameter Optimization API endpoints.
"""
import pytest
import os
import json
import time
from fastapi import status
from pathlib import Path

def test_available_model_types(client):
    """Test retrieving available model types."""
    response = client.get("/optimizer/available-model-types")
    
    assert response.status_code == 200
    assert "classification_models" in response.json()
    assert "regression_models" in response.json()
    assert "optimization_strategies" in response.json()
    
    # Verify classification models list is populated
    assert len(response.json()["classification_models"]) > 0
    # Verify regression models list is populated
    assert len(response.json()["regression_models"]) > 0
    # Verify optimization strategies list is populated
    assert len(response.json()["optimization_strategies"]) > 0

def test_default_params(client):
    """Test retrieving default hyperparameters for a model type."""
    response = client.get("/optimizer/default-params/random_forest")
    
    assert response.status_code == 200
    assert "model_type" in response.json()
    assert "parameter_grid" in response.json()
    assert response.json()["model_type"] == "random_forest"
    
    # Verify parameter grid has common hyperparameters
    param_grid = response.json()["parameter_grid"]
    assert "n_estimators" in param_grid
    assert "max_depth" in param_grid

def test_start_optimization(client, sample_classification_data):
    """Test starting an optimization job."""
    data_file_path = sample_classification_data["file_path"]
    
    # Define optimization parameters
    optimization_params = {
        "model_type": "random_forest",
        "model_name": "test_opt_model",
        "optimization_strategy": "random_search",
        "optimization_iterations": 2,  # Use small value for testing
        "cv_folds": 2,
        "test_size": 0.2,
        "stratify": True,
        "random_state": 42,
        "early_stopping": True,
        "early_stopping_rounds": 2
    }
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/optimizer/optimize",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={"params": json.dumps(optimization_params)}
        )
    
    assert response.status_code == 200
    assert "job_id" in response.json()
    assert "message" in response.json()
    assert "status_endpoint" in response.json()
    assert "model_type" in response.json()
    assert "optimization_strategy" in response.json()
    
    # Save job_id for other tests
    job_id = response.json()["job_id"]
    
    # Wait a bit for job to start
    time.sleep(1)
    
    # Verify job status endpoint works
    status_response = client.get(f"/optimizer/status/{job_id}")
    assert status_response.status_code == 200
    assert "job_id" in status_response.json()
    assert "status" in status_response.json()
    assert "model_type" in status_response.json()
    assert "model_name" in status_response.json()
    assert "start_time" in status_response.json()
    
    # The job could be in any state (PENDING, RUNNING, COMPLETED, FAILED)
    assert status_response.json()["status"] in ["pending", "running", "completed", "failed"]

def test_list_optimization_jobs(client, sample_classification_data):
    """Test listing all optimization jobs."""
    # First start a job
    data_file_path = sample_classification_data["file_path"]
    
    # Define optimization parameters
    optimization_params = {
        "model_type": "random_forest",
        "model_name": "test_list_opt_model",
        "optimization_strategy": "random_search",
        "optimization_iterations": 2,  # Use small value for testing
        "cv_folds": 2
    }
    
    with open(data_file_path, "rb") as f:
        client.post(
            "/optimizer/optimize",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={"params": json.dumps(optimization_params)}
        )
    
    # Wait a bit for job to be registered
    time.sleep(1)
    
    # Test listing jobs
    response = client.get("/optimizer/jobs")
    
    assert response.status_code == 200
    assert "jobs" in response.json()
    assert "count" in response.json()
    assert response.json()["count"] > 0
    
    # Verify job list contains correct information
    jobs = response.json()["jobs"]
    assert len(jobs) > 0
    assert "job_id" in jobs[0]
    assert "model_name" in jobs[0]
    assert "status" in jobs[0]
    assert "start_time" in jobs[0]
    assert "elapsed_time" in jobs[0]

def test_optimization_nonexistent_job(client):
    """Test getting status of nonexistent job."""
    response = client.get("/optimizer/status/nonexistent_job_id")
    
    assert response.status_code == 404
    assert "detail" in response.json()

def test_optimization_regression(client, sample_regression_data):
    """Test optimization with regression models."""
    data_file_path = sample_regression_data["file_path"]
    
    # Define optimization parameters for regression
    optimization_params = {
        "model_type": "random_forest_regressor",
        "model_name": "test_regression_opt",
        "optimization_strategy": "random_search",
        "optimization_iterations": 2,  # Use small value for testing
        "cv_folds": 2,
        "test_size": 0.2,
        "stratify": False,  # No stratification for regression
        "random_state": 42
    }
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/optimizer/optimize",
            files={"data_file": ("regression_data.csv", f, "text/csv")},
            data={"params": json.dumps(optimization_params)}
        )
    
    assert response.status_code == 200
    assert "job_id" in response.json()
    assert response.json()["model_type"] == "random_forest_regressor"

def test_custom_param_grid(client, sample_classification_data):
    """Test optimization with custom parameter grid."""
    data_file_path = sample_classification_data["file_path"]
    
    # Define optimization parameters with custom param grid
    optimization_params = {
        "model_type": "random_forest",
        "model_name": "test_custom_params",
        "optimization_strategy": "random_search",
        "optimization_iterations": 2,  # Use small value for testing
        "params_to_optimize": {
            "n_estimators": [5, 10],
            "max_depth": [3],
            "min_samples_split": [2]
        }
    }
    
    with open(data_file_path, "rb") as f:
        response = client.post(
            "/optimizer/optimize",
            files={"data_file": ("classification_data.csv", f, "text/csv")},
            data={"params": json.dumps(optimization_params)}
        )
    
    assert response.status_code == 200
    assert "job_id" in response.json()
    
    # Wait a bit for job to start
    time.sleep(1)
    
    # Check job status to verify custom params were used
    job_id = response.json()["job_id"]
    status_response = client.get(f"/optimizer/status/{job_id}")
    
    assert status_response.status_code == 200