from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tempfile
import os
import time
import logging
from datetime import datetime

# Import monitoring and engine modules
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType

router = APIRouter(prefix="/monitoring", tags=["Model Monitoring and Analysis"])

# Data models
class DataDriftParams(BaseModel):
    drift_threshold: float = 0.1
    reference_model: Optional[str] = None
    feature_importance_threshold: float = 0.05

class ErrorAnalysisParams(BaseModel):
    model_name: Optional[str] = None
    n_samples: int = 100
    include_plots: bool = True

class PerformanceMonitoringParams(BaseModel):
    model_name: Optional[str] = None
    metrics: List[str] = []
    period: str = "day"  # "hour", "day", "week", "month"

class ModelHealthCheckParams(BaseModel):
    model_name: Optional[str] = None
    check_drift: bool = True
    check_performance: bool = True
    check_resources: bool = True

class CompareModelsParams(BaseModel):
    model_names: List[str] = None
    metrics: List[str] = None
    include_plots: bool = True

# Dependency to get ML training engine instance
def get_ml_engine():
    config = MLTrainingEngineConfig()
    return MLTrainingEngine(config)

@router.post("/data-drift")
async def detect_data_drift(
    new_data: UploadFile = File(...),
    reference_data: Optional[UploadFile] = File(None),
    drift_threshold: float = Form(0.1),
    reference_model: Optional[str] = Form(None),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Detect data drift between new data and reference data or training data"""
    # Save new data temporarily
    new_data_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reference_data_file = None
    
    try:
        # Handle new data
        contents = await new_data.read()
        new_data_file.write(contents)
        new_data_file.close()
        new_df = pd.read_csv(new_data_file.name)
        
        # Handle reference data if provided
        if reference_data:
            reference_data_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            contents = await reference_data.read()
            reference_data_file.write(contents)
            reference_data_file.close()
            ref_df = pd.read_csv(reference_data_file.name)
        else:
            ref_df = None  # Will use training data
        
        # Detect drift
        drift_results = engine.detect_data_drift(
            new_data=new_df,
            reference_data=ref_df,
            drift_threshold=drift_threshold,
            reference_model=reference_model
        )
        
        # If there was an error
        if "error" in drift_results:
            raise HTTPException(status_code=400, detail=drift_results["error"])
        
        # Return the drift results
        return drift_results
    
    finally:
        # Clean up temporary files
        if os.path.exists(new_data_file.name):
            os.unlink(new_data_file.name)
        if reference_data_file and os.path.exists(reference_data_file.name):
            os.unlink(reference_data_file.name)

@router.post("/error-analysis")
async def perform_error_analysis(
    test_data: UploadFile = File(...),
    params: ErrorAnalysisParams = Depends(),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Perform error analysis on test data"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await test_data.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Extract X and y from dataframe (assuming last column is target)
        y_test = df.iloc[:, -1]
        X_test = df.iloc[:, :-1]
        
        # Create a temporary report file
        output_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"error_analysis_{int(time.time())}.md")
        
        # Perform error analysis
        analysis = engine.perform_error_analysis(
            model_name=params.model_name,  # Uses best model if None
            X_test=X_test,
            y_test=y_test,
            n_samples=params.n_samples,
            include_plot=params.include_plots,
            output_file=output_file
        )
        
        # If there was an error
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # Add report path if it was generated
        if os.path.exists(output_file):
            analysis["report_path"] = output_file
        
        return analysis
    
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.post("/feature-importance")
async def analyze_feature_importance(
    model_name: Optional[str] = None,
    top_n: int = 20,
    include_plots: bool = True,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Analyze feature importance for a model"""
    # Create a temporary report file
    output_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"feature_importance_{int(time.time())}.md")
    
    # Generate feature importance report
    result = engine.generate_feature_importance_report(
        model_name=model_name,  # Uses best model if None
        top_n=top_n,
        include_plot=include_plots,
        output_file=output_file
    )
    
    # If there was an error
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Add report path if it was generated
    if os.path.exists(output_file):
        result["report_path"] = output_file
    
    return result

@router.post("/model-health-check")
async def model_health_check(
    test_data: Optional[UploadFile] = File(None),
    params: ModelHealthCheckParams = Depends(),
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Comprehensive health check for a model"""
    temp_file = None
    try:
        # Load test data if provided
        if test_data:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            contents = await test_data.read()
            temp_file.write(contents)
            temp_file.close()
            
            # Read data
            df = pd.read_csv(temp_file.name)
            
            # Extract X and y from dataframe (assuming last column is target)
            y_test = df.iloc[:, -1]
            X_test = df.iloc[:, :-1]
        else:
            X_test = None
            y_test = None
        
        # Collect system resources data
        system_resources = {}
        if params.check_resources:
            try:
                import psutil
                system_resources = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            except ImportError:
                system_resources = {"error": "psutil library not available"}
        
        # Model health check
        health_report = {
            "model_name": params.model_name,
            "timestamp": datetime.now().isoformat(),
            "system_resources": system_resources,
            "status": "pending",
            "checks": []
        }
        
        # Basic model check
        model_check = {"name": "model_exists", "status": "pending"}
        try:
            if params.model_name:
                if params.model_name in engine.models:
                    model_check["status"] = "passed"
                else:
                    model_check["status"] = "failed"
                    model_check["message"] = f"Model {params.model_name} not found"
            else:
                if engine.best_model:
                    model_check["status"] = "passed"
                    health_report["model_name"] = engine.best_model
                else:
                    model_check["status"] = "failed"
                    model_check["message"] = "No models available"
        except Exception as e:
            model_check["status"] = "error"
            model_check["message"] = str(e)
        
        health_report["checks"].append(model_check)
        
        # Skip further checks if model check failed
        if model_check["status"] != "passed":
            health_report["status"] = "unhealthy"
            return health_report
        
        # Check model performance if test data is provided
        if X_test is not None and params.check_performance:
            performance_check = {"name": "performance", "status": "pending"}
            try:
                metrics = engine.evaluate_model(
                    model_name=params.model_name,
                    X_test=X_test,
                    y_test=y_test,
                    detailed=True
                )
                
                if "error" in metrics:
                    performance_check["status"] = "failed"
                    performance_check["message"] = metrics["error"]
                else:
                    performance_check["status"] = "passed"
                    performance_check["metrics"] = metrics
                    
                    # Determine if performance is good enough
                    if engine.config.task_type == TaskType.CLASSIFICATION:
                        if metrics.get("accuracy", 0) < 0.7:
                            performance_check["status"] = "warning"
                            performance_check["message"] = f"Accuracy is below threshold: {metrics.get('accuracy', 0):.4f}"
                    elif engine.config.task_type == TaskType.REGRESSION:
                        if metrics.get("r2", 0) < 0.5:
                            performance_check["status"] = "warning"
                            performance_check["message"] = f"R² is below threshold: {metrics.get('r2', 0):.4f}"
            
            except Exception as e:
                performance_check["status"] = "error"
                performance_check["message"] = str(e)
            
            health_report["checks"].append(performance_check)
        
        # Check for data drift if test data is provided and drift check is enabled
        if X_test is not None and params.check_drift:
            drift_check = {"name": "data_drift", "status": "pending"}
            try:
                drift_results = engine.detect_data_drift(
                    new_data=X_test,
                    reference_data=None,  # Use training data as reference
                    drift_threshold=0.2  # Higher threshold for health check
                )
                
                if "error" in drift_results:
                    drift_check["status"] = "skipped"
                    drift_check["message"] = drift_results["error"]
                elif drift_results["drift_detected"]:
                    drift_check["status"] = "warning"
                    drift_check["message"] = f"Data drift detected: {drift_results['dataset_drift']:.4f}"
                    drift_check["drift_score"] = drift_results["dataset_drift"]
                    drift_check["drifted_features"] = drift_results["drifted_features"]
                else:
                    drift_check["status"] = "passed"
                    drift_check["drift_score"] = drift_results["dataset_drift"]
            
            except Exception as e:
                drift_check["status"] = "error"
                drift_check["message"] = str(e)
            
            health_report["checks"].append(drift_check)
        
        # Determine overall health status
        failed_checks = [check for check in health_report["checks"] if check["status"] == "failed"]
        warning_checks = [check for check in health_report["checks"] if check["status"] == "warning"]
        
        if failed_checks:
            health_report["status"] = "unhealthy"
        elif warning_checks:
            health_report["status"] = "degraded"
        else:
            health_report["status"] = "healthy"
        
        return health_report
    
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.post("/generate-report")
async def generate_comprehensive_report(
    include_plots: bool = True,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Generate a comprehensive report of all models"""
    output_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"model_report_{int(time.time())}.md")
    
    # Generate the report
    report_path = engine.generate_reports(
        output_file=output_file,
        include_plots=include_plots
    )
    
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=500, detail="Failed to generate report")
    
    # Read the report content
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    return {
        "report_path": report_path,
        "model_count": len(engine.models),
        "best_model": engine.best_model,
        "report_content": report_content[:1000] + "..." if len(report_content) > 1000 else report_content
    }

@router.post("/compare-models")
async def compare_models(
    params: CompareModelsParams,
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Compare multiple trained models"""
    output_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"model_comparison_{int(time.time())}.md")
    
    # Compare models
    comparison = engine.compare_models(
        model_names=params.model_names,
        metrics=params.metrics,
        include_plot=params.include_plots,
        output_file=output_file
    )
    
    # If there was an error
    if "error" in comparison:
        raise HTTPException(status_code=400, detail=comparison["error"])
    
    # Add report path if it was generated
    if os.path.exists(output_file):
        comparison["report_path"] = output_file
    
    return comparison

@router.get("/performance-history")
async def get_performance_history(
    model_name: Optional[str] = None,
    metric: str = "accuracy",
    period: str = "week",
    engine: MLTrainingEngine = Depends(get_ml_engine)
):
    """Get historical performance data for a model"""
    # This would normally load from a database of logged metrics
    # For demonstration, we'll return mock data
    
    # Determine model to use
    if model_name is None and engine.best_model:
        model_name = engine.best_model
    elif model_name and model_name not in engine.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Mock data for demonstration
    from datetime import datetime, timedelta
    
    # Create date points based on period
    today = datetime.now()
    if period == "day":
        date_points = [today - timedelta(hours=i) for i in range(24)]
        date_format = "%H:%M"
    elif period == "week":
        date_points = [today - timedelta(days=i) for i in range(7)]
        date_format = "%a"
    elif period == "month":
        date_points = [today - timedelta(days=i) for i in range(30)]
        date_format = "%d %b"
    else:
        date_points = [today - timedelta(days=i*7) for i in range(12)]
        date_format = "%d %b"
    
    # Format dates for display
    date_labels = [d.strftime(date_format) for d in date_points]
    
    # Generate mock metric values with some randomness but trending downward for older data
    import random
    if engine.config.task_type == TaskType.CLASSIFICATION:
        base_value = 0.85  # Example for accuracy
        metric_values = [max(0, min(1, base_value - 0.002 * i + random.uniform(-0.03, 0.03))) for i in range(len(date_points))]
    else:
        base_value = 0.2  # Example for MSE
        metric_values = [max(0, base_value + 0.005 * i + random.uniform(-0.02, 0.02)) for i in range(len(date_points))]
    
    # Reverse lists to have chronological order
    date_labels.reverse()
    metric_values.reverse()
    
    return {
        "model_name": model_name,
        "metric": metric,
        "period": period,
        "dates": date_labels,
        "values": metric_values,
        "current_value": metric_values[-1] if metric_values else None,
        "trend": "stable" if abs(metric_values[-1] - metric_values[0]) < 0.05 else 
                 "improving" if metric_values[-1] > metric_values[0] else "degrading"
    }