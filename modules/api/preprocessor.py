from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tempfile
import os
import logging
from enum import Enum

# Import preprocessor module
from modules.engine.data_preprocessor import DataPreprocessor
from modules.configs import PreprocessorConfig, NormalizationType

router = APIRouter(prefix="/preprocessor", tags=["Data Preprocessing"])

# Data models
class PreprocessingParams(BaseModel):
    normalization: Optional[str] = "standard"
    handle_outliers: Optional[bool] = True
    handle_missing: Optional[bool] = True
    feature_selection: Optional[bool] = False
    feature_selection_method: Optional[str] = "mutual_info"
    categorical_encoding: Optional[str] = "one_hot"
    text_vectorization: Optional[str] = None

class DataAnalysisRequest(BaseModel):
    sample_rows: Optional[int] = 5
    include_stats: Optional[bool] = True
    include_correlation: Optional[bool] = True
    profile_data: Optional[bool] = False

class DatasetStatsResponse(BaseModel):
    shape: Dict[str, int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    sample_data: Optional[Dict[str, List[Any]]] = None
    numeric_stats: Optional[Dict[str, Dict[str, float]]] = None
    correlation: Optional[Dict[str, Dict[str, float]]] = None

# Dependency to get preprocessor instance
def get_preprocessor():
    config = PreprocessorConfig()
    return DataPreprocessor(config)

@router.post("/analyze")
async def analyze_data(
    data_file: UploadFile = File(...),
    params: DataAnalysisRequest = Depends()
):
    """Analyze a dataset and return statistics and insights"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Calculate basic statistics
        result = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns}
        }
        
        # Add sample data if requested
        if params.sample_rows > 0:
            sample = df.head(params.sample_rows)
            result["sample_data"] = {col: sample[col].tolist() for col in sample.columns}
        
        # Add numeric statistics if requested
        if params.include_stats:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats = df[numeric_cols].describe().to_dict()
                # Convert from nested dict of Series to regular nested dict
                numeric_stats = {}
                for col, col_stats in stats.items():
                    numeric_stats[col] = {stat: float(val) for stat, val in col_stats.items()}
                result["numeric_stats"] = numeric_stats
        
        # Add correlation if requested
        if params.include_correlation:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                corr_matrix = df[numeric_cols].corr().to_dict()
                # Convert from nested dict of Series to regular nested dict
                correlation = {}
                for col, col_corr in corr_matrix.items():
                    correlation[col] = {other_col: float(val) for other_col, val in col_corr.items()}
                result["correlation"] = correlation
        
        # Add advanced profiling if requested
        if params.profile_data:
            try:
                from pandas_profiling import ProfileReport
                profile = ProfileReport(df, minimal=True)
                profile_file = f"{temp_file.name}_profile.html"
                profile.to_file(profile_file)
                result["profile_path"] = profile_file
                result["profile_generated"] = True
            except ImportError:
                result["profile_generated"] = False
                result["profile_error"] = "pandas-profiling not installed"
        
        return result
    finally:
        os.unlink(temp_file.name)

@router.post("/preprocess")
async def preprocess_data(
    data_file: UploadFile = File(...),
    params: PreprocessingParams = Depends(),
    preprocessor: DataPreprocessor = Depends(get_preprocessor)
):
    """Preprocess a dataset with specified parameters"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Configure preprocessor based on parameters
        if params.normalization:
            try:
                preprocessor.config.normalization = NormalizationType[params.normalization.upper()]
            except KeyError:
                pass
                
        preprocessor.config.detect_outliers = params.handle_outliers
        preprocessor.config.handle_nan = params.handle_missing
        preprocessor.config.categorical_encoding = params.categorical_encoding
        preprocessor.config.text_vectorization = params.text_vectorization
        
        # Fit preprocessor
        preprocessor.fit(df)
        
        # Transform data
        transformed_df = preprocessor.transform(df)
        
        # If transformed result is not a DataFrame, convert it
        if not isinstance(transformed_df, pd.DataFrame):
            if isinstance(transformed_df, np.ndarray):
                # Try to preserve column names if possible
                if transformed_df.shape[1] == len(df.columns):
                    transformed_df = pd.DataFrame(transformed_df, columns=df.columns)
                else:
                    transformed_df = pd.DataFrame(transformed_df)
        
        # Create temporary file for transformed data
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        transformed_df.to_csv(output_file.name, index=False)
        
        # Read back file content for response
        with open(output_file.name, 'rb') as f:
            processed_data = f.read()
        
        # Return statistics and information about transformation
        stats = {
            "original_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "transformed_shape": {"rows": transformed_df.shape[0], "columns": transformed_df.shape[1]},
            "normalization": params.normalization,
            "features_added": transformed_df.shape[1] - df.shape[1] if transformed_df.shape[1] > df.shape[1] else 0,
            "features_removed": df.shape[1] - transformed_df.shape[1] if df.shape[1] > transformed_df.shape[1] else 0,
            "rows_removed": df.shape[0] - transformed_df.shape[0] if df.shape[0] > transformed_df.shape[0] else 0,
            "processed_columns": transformed_df.columns.tolist()
        }
        
        return {
            "statistics": stats,
            "output_path": output_file.name
        }
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@router.post("/feature-selection")
async def feature_selection(
    data_file: UploadFile = File(...),
    target_column: str = Form(...),
    method: str = Form("mutual_info"),
    top_k: Optional[int] = Form(10),
    preprocessor: DataPreprocessor = Depends(get_preprocessor)
):
    """Select important features from a dataset"""
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        contents = await data_file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Read data
        df = pd.read_csv(temp_file.name)
        
        # Validate target column
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")
        
        # Extract target and features
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Apply feature selection
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
        
        # Determine selection method based on target type
        is_classification = y.dtype == 'object' or len(y.unique()) < 10
        
        if method == "mutual_info":
            if is_classification:
                selector = SelectKBest(mutual_info_classif, k=top_k if top_k < X.shape[1] else X.shape[1])
            else:
                selector = SelectKBest(mutual_info_regression, k=top_k if top_k < X.shape[1] else X.shape[1])
        else:  # f_classif or f_regression
            if is_classification:
                selector = SelectKBest(f_classif, k=top_k if top_k < X.shape[1] else X.shape[1])
            else:
                selector = SelectKBest(f_regression, k=top_k if top_k < X.shape[1] else X.shape[1])
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Get scores for each feature
        feature_scores = selector.scores_
        
        # Create dataframe with selected features only
        X_selected_df = X.iloc[:, selected_indices]
        X_selected_df[target_column] = y  # Add back target column
        
        # Save to temporary file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        X_selected_df.to_csv(output_file.name, index=False)
        
        # Prepare feature importance scores
        importance = {}
        for i, feature in enumerate(X.columns):
            if i < len(feature_scores):
                importance[feature] = float(feature_scores[i])
        
        # Sort features by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "selected_features": selected_features,
            "feature_importance": sorted_importance,
            "original_feature_count": X.shape[1],
            "selected_feature_count": len(selected_features),
            "method": method,
            "is_classification": is_classification,
            "output_path": output_file.name
        }
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)