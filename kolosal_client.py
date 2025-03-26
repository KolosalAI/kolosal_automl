
import os
import requests
import pandas as pd
import json
import time
from typing import Optional, Dict, List, Union, Any, BinaryIO
from dataclasses import dataclass, field
import warnings

@dataclass
class KolosalConfig:
    """Configuration for Kolosal AutoML client"""
    base_url: str = "http://localhost:5000"
    api_prefix: str = "/api"
    token: Optional[str] = None
    timeout: int = 120
    verify_ssl: bool = True
    headers: Dict[str, str] = field(default_factory=dict)

class KolosalAutoML:
    """Python client for Kolosal AutoML API"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:5000",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 120,
        verify_ssl: bool = True
    ):
        """
        Initialize Kolosal AutoML client.
        
        Args:
            base_url: API base URL
            api_key: Direct API key (if not using username/password)
            username: Username for authentication
            password: Password for authentication 
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.config = KolosalConfig(
            base_url=base_url,
            timeout=timeout,
            verify_ssl=verify_ssl
        )
        
        # Set common headers
        self.config.headers = {
            "Accept": "application/json",
            "User-Agent": "KolosalAutoML-PythonClient/1.0"
        }
        
        # Set authorization if provided
        if api_key:
            self.config.token = api_key
            self.config.headers["Authorization"] = f"Bearer {api_key}"
        elif username and password:
            self._login(username, password)
    
    def _login(self, username: str, password: str) -> bool:
        """
        Login to Kolosal API and get authentication token.
        
        Args:
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            True if login successful, False otherwise
        """
        url = f"{self.config.base_url}{self.config.api_prefix}/login"
        payload = {"username": username, "password": password}
        
        try:
            response = requests.post(
                url, 
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            if response.status_code == 200:
                data = response.json()
                self.config.token = data["token"]
                self.config.headers["Authorization"] = f"Bearer {self.config.token}"
                return True
            else:
                raise Exception(f"Login failed: {response.text}")
        except Exception as e:
            raise ConnectionError(f"Login failed: {str(e)}")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        files: Dict[str, BinaryIO] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Kolosal API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json_data: JSON data
            files: Files to upload
            
        Returns:
            API response as dictionary
        """
        url = f"{self.config.base_url}{self.config.api_prefix}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json_data,
                files=files,
                headers=self.config.headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            # Handle common status codes
            if response.status_code == 200 or response.status_code == 201:
                # Check if response is JSON
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type and response.content:
                    return response.json()
                elif "octet-stream" in content_type:
                    # Handle file download
                    return {"file_content": response.content, "filename": self._get_filename_from_headers(response.headers)}
                else:
                    return {"status": "success", "message": response.text}
            elif response.status_code == 400:
                raise ValueError(f"Bad request: {response.text}")
            elif response.status_code == 401:
                raise PermissionError("Authentication required or token expired")
            elif response.status_code == 403:
                raise PermissionError("Not authorized to access this resource")
            elif response.status_code == 404:
                raise FileNotFoundError(f"Resource not found: {endpoint}")
            elif response.status_code == 500:
                raise Exception(f"Server error: {response.text}")
            else:
                raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            raise ConnectionError(f"Request failed: {str(e)}")
    
    def _get_filename_from_headers(self, headers: Dict[str, str]) -> str:
        """Extract filename from Content-Disposition header"""
        cd = headers.get("Content-Disposition", "")
        if "filename=" in cd:
            return cd.split("filename=")[1].strip('"')
        return "downloaded_file"
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check Kolosal API status.
        
        Returns:
            API status information
        """
        return self._make_request("GET", "status")
    
    def list_models(self) -> Dict[str, Any]:
        """
        List all available models.
        
        Returns:
            Dictionary with models list and count
        """
        return self._make_request("GET", "models")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        return self._make_request("GET", f"models/{model_name}")
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Deletion status
        """
        return self._make_request("DELETE", f"models/{model_name}")
    
    def train_model(
        self,
        data: Union[str, pd.DataFrame],
        target_column: str,
        model_type: str = "classification",
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        optimization_strategy: Optional[str] = None,
        optimization_iterations: Optional[int] = None,
        feature_selection: Optional[bool] = None,
        cv_folds: Optional[int] = None,
        random_state: int = 42,
        optimization_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a new machine learning model.
        
        Args:
            data: DataFrame or file path with training data
            target_column: Target column name
            model_type: Type of model (classification or regression)
            model_name: Optional name for the model
            task_type: Optional task type (will be inferred from model_type if not provided)
            test_size: Test set size (0.0 to 1.0)
            optimization_strategy: Strategy for hyperparameter optimization
            optimization_iterations: Number of optimization iterations
            feature_selection: Whether to perform feature selection
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            optimization_mode: Optimization mode for resource utilization
            
        Returns:
            Training results including model metrics
        """
        # Prepare file
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to CSV file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Open file for upload
            files = {"file": open(temp_file.name, "rb")}
            try:
                filepath = temp_file.name
            except:
                # In case of error, cleanup
                os.unlink(temp_file.name)
                raise
        elif isinstance(data, str):
            # Use file path
            files = {"file": open(data, "rb")}
            filepath = data
        else:
            raise ValueError("Data must be a DataFrame or file path")
        
        # Prepare form data
        form_data = {
            "target_column": target_column,
            "model_type": model_type
        }
        
        # Add optional parameters if provided
        if model_name:
            form_data["model_name"] = model_name
        if task_type:
            form_data["task_type"] = task_type
        if test_size:
            form_data["test_size"] = str(test_size)
        if optimization_strategy:
            form_data["optimization_strategy"] = optimization_strategy
        if optimization_iterations:
            form_data["optimization_iterations"] = str(optimization_iterations)
        if feature_selection is not None:
            form_data["feature_selection"] = str(feature_selection).lower()
        if cv_folds:
            form_data["cv_folds"] = str(cv_folds)
        if random_state:
            form_data["random_state"] = str(random_state)
        if optimization_mode:
            form_data["optimization_mode"] = optimization_mode
        
        try:
            # Make the request
            result = self._make_request("POST", "train", data=form_data, files=files)
            return result
        finally:
            # Close file handle
            files["file"].close()
            
            # Remove temporary file if created from DataFrame
            if isinstance(data, pd.DataFrame) and os.path.exists(filepath):
                os.unlink(filepath)
    
    def predict(
        self,
        model: str,
        data: Union[str, pd.DataFrame, List[List[float]]],
        batch_size: int = 0,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            model: Name of the model to use
            data: Data for prediction (DataFrame, file path, or list of feature values)
            batch_size: Batch size for large datasets (0 means direct prediction)
            return_proba: Whether to return probabilities instead of class labels
            
        Returns:
            Prediction results
        """
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to CSV
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Make prediction with file
            files = {"file": open(temp_file.name, "rb")}
            try:
                filepath = temp_file.name
                result = self._make_request(
                    "POST", 
                    f"predict?model={model}&batch_size={batch_size}&return_proba={str(return_proba).lower()}", 
                    files=files
                )
                return result
            finally:
                # Close file handle and cleanup
                files["file"].close()
                if os.path.exists(filepath):
                    os.unlink(filepath)
        
        elif isinstance(data, str):
            # Use file path
            files = {"file": open(data, "rb")}
            try:
                result = self._make_request(
                    "POST", 
                    f"predict?model={model}&batch_size={batch_size}&return_proba={str(return_proba).lower()}", 
                    files=files
                )
                return result
            finally:
                # Close file handle
                files["file"].close()
        
        elif isinstance(data, list):
            # Use JSON data
            json_data = {
                "model": model,
                "data": data,
                "batch_size": batch_size,
                "return_proba": return_proba
            }
            
            return self._make_request("POST", "predict", json_data=json_data)
        
        else:
            raise ValueError("Data must be a DataFrame, file path, or list of values")
    
    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """
        Get model performance metrics.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metrics
        """
        return self._make_request("GET", f"models/{model_name}/metrics")
    
    def compare_models(
        self,
        models: List[str],
        metrics: Optional[List[str]] = None,
        include_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple models' performance.
        
        Args:
            models: List of model names to compare
            metrics: Optional list of metrics to compare
            include_plot: Whether to include visualization plots
            
        Returns:
            Comparison results
        """
        json_data = {
            "models": models,
            "metrics": metrics,
            "include_plot": include_plot
        }
        
        return self._make_request("POST", "models/compare", json_data=json_data)
    
    def error_analysis(
        self,
        model_name: str,
        test_data: Optional[Union[str, pd.DataFrame]] = None,
        n_samples: int = 100,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Perform detailed error analysis on model predictions.
        
        Args:
            model_name: Name of the model
            test_data: Optional test data (DataFrame or file path)
            n_samples: Number of error samples to analyze
            include_plots: Whether to include visualization plots
            
        Returns:
            Error analysis results
        """
        # Create request JSON
        json_data = {
            "n_samples": n_samples,
            "include_plots": include_plots
        }
        
        # If test data is provided
        if test_data is not None:
            if isinstance(test_data, pd.DataFrame):
                # Convert DataFrame to CSV
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
                test_data.to_csv(temp_file.name, index=False)
                temp_file.close()
                
                # Make request with file
                files = {"file": open(temp_file.name, "rb")}
                try:
                    filepath = temp_file.name
                    result = self._make_request(
                        "POST", 
                        f"error-analysis/{model_name}", 
                        json_data=json_data,
                        files=files
                    )
                    return result
                finally:
                    # Close file handle and cleanup
                    files["file"].close()
                    if os.path.exists(filepath):
                        os.unlink(filepath)
            
            elif isinstance(test_data, str):
                # Use file path
                files = {"file": open(test_data, "rb")}
                try:
                    result = self._make_request(
                        "POST", 
                        f"error-analysis/{model_name}", 
                        json_data=json_data,
                        files=files
                    )
                    return result
                finally:
                    # Close file handle
                    files["file"].close()
        else:
            # No test data provided, use cached test data
            return self._make_request("POST", f"error-analysis/{model_name}", json_data=json_data)
    
    def detect_drift(
        self,
        model_name: str,
        new_data: Union[str, pd.DataFrame],
        reference_dataset: Optional[str] = None,
        drift_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and new data.
        
        Args:
            model_name: Name of the model
            new_data: New data to check for drift (DataFrame or file path)
            reference_dataset: Optional reference dataset name (uses training data if None)
            drift_threshold: Threshold for detecting significant drift
            
        Returns:
            Drift detection results
        """
        # Create request JSON
        json_data = {
            "reference_dataset": reference_dataset,
            "drift_threshold": drift_threshold
        }
        
        # Handle new data
        if isinstance(new_data, pd.DataFrame):
            # Convert DataFrame to CSV
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            new_data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Make request with file
            files = {"file": open(temp_file.name, "rb")}
            try:
                filepath = temp_file.name
                result = self._make_request(
                    "POST", 
                    f"drift-detection/{model_name}", 
                    json_data=json_data,
                    files=files
                )
                return result
            finally:
                # Close file handle and cleanup
                files["file"].close()
                if os.path.exists(filepath):
                    os.unlink(filepath)
        
        elif isinstance(new_data, str):
            # Use file path
            files = {"file": open(new_data, "rb")}
            try:
                result = self._make_request(
                    "POST", 
                    f"drift-detection/{model_name}", 
                    json_data=json_data,
                    files=files
                )
                return result
            finally:
                # Close file handle
                files["file"].close()
        else:
            raise ValueError("New data must be a DataFrame or file path")
    
    def feature_importance(
        self,
        model_name: str,
        top_n: int = 20,
        include_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a feature importance report for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to include
            include_plot: Whether to include visualization plot
            
        Returns:
            Feature importance analysis results
        """
        json_data = {
            "top_n": top_n,
            "include_plot": include_plot
        }
        
        return self._make_request("POST", f"feature-importance/{model_name}", json_data=json_data)
    
    def quantize_model(
        self,
        model_name: str,
        quantization_type: str = "int8",
        quantization_mode: str = "dynamic_per_batch"
    ) -> Dict[str, Any]:
        """
        Quantize a model for improved deployment efficiency.
        
        Args:
            model_name: Name of the model to quantize
            quantization_type: Quantization type (int8, float16, etc.)
            quantization_mode: Quantization mode
            
        Returns:
            Quantization results
        """
        json_data = {
            "quantization_type": quantization_type,
            "quantization_mode": quantization_mode
        }
        
        return self._make_request("POST", f"quantize/{model_name}", json_data=json_data)
    
    def export_model(
        self,
        model_name: str,
        format: str = "sklearn",
        include_pipeline: bool = True,
        output_path: Optional[str] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Export a model in different formats.
        
        Args:
            model_name: Name of the model to export
            format: Export format (sklearn, onnx, pmml, tf, torchscript)
            include_pipeline: Whether to include preprocessing pipeline
            output_path: Optional path to save the exported model
            
        Returns:
            Path to exported model if output_path is provided, otherwise file content
        """
        params = {
            "format": format,
            "include_pipeline": str(include_pipeline).lower()
        }
        
        result = self._make_request(
            "GET", 
            f"models/export/{model_name}", 
            params=params
        )
        
        # Check if result contains file
        if "file_content" in result:
            if output_path:
                # Save to output path
                with open(output_path, "wb") as f:
                    f.write(result["file_content"])
                return output_path
            else:
                # Return result with file content
                return result
        
        return result
    
    def preprocess_data(
        self,
        data: Union[str, pd.DataFrame],
        normalize: str = "standard",
        handle_missing: bool = True,
        detect_outliers: bool = True,
        output_path: Optional[str] = None
    ) -> Union[pd.DataFrame, str, Dict[str, Any]]:
        """
        Preprocess data using the data preprocessor.
        
        Args:
            data: Data to preprocess (DataFrame or file path)
            normalize: Normalization type (standard, minmax, robust, none)
            handle_missing: Whether to handle missing values
            detect_outliers: Whether to detect outliers
            output_path: Optional path to save the preprocessed data
            
        Returns:
            Preprocessed data as DataFrame if output_path is None, otherwise path to saved file
        """
        form_data = {
            "normalize": normalize,
            "handle_missing": str(handle_missing).lower(),
            "detect_outliers": str(detect_outliers).lower()
        }
        
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to CSV
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Make request with file
            files = {"file": open(temp_file.name, "rb")}
            try:
                filepath = temp_file.name
                result = self._make_request(
                    "POST", 
                    "preprocess", 
                    data=form_data,
                    files=files
                )
                
                # Check if result contains file
                if "file_content" in result:
                    if output_path:
                        # Save to output path
                        with open(output_path, "wb") as f:
                            f.write(result["file_content"])
                        return output_path
                    else:
                        # Return as DataFrame
                        import io
                        return pd.read_csv(io.BytesIO(result["file_content"]))
                
                return result
            finally:
                # Close file handle and cleanup
                files["file"].close()
                if os.path.exists(filepath):
                    os.unlink(filepath)
        
        elif isinstance(data, str):
            # Use file path
            files = {"file": open(data, "rb")}
            try:
                result = self._make_request(
                    "POST", 
                    "preprocess", 
                    data=form_data,
                    files=files
                )
                
                # Check if result contains file
                if "file_content" in result:
                    if output_path:
                        # Save to output path
                        with open(output_path, "wb") as f:
                            f.write(result["file_content"])
                        return output_path
                    else:
                        # Return as DataFrame
                        import io
                        return pd.read_csv(io.BytesIO(result["file_content"]))
                
                return result
            finally:
                # Close file handle
                files["file"].close()
        else:
            raise ValueError("Data must be a DataFrame or file path")
    
    def get_automl_config(self, optimization_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current AutoML configuration.
        
        Args:
            optimization_mode: Optional optimization mode to use
            
        Returns:
            AutoML configuration
        """
        params = {}
        if optimization_mode:
            params["optimization_mode"] = optimization_mode
        
        return self._make_request("GET", "config", params=params)
    
    def download_model(self, model_name: str, output_path: str) -> str:
        """
        Download a trained model.
        
        Args:
            model_name: Name of the model to download
            output_path: Path to save the downloaded model
            
        Returns:
            Path to the downloaded model file
        """
        result = self._make_request("GET", f"models/{model_name}/download")
        
        # Check if result contains file
        if "file_content" in result:
            # Save to output path
            with open(output_path, "wb") as f:
                f.write(result["file_content"])
            return output_path
        
        raise ValueError("Failed to download model file")
    
    def update_model_metadata(self, model_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update model metadata.
        
        Args:
            model_name: Name of the model
            metadata: Metadata to update
            
        Returns:
            Updated metadata
        """
        return self._make_request("POST", f"models/{model_name}/metadata", json_data=metadata)


# Example usage of the client
if __name__ == "__main__":
    # Initialize client
    client = KolosalAutoML(
        base_url="http://localhost:5000",
        username="admin",
        password="admin123"
    )
    
    # Check API status
    print("API Status:", client.check_status())
    
    # List models
    models = client.list_models()
    print(f"Found {models['count']} models")
    
    # Train a simple model if no models exist
    if models['count'] == 0:
        print("Training a sample model...")
        
        # Create sample data
        from sklearn.datasets import load_iris
        import pandas as pd
        
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        # Train model
        result = client.train_model(
            data=df,
            target_column='target',
            model_type='classification',
            model_name='iris_classifier'
        )
        
        print("Training result:", result)
        
        # Make predictions
        X_test = df.drop('target', axis=1).iloc[:5]
        predictions = client.predict(model='iris_classifier', data=X_test)
        print("Predictions:", predictions)