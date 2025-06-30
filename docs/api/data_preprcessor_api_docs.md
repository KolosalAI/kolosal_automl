# Data Preprocessor API Documentation

## Overview
The Data Preprocessor API provides a RESTful interface for data preprocessing operations including normalization, outlier detection, and missing value handling. The API allows creating configurable preprocessor instances, fitting them to data, transforming data, and managing preprocessing models through a set of HTTP endpoints.

## Prerequisites
- Python ≥3.10
- FastAPI
- NumPy
- Pandas
- Required dependencies as specified in import statements

## Installation
```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export MODEL_DIR=./models
export TEMP_DATA_DIR=./temp_data
```

## Usage
```bash
# Start the server
python data_preprocessor_api.py

# Or with uvicorn directly
uvicorn data_preprocessor_api:app --host 0.0.0.0 --port 8000
```

## Configuration
| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `MODEL_DIR` | `./models` | Directory for storing serialized preprocessor models |
| `TEMP_DATA_DIR` | `./temp_data` | Directory for temporary data storage |

## Architecture
The API is built on FastAPI and provides RESTful endpoints for interacting with the `DataPreprocessor` class. Data can be provided as JSON or CSV, and responses can be returned in various formats.

### Core Components:
- **FastAPI Application**: Handles HTTP requests and responses
- **DataPreprocessor Class**: Performs actual data preprocessing operations
- **In-memory Preprocessor Storage**: Maintains active preprocessor instances

## API Endpoints

### Preprocessor Management

#### Create a Preprocessor
```
POST /preprocessors
```
Creates a new preprocessor instance with the specified configuration.

**Request Body**:
```json
{
  "normalization": "STANDARD",
  "handle_nan": true,
  "handle_inf": true,
  "detect_outliers": false,
  "nan_strategy": "MEAN",
  "inf_strategy": "MAX_VALUE",
  "outlier_method": "ZSCORE",
  "outlier_handling": "CLIP",
  "robust_percentiles": [25.0, 75.0],
  "outlier_iqr_multiplier": 1.5,
  "outlier_zscore_threshold": 3.0,
  "outlier_percentiles": [1.0, 99.0],
  "epsilon": 1e-8,
  "clip_values": false,
  "clip_min": -Infinity,
  "clip_max": Infinity,
  "nan_fill_value": 0.0,
  "copy_X": true,
  "dtype": "float32",
  "debug_mode": false,
  "parallel_processing": false,
  "n_jobs": -1,
  "chunk_size": 10000,
  "cache_enabled": true,
  "enable_input_validation": true,
  "input_size_limit": null,
  "version": "1.0.0"
}
```

**Response (201 Created)**:
```json
{
  "preprocessor_id": "uuid-string",
  "config": { /* configuration settings */ },
  "created_at": "2025-05-11T10:00:00.000Z"
}
```

#### List All Preprocessors
```
GET /preprocessors
```
Lists all currently active preprocessor instances.

**Response**:
```json
{
  "preprocessors": [
    {
      "preprocessor_id": "uuid-string",
      "fitted": true,
      "n_features": 10,
      "n_samples_seen": 1000,
      "feature_names": ["feature1", "feature2"],
      "config": { /* configuration settings */ },
      "created_at": "2025-05-11T10:00:00.000Z"
    }
  ]
}
```

#### Get Preprocessor Information
```
GET /preprocessors/{preprocessor_id}
```
Gets detailed information about a specific preprocessor instance.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor

**Response**:
```json
{
  "preprocessor_id": "uuid-string",
  "fitted": true,
  "n_features": 10,
  "n_samples_seen": 1000,
  "feature_names": ["feature1", "feature2"],
  "config": { /* configuration settings */ },
  "created_at": "2025-05-11T10:00:00.000Z"
}
```

#### Delete a Preprocessor
```
DELETE /preprocessors/{preprocessor_id}
```
Deletes a preprocessor instance from memory.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor to delete

**Response**:
```json
{
  "success": true,
  "message": "Preprocessor {preprocessor_id} deleted successfully"
}
```

### Data Operations

#### Fit a Preprocessor
```
POST /preprocessors/{preprocessor_id}/fit
```
Fits a preprocessor to the provided data.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor
- `has_header` (query): Whether the CSV file has a header row (default: true)

**Request Body** (JSON data format):
```json
{
  "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  "feature_names": ["feature1", "feature2", "feature3"]
}
```

**Alternative**: Upload a CSV file using multipart/form-data

**Response**:
```json
{
  "success": true,
  "message": "Preprocessor {preprocessor_id} fitted successfully to {n_samples} samples with {n_features} features"
}
```

#### Transform Data
```
POST /preprocessors/{preprocessor_id}/transform
```
Transforms data using a fitted preprocessor.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor
- `has_header` (query): Whether the CSV file has a header row (default: true)
- `output_format` (query): Output format (json or csv, default: json)

**Request Body** (Transform options):
```json
{
  "copy": true
}
```

**Data Input**: Either JSON data or a CSV file

**Response** (JSON format):
```json
{
  "transformed_data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  "shape": [2, 3]
}
```

**Response** (CSV format): A downloaded CSV file with the transformed data

#### Fit and Transform in One Operation
```
POST /preprocessors/{preprocessor_id}/fit-transform
```
Fits the preprocessor to data and transforms it in one operation.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor
- `has_header` (query): Whether the CSV file has a header row (default: true)
- `output_format` (query): Output format (json or csv, default: json)

**Request Body** (Transform options):
```json
{
  "copy": true
}
```

**Data Input**: Either JSON data or a CSV file

**Response** (JSON format):
```json
{
  "transformed_data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  "shape": [2, 3]
}
```

**Response** (CSV format): A downloaded CSV file with the transformed data

#### Reverse Transform Data
```
POST /preprocessors/{preprocessor_id}/reverse-transform
```
Reverse transforms data using a fitted preprocessor, converting normalized data back to its original scale.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor
- `has_header` (query): Whether the CSV file has a header row (default: true)
- `output_format` (query): Output format (json or csv, default: json)

**Request Body** (Transform options):
```json
{
  "copy": true
}
```

**Data Input**: Either JSON data or a CSV file

**Response** (JSON format):
```json
{
  "reverse_transformed_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  "shape": [2, 3]
}
```

**Response** (CSV format): A downloaded CSV file with the reverse transformed data

#### Partially Fit a Preprocessor
```
POST /preprocessors/{preprocessor_id}/partial-fit
```
Updates a preprocessor with new data incrementally.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor
- `has_header` (query): Whether the CSV file has a header row (default: true)

**Data Input**: Either JSON data or a CSV file

**Response**:
```json
{
  "success": true,
  "message": "Preprocessor {preprocessor_id} partially fitted with {n_samples} additional samples"
}
```

### Preprocessor Configuration and State

#### Reset a Preprocessor
```
POST /preprocessors/{preprocessor_id}/reset
```
Resets a preprocessor to its initial state, clearing all fitted statistics.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor

**Response**:
```json
{
  "success": true,
  "message": "Preprocessor {preprocessor_id} reset successfully"
}
```

#### Update Preprocessor Configuration
```
POST /preprocessors/{preprocessor_id}/update-config
```
Updates the configuration of an existing preprocessor.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor

**Request Body**: Same as the configuration for creating a preprocessor

**Response**:
```json
{
  "success": true,
  "message": "Preprocessor {preprocessor_id} configuration updated successfully"
}
```

#### Get Preprocessor Statistics
```
GET /preprocessors/{preprocessor_id}/statistics
```
Gets the statistics computed by a fitted preprocessor.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor

**Response**:
```json
{
  "preprocessor_id": "uuid-string",
  "statistics": {
    "mean": [0.1, 0.2, 0.3],
    "std": [1.0, 1.0, 1.0],
    "min": [-1.0, -1.0, -1.0],
    "max": [1.0, 1.0, 1.0]
  }
}
```

#### Get Preprocessor Performance Metrics
```
GET /preprocessors/{preprocessor_id}/metrics
```
Gets performance metrics collected during preprocessing operations.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor

**Response**:
```json
{
  "preprocessor_id": "uuid-string",
  "metrics": {
    "fit_time": [0.1, 0.2],
    "transform_time": [0.05, 0.06],
    "memory_usage": [1024, 2048]
  }
}
```

### Model Persistence

#### Save Preprocessor to Disk
```
POST /preprocessors/{preprocessor_id}/serialize
```
Serializes (saves) a preprocessor to disk.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor
- `filename` (query, optional): Custom filename to use (without extension)

**Response**:
```json
{
  "success": true,
  "message": "Preprocessor serialized successfully to {file_path}"
}
```

#### Load Preprocessor from Disk
```
POST /preprocessors/deserialize
```
Deserializes (loads) a preprocessor from a file.

**Parameters**:
- `custom_id` (query, optional): Custom ID to assign to the loaded preprocessor

**Request Body**: Multipart form with a serialized preprocessor file

**Response**:
```json
{
  "preprocessor_id": "uuid-string",
  "config": { /* configuration settings */ },
  "created_at": "2025-05-11T10:00:00.000Z"
}
```

#### Download Serialized Preprocessor
```
GET /preprocessors/{preprocessor_id}/download
```
Serializes and downloads a preprocessor.

**Parameters**:
- `preprocessor_id`: Unique ID of the preprocessor

**Response**: A serialized preprocessor file as a download

### Utilities

#### Health Check
```
GET /health
```
Health check endpoint to verify the API is running.

**Response**:
```json
{
  "success": true,
  "message": "Data Preprocessor API is operational"
}
```

## Request/Response Models

### Enums
- **NormalizationTypeEnum**: NONE, STANDARD, MINMAX, ROBUST, LOG, QUANTILE, POWER, CUSTOM
- **OutlierMethodEnum**: IQR, ZSCORE, PERCENTILE
- **OutlierHandlingEnum**: CLIP, REMOVE, WINSORIZE, MEAN
- **NanStrategyEnum**: MEAN, MEDIAN, MOST_FREQUENT, CONSTANT, ZERO
- **InfStrategyEnum**: MAX_VALUE, MEDIAN, CONSTANT, NAN

### Request Models
- **PreprocessorConfigRequest**: Configuration for creating a new preprocessor instance
- **DataRequest**: Request model for inline data submission
- **TransformOptions**: Options for transform operations

### Response Models
- **PreprocessorCreateResponse**: Response model for preprocessor creation
- **StatusResponse**: Generic status response
- **PreprocessorInfoResponse**: Information about a preprocessor instance
- **StatisticsResponse**: Response model for preprocessor statistics
- **MetricsResponse**: Response model for preprocessor performance metrics
- **AllPreprocessorsResponse**: Response listing all available preprocessors

## Error Handling
The API uses standard HTTP status codes for error responses:
- **400 Bad Request**: For input validation errors or preprocessing errors
- **404 Not Found**: When a preprocessor with the specified ID does not exist
- **500 Internal Server Error**: For unexpected errors during processing

Error responses include:
- **Error Type**: The class name of the exception
- **Error Message**: A description of what went wrong
- **Error Details**: Additional information when available

## Security & Compliance
- No authentication mechanism is implemented in this version
- Data is processed in-memory and can be persisted to disk
- No encryption is applied to stored models

> Last Updated: 2025-05-11