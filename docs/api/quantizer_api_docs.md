# Quantizer API Documentation

## Overview
The Quantizer API provides a high-performance RESTful interface for data quantization operations with advanced configuration options. It allows clients to create and manage multiple quantizer instances, perform quantization and dequantization operations, calibrate quantizers with representative data, and retrieve statistics about the quantization process.

## Prerequisites
- Python ≥3.8
- Required packages:
  ```bash
  fastapi>=0.95.0
  uvicorn>=0.15.0
  numpy>=1.20.0
  pydantic>=2.0.0
  ```
- Custom modules:
  - `modules.configs` with `QuantizationConfig`, `QuantizationType`, `QuantizationMode` classes
  - `modules.engine.quantizer` with `Quantizer` class

## Installation
```bash
# Install required packages
pip install fastapi uvicorn numpy pydantic

# Clone or download the repository containing the required modules
git clone <repository-url>

# Navigate to the project directory
cd <project-directory>
```

## Usage
```bash
# Run the API server
python main.py
```

Alternatively, you can run the server using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

The API can be configured through environment variables and runtime parameters. The primary configuration occurs via the `QuantizerConfigModel` which provides numerous options for tailoring the quantization process.

### Default Quantizer Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantization_type` | `INT8` | Type of quantization |
| `quantization_mode` | `DYNAMIC` | Mode of quantization |
| `num_bits` | `8` | Number of bits for quantization |
| `symmetric` | `False` | Whether to use symmetric quantization |
| `enable_cache` | `True` | Whether to enable caching |
| ... | ... | ... |

## Architecture

The Quantizer API follows a RESTful architecture with FastAPI as the web framework. Key components include:

1. **Lifespan Management**: Creates and manages quantizer instances
2. **CORS Middleware**: Enables cross-origin resource sharing
3. **Pydantic Models**: Provides request/response validation
4. **Endpoints**: Expose quantization operations via HTTP methods
5. **Error Handling**: Standardized error responses

The API maintains a dictionary of quantizer instances, with a default instance created at startup.

---

## API Endpoints

### Health Check

```python
@app.get("/")
async def root()
```

- **Description**:  
  Health check endpoint to verify the API is running.

- **Returns**:  
  - JSON response: `{"status": "healthy", "service": "Quantizer API"}`

---

### Quantizer Instance Management

#### Get All Instances

```python
@app.get("/instances", response_model=List[str])
async def get_instances()
```

- **Description**:  
  Get a list of all quantizer instances.

- **Returns**:  
  - List of instance IDs as strings

---

#### Create Instance

```python
@app.post("/instances/{instance_id}")
async def create_instance(instance_id: str, config: QuantizerConfigModel = Body(...))
```

- **Description**:  
  Create a new quantizer instance with the given ID and configuration.

- **Parameters**:  
  - `instance_id (str)`: Unique identifier for the new quantizer instance
  - `config (QuantizerConfigModel)`: Configuration for the new quantizer

- **Returns**:  
  - JSON response: `{"message": "Quantizer instance {instance_id} created successfully"}`
  
- **Raises**:  
  - `409 Conflict`: If an instance with the given ID already exists
  - `500 Internal Server Error`: If there's an error creating the instance

- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/instances/custom_quantizer" \
       -H "Content-Type: application/json" \
       -d '{"quantization_type": "INT8", "num_bits": 8, "symmetric": true}'
  ```

---

#### Delete Instance

```python
@app.delete("/instances/{instance_id}")
async def delete_instance(instance_id: str)
```

- **Description**:  
  Delete a quantizer instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to delete

- **Returns**:  
  - JSON response: `{"message": "Quantizer instance {instance_id} deleted successfully"}`
  
- **Raises**:  
  - `403 Forbidden`: If attempting to delete the default instance
  - `404 Not Found`: If the instance ID doesn't exist

---

#### Get Instance Configuration

```python
@app.get("/instances/{instance_id}/config", response_model=Dict[str, Any])
async def get_config(instance_id: str = "default")
```

- **Description**:  
  Get the configuration of a quantizer instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance, defaults to "default"

- **Returns**:  
  - Dictionary containing the configuration parameters

- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist

---

#### Update Instance Configuration

```python
@app.put("/instances/{instance_id}/config")
async def update_config(instance_id: str, config: QuantizerConfigModel)
```

- **Description**:  
  Update the configuration of a quantizer instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to update
  - `config (QuantizerConfigModel)`: New configuration parameters

- **Returns**:  
  - JSON response: `{"message": "Configuration for quantizer instance {instance_id} updated successfully"}`
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error updating the configuration

---

### Quantization Operations

#### Quantize Data

```python
@app.post("/instances/{instance_id}/quantize", response_model=QuantizationResponse)
async def quantize_data(instance_id: str, request: QuantizationRequest)
```

- **Description**:  
  Quantize input data using the specified quantizer instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to use
  - `request (QuantizationRequest)`: Data to quantize and additional parameters
    - `data (List[List[float]])`: 2D array of data to quantize
    - `validate (bool)`: Whether to validate input
    - `channel_dim (Optional[int])`: Dimension index for per-channel quantization
    - `layer_name (Optional[str])`: Layer name for mixed precision handling

- **Returns**:  
  - `QuantizationResponse` with fields:
    - `data (List[List[Any]])`: Quantized data
    - `quantization_type (str)`: Type of quantization used
    - `original_shape (List[int])`: Shape of the original input data
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error during quantization

- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/instances/default/quantize" \
       -H "Content-Type: application/json" \
       -d '{"data": [[1.2, 3.4, 5.6], [7.8, 9.0, 2.1]], "validate": true}'
  ```

---

#### Dequantize Data

```python
@app.post("/instances/{instance_id}/dequantize", response_model=Dict[str, Any])
async def dequantize_data(instance_id: str, request: DequantizationRequest)
```

- **Description**:  
  Dequantize input data using the specified quantizer instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to use
  - `request (DequantizationRequest)`: Data to dequantize and additional parameters
    - `data (List[List[int]])`: 2D array of quantized data to dequantize
    - `channel_dim (Optional[int])`: Dimension index for per-channel dequantization
    - `layer_name (Optional[str])`: Layer name for mixed precision handling

- **Returns**:  
  - JSON response with fields:
    - `data (List[List[Any]])`: Dequantized data
    - `original_shape (List[int])`: Shape of the original input data
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error during dequantization

---

#### Quantize-Dequantize Data

```python
@app.post("/instances/{instance_id}/quantize_dequantize", response_model=Dict[str, Any])
async def quantize_dequantize_data(instance_id: str, request: QuantizationRequest)
```

- **Description**:  
  Quantize and then dequantize input data to simulate quantization effects without actually changing the data type.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to use
  - `request (QuantizationRequest)`: Data and additional parameters
    - `data (List[List[float]])`: 2D array of data to process
    - `validate (bool)`: Whether to validate input
    - `channel_dim (Optional[int])`: Dimension index for per-channel operations
    - `layer_name (Optional[str])`: Layer name for mixed precision handling

- **Returns**:  
  - JSON response with fields:
    - `data (List[List[Any]])`: Processed data
    - `original_shape (List[int])`: Shape of the original input data
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error during processing

---

### Calibration and Parameters

#### Calibrate Quantizer

```python
@app.post("/instances/{instance_id}/calibrate")
async def calibrate_quantizer(instance_id: str, request: CalibrationRequest)
```

- **Description**:  
  Calibrate the quantizer using representative data.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to calibrate
  - `request (CalibrationRequest)`: Calibration data
    - `data (List[List[List[float]]])`: List of 2D arrays for calibration

- **Returns**:  
  - JSON response with fields:
    - `message (str)`: "Calibration successful"
    - `scale (float)`: Calculated scale factor
    - `zero_point (float)`: Calculated zero point
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error during calibration

---

#### Compute Scale and Zero Point

```python
@app.post("/instances/{instance_id}/compute_params", response_model=ScaleZeroPointResponse)
async def compute_scale_zero_point(instance_id: str, request: QuantizationParamsRequest)
```

- **Description**:  
  Compute scale and zero point for input data without performing quantization.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to use
  - `request (QuantizationParamsRequest)`: Data and parameters
    - `data (List[List[float]])`: 2D array of data to compute parameters for
    - `channel_idx (Optional[int])`: Channel index for per-channel quantization
    - `layer_name (Optional[str])`: Layer name for mixed precision handling

- **Returns**:  
  - `ScaleZeroPointResponse` with fields:
    - `scale (float)`: Calculated scale factor
    - `zero_point (float)`: Calculated zero point
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error computing parameters

---

### Statistics and Cache Management

#### Get Quantization Statistics

```python
@app.get("/instances/{instance_id}/stats", response_model=StatsResponse)
async def get_stats(instance_id: str)
```

- **Description**:  
  Get quantization statistics for a specific instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance

- **Returns**:  
  - `StatsResponse` with fields:
    - `clipped_values (int)`: Number of values clipped during quantization
    - `total_values (int)`: Total number of values processed
    - `quantize_calls (int)`: Number of calls to quantize
    - `dequantize_calls (int)`: Number of calls to dequantize
    - `cache_hits (int)`: Number of cache hits
    - `cache_misses (int)`: Number of cache misses
    - `last_scale (float)`: Last scale factor used
    - `last_zero_point (float)`: Last zero point used
    - `processing_time_ms (float)`: Processing time in milliseconds
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist

---

#### Clear Cache

```python
@app.post("/instances/{instance_id}/clear_cache")
async def clear_cache(instance_id: str)
```

- **Description**:  
  Clear the dequantization cache for a specific instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance

- **Returns**:  
  - JSON response: `{"message": "Cache cleared successfully"}`
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist

---

#### Reset Statistics

```python
@app.post("/instances/{instance_id}/reset_stats")
async def reset_stats(instance_id: str)
```

- **Description**:  
  Reset quantization statistics for a specific instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance

- **Returns**:  
  - JSON response: `{"message": "Statistics reset successfully"}`
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist

---

### Parameter Export/Import

#### Export Parameters

```python
@app.get("/instances/{instance_id}/parameters", response_model=Dict[str, Any])
async def export_parameters(instance_id: str)
```

- **Description**:  
  Export quantization parameters for a specific instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance

- **Returns**:  
  - Dictionary containing the quantization parameters
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist

---

#### Import Parameters

```python
@app.post("/instances/{instance_id}/parameters")
async def import_parameters(instance_id: str, request: ImportExportParams)
```

- **Description**:  
  Import quantization parameters for a specific instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance
  - `request (ImportExportParams)`: Parameters to import
    - `parameters (Dict[str, Any])`: Dictionary of parameters

- **Returns**:  
  - JSON response: `{"message": "Parameters imported successfully"}`
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error importing parameters

---

#### Get Layer Parameters

```python
@app.get("/instances/{instance_id}/layer_params/{layer_name}", response_model=Dict[str, Any])
async def get_layer_params(instance_id: str, layer_name: str)
```

- **Description**:  
  Get quantization parameters for a specific layer.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance
  - `layer_name (str)`: Name of the layer

- **Returns**:  
  - Dictionary containing the layer's quantization parameters
  
- **Raises**:  
  - `404 Not Found`: If the instance ID doesn't exist

---

### File Upload

#### Upload NumPy Array

```python
@app.post("/instances/{instance_id}/upload_numpy", response_model=Dict[str, Any])
async def upload_numpy(
    instance_id: str,
    file: UploadFile = File(...),
    operation: str = Query("quantize", description="Operation to perform: quantize, dequantize, or quantize_dequantize"),
    validate: bool = Query(True, description="Whether to validate input"),
    channel_dim: Optional[int] = Query(None, description="Dimension index for per-channel operation"),
    layer_name: Optional[str] = Query(None, description="Layer name for mixed precision handling")
)
```

- **Description**:  
  Upload a NumPy array (.npy file) and perform quantization operations.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance to use
  - `file (UploadFile)`: .npy file containing a NumPy array
  - `operation (str)`: Operation to perform: "quantize", "dequantize", or "quantize_dequantize"
  - `validate (bool)`: Whether to validate input
  - `channel_dim (Optional[int])`: Dimension index for per-channel operation
  - `layer_name (Optional[str])`: Layer name for mixed precision handling

- **Returns**:  
  - JSON response with metadata about the processed array
  
- **Raises**:  
  - `400 Bad Request`: If the file is not a .npy file or the operation is invalid
  - `404 Not Found`: If the instance ID doesn't exist
  - `500 Internal Server Error`: If there's an error processing the file

---

### Documentation

#### Get API Documentation

```python
@app.get("/documentation")
async def get_documentation()
```

- **Description**:  
  Get API documentation and usage examples.

- **Returns**:  
  - JSON response containing documentation

---

## Data Models

### QuantizerConfigModel

```python
class QuantizerConfigModel(BaseModel)
```

- **Description**:  
  Configuration model for quantizer instances.

- **Attributes**:  
  - `quantization_type (str)`: Quantization type (INT8, UINT8, INT16, FLOAT16, NONE, MIXED)
  - `quantization_mode (str)`: Quantization mode (DYNAMIC, DYNAMIC_PER_BATCH, CALIBRATED, etc.)
  - `num_bits (int)`: Number of bits for quantization
  - `symmetric (bool)`: Whether to use symmetric quantization
  - `enable_cache (bool)`: Whether to enable caching
  - `cache_size (int)`: Size of the cache
  - `enable_mixed_precision (bool)`: Whether to enable mixed precision
  - `per_channel (bool)`: Whether to use per-channel quantization
  - `buffer_size (int)`: Size of preallocated buffers
  - `use_percentile (bool)`: Whether to use percentiles for range calculation
  - `min_percentile (float)`: Minimum percentile for range calculation
  - `max_percentile (float)`: Maximum percentile for range calculation
  - `error_on_nan (bool)`: Whether to error on NaN values
  - `error_on_inf (bool)`: Whether to error on Inf values
  - `optimize_memory (bool)`: Whether to optimize memory usage
  - `enable_requantization (bool)`: Whether to enable requantization
  - `requantization_threshold (float)`: Threshold for requantization
  - `outlier_threshold (Optional[float])`: Threshold for outlier removal
  - `skip_layers (List[str])`: Layers to skip during quantization
  - `quantize_bias (bool)`: Whether to quantize bias layers
  - `quantize_weights_only (bool)`: Whether to quantize weights only
  - `mixed_precision_layers (List[str])`: Layers to use mixed precision for
  - `custom_quantization_config (Dict[str, Dict[str, Any]])`: Custom quantization configuration

---

### QuantizationRequest

```python
class QuantizationRequest(BaseModel)
```

- **Description**:  
  Request model for quantization operations.

- **Attributes**:  
  - `data (List[List[float]])`: 2D array of data to quantize
  - `validate (bool)`: Whether to validate input
  - `channel_dim (Optional[int])`: Dimension index for per-channel quantization
  - `layer_name (Optional[str])`: Layer name for mixed precision handling

---

### DequantizationRequest

```python
class DequantizationRequest(BaseModel)
```

- **Description**:  
  Request model for dequantization operations.

- **Attributes**:  
  - `data (List[List[int]])`: 2D array of quantized data to dequantize
  - `channel_dim (Optional[int])`: Dimension index for per-channel dequantization
  - `layer_name (Optional[str])`: Layer name for mixed precision handling

---

### CalibrationRequest

```python
class CalibrationRequest(BaseModel)
```

- **Description**:  
  Request model for calibration operations.

- **Attributes**:  
  - `data (List[List[List[float]]])`: List of 2D arrays for calibration

---

### QuantizationParamsRequest

```python
class QuantizationParamsRequest(BaseModel)
```

- **Description**:  
  Request model for computing quantization parameters.

- **Attributes**:  
  - `data (List[List[float]])`: 2D array of data to compute parameters for
  - `channel_idx (Optional[int])`: Channel index for per-channel quantization
  - `layer_name (Optional[str])`: Layer name for mixed precision handling

---

### ScaleZeroPointResponse

```python
class ScaleZeroPointResponse(BaseModel)
```

- **Description**:  
  Response model for scale and zero point values.

- **Attributes**:  
  - `scale (float)`: Scale factor
  - `zero_point (float)`: Zero point value

---

### QuantizationResponse

```python
class QuantizationResponse(BaseModel)
```

- **Description**:  
  Response model for quantization operations.

- **Attributes**:  
  - `data (List[List[Any]])`: Quantized data
  - `quantization_type (str)`: Type of quantization used
  - `original_shape (List[int])`: Shape of the original input data

---

### StatsResponse

```python
class StatsResponse(BaseModel)
```

- **Description**:  
  Response model for quantization statistics.

- **Attributes**:  
  - `clipped_values (int)`: Number of values clipped during quantization
  - `total_values (int)`: Total number of values processed
  - `quantize_calls (int)`: Number of calls to quantize
  - `dequantize_calls (int)`: Number of calls to dequantize
  - `cache_hits (int)`: Number of cache hits
  - `cache_misses (int)`: Number of cache misses
  - `last_scale (float)`: Last scale factor used
  - `last_zero_point (float)`: Last zero point used
  - `processing_time_ms (float)`: Processing time in milliseconds

---

### ImportExportParams

```python
class ImportExportParams(BaseModel)
```

- **Description**:  
  Model for importing and exporting quantization parameters.

- **Attributes**:  
  - `parameters (Dict[str, Any])`: Dictionary of parameters

---

## Helper Functions

### get_quantizer

```python
def get_quantizer(instance_id: str = "default") -> Quantizer
```

- **Description**:  
  Helper function to get a quantizer instance.

- **Parameters**:  
  - `instance_id (str)`: ID of the quantizer instance, defaults to "default"

- **Returns**:  
  - `Quantizer` instance

- **Raises**:  
  - `HTTPException`: If the instance ID doesn't exist

---

### numpy_to_list

```python
def numpy_to_list(arr)
```

- **Description**:  
  Helper function to convert numpy arrays to Python lists.

- **Parameters**:  
  - `arr`: Array to convert

- **Returns**:  
  - Python list representation of the array

---

### list_to_numpy

```python
def list_to_numpy(data, dtype=np.float32)
```

- **Description**:  
  Helper function to convert Python lists to numpy arrays.

- **Parameters**:  
  - `data`: List to convert
  - `dtype`: Data type for the numpy array, defaults to np.float32

- **Returns**:  
  - Numpy array

---

## Testing
```bash
# Run API server
python main.py

# Test health check endpoint
curl http://localhost:8000/

# Test creating a custom quantizer instance
curl -X POST "http://localhost:8000/instances/custom_quantizer" \
     -H "Content-Type: application/json" \
     -d '{"quantization_type": "INT8", "num_bits": 8, "symmetric": true}'

# Test quantizing data
curl -X POST "http://localhost:8000/instances/default/quantize" \
     -H "Content-Type: application/json" \
     -d '{"data": [[1.2, 3.4, 5.6], [7.8, 9.0, 2.1]], "validate": true}'
```

## Security & Compliance
- CORS enabled for all origins (modify for production environments)
- Input validation through Pydantic models
- Error handling with appropriate HTTP status codes
- Logging with timestamps and log levels

> Last Updated: 2025-05-11