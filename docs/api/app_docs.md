# kolosal AutoML API

## Overview
The kolosal AutoML API is a comprehensive FastAPI-based application that provides a suite of machine learning tools for data preprocessing, model training, inference, quantization, and deployment. It features a modular architecture with separate API components for different ML workflow stages, robust error handling, request tracking, performance metrics collection, and comprehensive security features.

The API integrates all components of the kolosal AutoML system into a unified REST interface with automatic OpenAPI documentation, middleware support, and production-ready configurations.

## Prerequisites
- Python ≥3.10
- FastAPI
- Uvicorn
- Pydantic
- Required packages:
  ```bash
  pip install fastapi uvicorn pydantic
  ```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server
```bash
# Using the Python module
python -m modules.api.app

# Or using the main entry point
python start_api.py

# Or using Uvicorn directly
uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --reload

# Production deployment
uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation
Once the server is running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Health Check
```bash
curl http://localhost:8000/health
```

## Configuration
The API can be configured through environment variables:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `API_ENV` | `"development"` | Deployment environment (development, staging, production) |
| `API_DEBUG` | `"False"` | Enable debug mode (True, False) |
| `API_HOST` | `"0.0.0.0"` | Host IP address to bind |
| `API_PORT` | `8000` | Port to listen on |
| `API_WORKERS` | `1` | Number of worker processes |
| `API_KEY_HEADER` | `"X-API-Key"` | HTTP header name for API key |
| `API_KEYS` | `"dev_key"` | Comma-separated list of valid API keys |
| `REQUIRE_API_KEY` | `"False"` | Require API key for authentication |

## Architecture

### Component Structure
The API is organized into several modules:

1. **Data Preprocessor** (`/api/preprocessor/*`): Data preprocessing and transformation
2. **Device Optimizer** (`/api/device/*`): Hardware optimization for ML models
3. **Inference Engine** (`/api/inference/*`): Model inference and prediction
4. **Model Manager** (`/api/models/*`): Model versioning and lifecycle management
5. **Quantizer** (`/api/quantizer/*`): Model quantization and compression
6. **Training Engine** (`/api/train/*`): Model training and hyperparameter tuning

### Key Endpoints

- **Root** (`/`): API information and documentation links
- **Health Check** (`/health`): System health status
- **Metrics** (`/metrics`): Performance metrics (requires authentication)
- **API Documentation**:
  - Swagger UI: `/docs`
  - ReDoc: `/redoc`
  - OpenAPI Schema: `/openapi.json`

### Middleware and Features

- **Request ID Tracking**: Each request is assigned a unique ID
- **CORS Support**: Cross-Origin Resource Sharing enabled
- **GZip Compression**: Automatic response compression
- **Metrics Collection**: Request counts, response times, and error rates
- **API Key Authentication**: Optional security for endpoints
- **Global Exception Handling**: Standardized error responses

---

## Functions

### `lifespan(app: FastAPI)`
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
```
- **Description**:  
  Manages application startup and shutdown events, initializing app state and metrics.

- **Parameters**:  
  - `app (FastAPI)`: The FastAPI application instance.

- **Returns**:  
  - Async context manager that yields control during the application's lifespan.

### `verify_api_key(api_key: str = Header(None, alias=API_KEY_HEADER))`
```python
async def verify_api_key(api_key: str = Header(None, alias=API_KEY_HEADER)):
```
- **Description**:  
  Dependency function that verifies the provided API key is valid if authentication is enabled.

- **Parameters**:  
  - `api_key (str)`: The API key from the request header.

- **Returns**:  
  - `bool`: True if the API key is valid or not required.

- **Raises**:  
  - `HTTPException(401)`: If API key is required but missing.
  - `HTTPException(403)`: If API key is invalid.

### `add_request_id(request: Request, call_next: Callable) -> Response`
```python
@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable) -> Response:
```
- **Description**:  
  Middleware that adds a unique request ID to each request, tracks metrics, and monitors request processing.

- **Parameters**:  
  - `request (Request)`: The incoming HTTP request.
  - `call_next (Callable)`: The next middleware function in the chain.

- **Returns**:  
  - `Response`: The HTTP response with added request ID header.

### `include_component_routes(app: FastAPI, router: APIRouter, prefix: str, tags: List[str])`
```python
def include_component_routes(app: FastAPI, router: APIRouter, prefix: str, tags: List[str]):
```
- **Description**:  
  Includes routes from a FastAPI app into a router with specific prefix and tags.

- **Parameters**:  
  - `app (FastAPI)`: The FastAPI app containing the routes.
  - `router (APIRouter)`: The router to add the routes to.
  - `prefix (str)`: URL prefix for the component.
  - `tags (List[str])`: OpenAPI tags for the component.

- **Returns**:  
  - `None`

### `root()`
```python
@health_router.get("/", response_model=dict)
async def root():
```
- **Description**:  
  Root endpoint that provides basic API information and links to documentation.

- **Returns**:  
  - `dict`: API name, version, and links to documentation and health check.

### `health_check()`
```python
@health_router.get("/health", response_model=HealthResponse)
async def health_check():
```
- **Description**:  
  Health check endpoint to verify that the API and all its components are operational.

- **Returns**:  
  - `HealthResponse`: The health status of the API and its components.

### `get_metrics()`
```python
@health_router.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(verify_api_key)])
async def get_metrics():
```
- **Description**:  
  Endpoint that returns API performance metrics (requires API key authentication).

- **Returns**:  
  - `MetricsResponse`: Performance metrics including request counts, errors, and response times.

### `get_docs(authorized: bool = Depends(verify_api_key))`
```python
@app.get("/docs", include_in_schema=False)
async def get_docs(authorized: bool = Depends(verify_api_key)):
```
- **Description**:  
  Custom Swagger UI endpoint that requires API key authentication if enabled.

- **Parameters**:  
  - `authorized (bool)`: True if the API key is valid or not required.

- **Returns**:  
  - Swagger UI HTML response.

### `get_redoc(authorized: bool = Depends(verify_api_key))`
```python
@app.get("/redoc", include_in_schema=False)
async def get_redoc(authorized: bool = Depends(verify_api_key)):
```
- **Description**:  
  Custom ReDoc endpoint that requires API key authentication if enabled.

- **Parameters**:  
  - `authorized (bool)`: True if the API key is valid or not required.

- **Returns**:  
  - ReDoc HTML response.

### `global_exception_handler(request: Request, exc: Exception)`
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
```
- **Description**:  
  Global exception handler that processes all unhandled exceptions and returns standardized error responses.

- **Parameters**:  
  - `request (Request)`: The request that caused the exception.
  - `exc (Exception)`: The exception that was raised.

- **Returns**:  
  - `JSONResponse`: A standardized error response with details about the exception.

---

## Classes

### `HealthResponse`
```python
class HealthResponse(BaseModel):
```
- **Description**:  
  Pydantic model for the health check response.

- **Attributes**:  
  - `status (str)`: API status (e.g., "healthy").
  - `version (str)`: API version.
  - `environment (str)`: API environment (development, staging, production).
  - `uptime_seconds (float)`: API uptime in seconds.
  - `components (Dict[str, str])`: Status of individual components.
  - `timestamp (str)`: Response timestamp (ISO format).

### `MetricsResponse`
```python
class MetricsResponse(BaseModel):
```
- **Description**:  
  Pydantic model for the API metrics response.

- **Attributes**:  
  - `total_requests (int)`: Total number of requests processed.
  - `errors (int)`: Total number of errors encountered.
  - `uptime_seconds (float)`: API uptime in seconds.
  - `requests_per_endpoint (Dict[str, int])`: Request count per endpoint.
  - `average_response_time (Dict[str, float])`: Average response time per endpoint.
  - `active_connections (int)`: Number of active connections.
  - `timestamp (str)`: Response timestamp (ISO format).

---

## Testing
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/ -v

# Load testing (requires locust)
locust -f tests/performance/locustfile.py
```

## Security & Compliance
- API key authentication for sensitive endpoints
- Request ID tracking for audit trails
- Standardized error responses
- CORS protection
- Metrics collection for monitoring and alerting

## API Modules
kolosal AutoML API includes the following modules, each with their own set of endpoints:

1. **Data Preprocessor**: `/api/preprocessor/`
   - Data validation, cleaning, and transformation

2. **Device Optimizer**: `/api/device/`
   - Hardware-specific model optimization

3. **Inference Engine**: `/api/inference/`
   - Model inference and batch prediction

4. **Model Manager**: `/api/models/`
   - Model versioning, storage, and retrieval

5. **Quantizer**: `/api/quantizer/`
   - Model compression and quantization

6. **Training Engine**: `/api/train/`
   - Model training and hyperparameter tuning

> Last Updated: 2025-05-11