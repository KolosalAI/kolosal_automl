# Genta AutoML API

The Genta AutoML API provides a comprehensive RESTful interface for machine learning operations including data preprocessing, model training, inference, quantization, and optimization.

## Quick Start

### Running the API

You can run the API in several ways:

1. **Using the startup script (recommended):**
   ```bash
   uv run python start_api.py
   ```

2. **Direct execution:**
   ```bash
   uv run python modules/api/app.py
   ```

3. **Using uvicorn directly:**
   ```bash
   uv run uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Testing the API

Once the server is running, you can test it:

```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open http://localhost:8000/docs in your browser
```

## API Components

The API consists of several specialized modules:

### 1. Main API (`app.py`)
- **Purpose**: Central orchestration and routing
- **Endpoints**: 
  - `GET /` - API information
  - `GET /health` - Health check
  - `GET /metrics` - Performance metrics
  - `GET /docs` - Interactive documentation

### 2. Data Preprocessor API (`data_preprocessor_api.py`)
- **Purpose**: Data preprocessing and transformation
- **Base Path**: `/api/preprocessor`
- **Features**: Data cleaning, normalization, feature engineering

### 3. Model Manager API (`model_manager_api.py`)
- **Purpose**: Model lifecycle management
- **Base Path**: `/api/models`
- **Features**: Model loading, saving, versioning, metadata management

### 4. Inference Engine API (`inference_engine_api.py`)
- **Purpose**: Model inference and prediction
- **Base Path**: `/api/inference`
- **Features**: Batch inference, real-time prediction, performance optimization

### 5. Training Engine API (`train_engine_api.py`)
- **Purpose**: Model training and evaluation
- **Base Path**: `/api/train`
- **Features**: Model training, hyperparameter tuning, experiment tracking

### 6. Quantizer API (`quantizer_api.py`)
- **Purpose**: Model quantization and compression
- **Base Path**: `/api/quantizer`
- **Features**: Model quantization, compression, optimization

### 7. Device Optimizer API (`device_optimizer_api.py`)
- **Purpose**: Hardware optimization and configuration
- **Base Path**: `/api/device`
- **Features**: Device detection, optimization recommendations

### 8. Batch Processor API (`batch_processor_api.py`)
- **Purpose**: Batch processing operations
- **Base Path**: `/api/batch`
- **Features**: Batch processing, queue management, priority handling

## Configuration

The API can be configured using environment variables:

### General Configuration
- `API_ENV`: Environment mode (`development`, `production`) - default: `development`
- `API_DEBUG`: Enable debug mode (`True`, `False`) - default: `False`
- `API_HOST`: Host address - default: `0.0.0.0`
- `API_PORT`: Port number - default: `8000`
- `API_WORKERS`: Number of worker processes - default: `1`

### Security Configuration
- `REQUIRE_API_KEY`: Require API key authentication (`True`, `False`) - default: `False`
- `API_KEYS`: Comma-separated list of valid API keys - default: `dev_key`
- `API_KEY_HEADER`: Header name for API key - default: `X-API-Key`

### Component-Specific Configuration
Each component has its own set of environment variables for fine-tuning behavior. See individual component documentation for details.

## API Features

### Security
- Optional API key authentication
- CORS support
- Input validation
- Rate limiting (configurable)

### Performance
- Gzip compression
- Request/response caching
- Connection pooling
- Async processing
- Batch operations

### Monitoring
- Health checks
- Performance metrics
- Request tracking
- Error logging
- Component status monitoring

### Documentation
- Interactive Swagger UI
- ReDoc documentation
- OpenAPI schema
- Comprehensive examples

## Example Usage

### Health Check
```python
import requests

response = requests.get('http://localhost:8000/health')
print(response.json())
```

### Using with API Key
```python
import requests

headers = {'X-API-Key': 'your_api_key'}
response = requests.get('http://localhost:8000/metrics', headers=headers)
print(response.json())
```

### Model Inference
```python
import requests
import numpy as np

# Prepare data
data = np.random.rand(10, 5).tolist()

# Make inference request
response = requests.post(
    'http://localhost:8000/api/inference/predict',
    json={'data': data}
)
print(response.json())
```

## Development

### Project Structure
```
modules/api/
├── app.py                    # Main API application
├── data_preprocessor_api.py  # Data preprocessing endpoints
├── model_manager_api.py      # Model management endpoints
├── inference_engine_api.py   # Inference endpoints
├── train_engine_api.py       # Training endpoints
├── quantizer_api.py          # Quantization endpoints
├── device_optimizer_api.py   # Device optimization endpoints
├── batch_processor_api.py    # Batch processing endpoints
└── __init__.py              # Package initialization
```

### Adding New Endpoints

To add new endpoints to an existing component:

1. Open the relevant API file (e.g., `inference_engine_api.py`)
2. Add your endpoint function with appropriate decorators
3. Include proper request/response models using Pydantic
4. Add error handling and logging
5. Update the component's documentation

### Adding New Components

To add a new API component:

1. Create a new file in `modules/api/` (e.g., `new_component_api.py`)
2. Follow the existing pattern with FastAPI app creation
3. Add the import and route inclusion in `app.py`
4. Update this README with the new component information

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Port Already in Use**: Change the `API_PORT` environment variable
3. **Permission Errors**: Ensure proper file permissions and directory access
4. **Module Not Found**: Check that all dependencies are installed with `uv install`

### Logs

The API generates logs in the following files:
- `kolosal_api.log` - Main API logs
- `data_preprocessor_api.log` - Data preprocessor logs
- `model_manager_api.log` - Model manager logs
- `inference_api.log` - Inference engine logs
- `ml_api.log` - Training engine logs

### Debug Mode

Enable debug mode for detailed logging:
```bash
export API_DEBUG=True
uv run python modules/api/app.py
```

## Contributing

1. Follow the existing code style and patterns
2. Add appropriate error handling and logging
3. Include proper documentation and type hints
4. Test your changes thoroughly
5. Update this README if adding new features

## License

This project is licensed under the terms specified in the LICENSE file.
