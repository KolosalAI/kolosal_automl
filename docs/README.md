# kolosal AutoML Documentation

## Overview
kolosal AutoML is a comprehensive, production-ready automated machine learning framework designed to streamline the entire ML lifecycle from data preprocessing to model deployment. Built with Python 3.10+, it provides a modern, modular architecture with both web-based and API interfaces.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Module Documentation](#module-documentation)
4. [API Documentation](#api-documentation)
5. [Configuration System](#configuration-system)
6. [Development Guide](#development-guide)
7. [Production Deployment](#production-deployment)

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# Install dependencies using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface
```bash
# Interactive mode - choose between GUI, API, or system info
python main.py

# Direct mode selection
python main.py --gui          # Launch web interface
python main.py --api          # Launch API server
python main.py --info         # Show system information
python main.py --version      # Show version information
```

#### Web Interface
```bash
# Launch the Gradio-powered web interface
python main.py --gui
# Access at: http://localhost:7860
```

#### API Server
```bash
# Launch the FastAPI server
python start_api.py
# Access documentation at: http://localhost:8000/docs
```

#### Python API
```python
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType

# Configure the training engine
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    enable_automl=True,
    cv_folds=5,
    max_iter=1000
)

# Initialize and train
engine = MLTrainingEngine(config)
engine.fit(X_train, y_train)

# Get predictions
predictions = engine.predict(X_test)
```

## Architecture Overview

kolosal AutoML follows a modular, component-based architecture that exactly mirrors the module structure:

```
kolosal-automl/
├── modules/                  # Core implementation
│   ├── api/                  # REST API endpoints
│   │   ├── app.py           # Main FastAPI application
│   │   ├── train_engine_api.py
│   │   ├── inference_engine_api.py
│   │   ├── data_preprocessor_api.py
│   │   ├── model_manager_api.py
│   │   ├── device_optimizer_api.py
│   │   ├── quantizer_api.py
│   │   └── batch_processor_api.py
│   ├── engine/               # Core ML engines
│   │   ├── train_engine.py
│   │   ├── inference_engine.py
│   │   ├── data_preprocessor.py
│   │   ├── batch_processor.py
│   │   ├── quantizer.py
│   │   ├── experiment_tracker.py
│   │   ├── lru_ttl_cache.py
│   │   ├── mixed_precision.py
│   │   ├── jit_compiler.py
│   │   └── [other engine modules]
│   ├── optimizer/            # Hyperparameter optimizers
│   │   ├── asht.py          # ASHT optimizer
│   │   └── hyperoptx.py     # HyperOptX optimizer
│   ├── configs.py           # Configuration system
│   ├── device_optimizer.py  # Hardware optimization
│   └── model_manager.py     # Model management
├── docs/                    # Documentation (mirrors modules/)
│   ├── modules/             # Module-specific documentation
│   │   ├── api/            # API module documentation
│   │   ├── engine/         # Engine module documentation
│   │   ├── optimizer/      # Optimizer module documentation
│   │   ├── configs.md      # Configuration system docs
│   │   ├── device_optimizer.md
│   │   └── model_manager.md
│   ├── README.md           # This file
│   └── INDEX.md            # Documentation index
├── tests/                  # Test suite
├── static/                 # Web interface assets
└── main.py                # CLI entry point
```

## Module Documentation

The documentation structure exactly mirrors the `modules/` directory structure. Each Python file has its own dedicated documentation file:

### Root Level Modules
- [📄 **configs.py**](modules/configs.md) - Type-safe configuration system
- [🔧 **device_optimizer.py**](modules/device_optimizer.md) - Hardware-aware optimization
- [🔐 **model_manager.py**](modules/model_manager.md) - Secure model storage and management

### API Modules (`modules/api/`)
- [🌐 **app.py**](modules/api/app.md) - Main FastAPI application
- [🚂 **train_engine_api.py**](modules/api/train_engine_api.md) - Training API endpoints
- [⚡ **inference_engine_api.py**](modules/api/inference_engine_api.md) - Inference API endpoints
- [🔄 **data_preprocessor_api.py**](modules/api/data_preprocessor_api.md) - Data processing API
- [🔐 **model_manager_api.py**](modules/api/model_manager_api.md) - Model management API
- [🔧 **device_optimizer_api.py**](modules/api/device_optimizer_api.md) - Device optimization API
- [🎯 **quantizer_api.py**](modules/api/quantizer_api.md) - Quantization API
- [📦 **batch_processor_api.py**](modules/api/batch_processor_api.md) - Batch processing API

### Engine Modules (`modules/engine/`)
- [🚂 **train_engine.py**](modules/engine/train_engine.md) - Comprehensive ML training system
- [⚡ **inference_engine.py**](modules/engine/inference_engine.md) - High-performance inference
- [🔄 **data_preprocessor.py**](modules/engine/data_preprocessor.md) - Advanced data preprocessing
- [📦 **batch_processor.py**](modules/engine/batch_processor.md) - Asynchronous batch processing
- [🎯 **quantizer.py**](modules/engine/quantizer.md) - Model quantization system
- [📊 **experiment_tracker.py**](modules/engine/experiment_tracker.md) - Experiment tracking
- [💾 **lru_ttl_cache.py**](modules/engine/lru_ttl_cache.md) - Thread-safe caching
- [⚡ **mixed_precision.py**](modules/engine/mixed_precision.md) - Mixed precision training
- [🔥 **jit_compiler.py**](modules/engine/jit_compiler.md) - JIT compilation
- [🧠 **adaptive_hyperopt.py**](modules/engine/adaptive_hyperopt.md) - Adaptive hyperparameter optimization
- [🌊 **streaming_pipeline.py**](modules/engine/streaming_pipeline.md) - Streaming data pipeline
- [📈 **performance_metrics.py**](modules/engine/performance_metrics.md) - Performance monitoring
- [🔧 **simd_optimizer.py**](modules/engine/simd_optimizer.md) - SIMD optimization
- [💾 **memory_pool.py**](modules/engine/memory_pool.md) - Memory pool management
- [🔄 **dynamic_batcher.py**](modules/engine/dynamic_batcher.md) - Dynamic batching
- [📊 **batch_stats.py**](modules/engine/batch_stats.md) - Batch processing statistics
- [🔍 **prediction_request.py**](modules/engine/prediction_request.md) - Prediction request handling
- [⚠️ **preprocessing_exceptions.py**](modules/engine/preprocessing_exceptions.md) - Preprocessing exceptions
- [🛠️ **utils.py**](modules/engine/utils.md) - Utility functions

### Optimizer Modules (`modules/optimizer/`)
- [🧬 **asht.py**](modules/optimizer/asht.md) - Adaptive Surrogate-Assisted Hyperparameter Tuning
- [🚀 **hyperoptx.py**](modules/optimizer/hyperoptx.md) - Extended hyperparameter optimization

## API Documentation

### REST API Endpoints

The kolosal AutoML API provides comprehensive REST endpoints for all system components:

#### Main API (`modules/api/app.py`)
- **Base URL:** `http://localhost:8000`
- **Documentation:** `http://localhost:8000/docs`
- **Health Check:** `GET /health`

#### Component APIs
1. **Training Engine API** - `/api/train-engine/*`
2. **Inference Engine API** - `/api/inference/*`
3. **Data Preprocessor API** - `/api/preprocessor/*`
4. **Model Manager API** - `/api/model-manager/*`
5. **Device Optimizer API** - `/api/device-optimizer/*`
6. **Quantizer API** - `/api/quantizer/*`
7. **Batch Processor API** - `/api/batch-processor/*`

## Configuration System

kolosal AutoML uses a comprehensive configuration system based on Python dataclasses and enums:

### Configuration Classes
- `QuantizationConfig` - Model quantization settings
- `BatchProcessorConfig` - Batch processing configuration
- `PreprocessorConfig` - Data preprocessing settings
- `InferenceEngineConfig` - Inference engine configuration
- `MLTrainingEngineConfig` - Training engine configuration

### Usage Example
```python
from modules.configs import (
    MLTrainingEngineConfig, TaskType, OptimizationMode,
    QuantizationConfig, QuantizationType
)

# Create quantization config
quant_config = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    symmetric=True,
    calibration_samples=100
)

# Create training config
training_config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    enable_automl=True,
    optimization_mode=OptimizationMode.PERFORMANCE,
    quantization_config=quant_config
)
```

## Development Guide

### Setting Up Development Environment
```bash
# Clone and setup
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
python run_tests.py
```

### Testing
```bash
# Run all tests
pytest -v

# Run specific test categories
pytest -v -m unit          # Unit tests
pytest -v -m functional    # Functional tests
pytest -v -m integration   # Integration tests

# Run with coverage
pytest --cov=modules --cov-report=html
```

### Code Style
- **Formatter:** Black
- **Linter:** Flake8, pylint
- **Type Checking:** mypy
- **Import Sorting:** isort

```bash
# Format code
black modules/
isort modules/

# Lint code
flake8 modules/
pylint modules/
```

## Production Deployment

### Docker Deployment
```bash
# Build image
docker build -t kolosal-automl .

# Run container
docker run -p 8000:8000 kolosal-automl

# Using docker-compose
docker-compose up -d
```

### API Server Configuration
```bash
# Production settings
export API_ENV="production"
export API_DEBUG="False"
export API_WORKERS=4
export REQUIRE_API_KEY="True"
export API_KEYS="your-secure-api-key"

# Start production server
uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `API_ENV` | `development` | Environment (development, staging, production) |
| `API_DEBUG` | `False` | Enable debug mode |
| `API_HOST` | `0.0.0.0` | Host to bind |
| `API_PORT` | `8000` | Port to listen |
| `API_WORKERS` | `1` | Number of worker processes |
| `REQUIRE_API_KEY` | `False` | Require API key authentication |
| `API_KEYS` | `dev_key` | Valid API keys (comma-separated) |

## Version Information

**Current Version:** v0.1.4

### Recent Updates (v0.1.4)
- Complete pytest test suite migration
- Advanced test runner with category-based execution
- Comprehensive test fixtures and markers
- Individual test execution capabilities
- Enhanced error handling with pytest.skip
- CI/CD ready test configuration
- Production hardening improvements
- **NEW**: Modular documentation structure matching code organization

### Previous Releases
- **v0.1.3:** Advanced batch processing, unified CLI, enhanced API integration
- **v0.1.2:** Gradio web interface, real-time visualization, secure model management
- **v0.1.1:** Core AutoML functionality, hyperparameter optimization
- **v0.1.0:** Initial release with basic ML pipeline

## Support and Contributing

### Getting Help
- **Documentation:** Module-specific docs in `docs/modules/` directory
- **Issues:** GitHub Issues for bug reports and feature requests
- **API Documentation:** Interactive docs at `/docs` endpoint

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update corresponding documentation in `docs/modules/`
6. Run the test suite
7. Submit a pull request

### Documentation Guidelines
- Each Python file should have a corresponding `.md` file in `docs/modules/`
- Follow the established documentation structure and format
- Include comprehensive usage examples
- Document all public methods and classes
- Keep examples up-to-date with code changes

### License
MIT License - see [LICENSE](../LICENSE) file for details.

---

*For detailed module documentation, refer to the corresponding files in the `docs/modules/` directory that mirror the `modules/` code structure.*

## Core Components

### 1. Configuration System (`modules/configs.py`)
Type-safe configuration classes for all system components.

**Key Features:**
- Dataclass-based configurations
- Enum-based type safety
- Serialization/deserialization support
- Configuration validation

**Documentation:** [Configuration System Docs](configs_docs.md)

### 2. Device Optimizer (`modules/device_optimizer.py`)
Hardware-aware optimization for CPU architectures.

**Key Features:**
- Automatic hardware detection
- Performance optimization profiles
- Resource-aware configuration generation
- Multi-architecture support

**Documentation:** [Device Optimizer Docs](device_optimizer_docs.md)

### 3. Training Engine (`modules/engine/train_engine.py`)
Comprehensive ML training system with automation capabilities.

**Key Features:**
- Multi-algorithm support (scikit-learn, XGBoost, LightGBM, CatBoost)
- Automated hyperparameter optimization
- Advanced performance optimizations (JIT, mixed precision)
- Experiment tracking and visualization
- Incremental learning support

**Documentation:** [Training Engine Docs](engine/train_engine_docs.md)

### 4. Inference Engine (`modules/engine/inference_engine.py`)
High-performance inference system for production deployments.

**Key Features:**
- Dynamic batching
- Model compilation and optimization
- Memory pooling
- Request deduplication
- Performance monitoring

**Documentation:** [Inference Engine Docs](engine/inference_engine_docs.md)

### 5. Data Preprocessor (`modules/engine/data_preprocessor.py`)
Advanced data preprocessing pipeline.

**Key Features:**
- Multiple normalization strategies
- Outlier detection and handling
- Missing value imputation
- Feature selection and extraction
- Thread-safe operations

**Documentation:** [Data Preprocessor Docs](engine/data_preprocessor_docs.md)

### 6. Model Manager (`modules/model_manager.py`)
Secure model storage and management system.

**Key Features:**
- Encryption-based security
- Model versioning
- Integrity verification
- Quantization support
- Automated model selection

**Documentation:** [Model Manager Docs](model_manager_docs.md)

### 7. Hyperparameter Optimizers (`modules/optimizer/`)
Advanced optimization algorithms for hyperparameter tuning.

**Available Optimizers:**
- ASHT (Adaptive Surrogate-Assisted Hyperparameter Tuning)
- HyperOptX (Extended Hyperparameter Optimization)

**Documentation:** 
- [ASHT Optimizer Docs](optimizers/optimizer_asht_docs.md)
- [HyperOptX Optimizer Docs](optimizers/optimizer_hyperoptx_docs.md)

## API Documentation

### REST API Endpoints

The kolosal AutoML API provides comprehensive REST endpoints for all system components:

#### Main API (`modules/api/app.py`)
- **Base URL:** `http://localhost:8000`
- **Documentation:** `http://localhost:8000/docs`
- **Health Check:** `GET /health`

#### Component APIs
1. **Training Engine API** - `/api/train-engine/*`
2. **Inference Engine API** - `/api/inference/*`
3. **Data Preprocessor API** - `/api/preprocessor/*`
4. **Model Manager API** - `/api/model-manager/*`
5. **Device Optimizer API** - `/api/device-optimizer/*`
6. **Quantizer API** - `/api/quantizer/*`

**Detailed Documentation:**
- [Main API Docs](api/app_docs.md)
- [Training Engine API Docs](api/train_engine_api_docs.md)
- [Inference Engine API Docs](api/inference_engine_api_docs.md)
- [Data Preprocessor API Docs](api/data_preprcessor_api_docs.md)
- [Model Manager API Docs](api/model_manager_api_docs.md)
- [Device Optimizer API Docs](api/device_optimizer_api_docs.md)
- [Quantizer API Docs](api/quantizer_api_docs.md)

## Configuration System

kolosal AutoML uses a comprehensive configuration system based on Python dataclasses and enums:

### Configuration Classes
- `QuantizationConfig` - Model quantization settings
- `BatchProcessorConfig` - Batch processing configuration
- `PreprocessorConfig` - Data preprocessing settings
- `InferenceEngineConfig` - Inference engine configuration
- `MLTrainingEngineConfig` - Training engine configuration

### Usage Example
```python
from modules.configs import (
    MLTrainingEngineConfig, TaskType, OptimizationMode,
    QuantizationConfig, QuantizationType
)

# Create quantization config
quant_config = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    symmetric=True,
    calibration_samples=100
)

# Create training config
training_config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    enable_automl=True,
    optimization_mode=OptimizationMode.PERFORMANCE,
    quantization_config=quant_config
)
```

## Development Guide

### Setting Up Development Environment
```bash
# Clone and setup
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
python run_tests.py
```

### Testing
```bash
# Run all tests
pytest -v

# Run specific test categories
pytest -v -m unit          # Unit tests
pytest -v -m functional    # Functional tests
pytest -v -m integration   # Integration tests

# Run with coverage
pytest --cov=modules --cov-report=html
```

### Code Style
- **Formatter:** Black
- **Linter:** Flake8, pylint
- **Type Checking:** mypy
- **Import Sorting:** isort

```bash
# Format code
black modules/
isort modules/

# Lint code
flake8 modules/
pylint modules/
```

## Production Deployment

### Docker Deployment
```bash
# Build image
docker build -t kolosal-automl .

# Run container
docker run -p 8000:8000 kolosal-automl

# Using docker-compose
docker-compose up -d
```

### API Server Configuration
```bash
# Production settings
export API_ENV="production"
export API_DEBUG="False"
export API_WORKERS=4
export REQUIRE_API_KEY="True"
export API_KEYS="your-secure-api-key"

# Start production server
uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `API_ENV` | `development` | Environment (development, staging, production) |
| `API_DEBUG` | `False` | Enable debug mode |
| `API_HOST` | `0.0.0.0` | Host to bind |
| `API_PORT` | `8000` | Port to listen |
| `API_WORKERS` | `1` | Number of worker processes |
| `REQUIRE_API_KEY` | `False` | Require API key authentication |
| `API_KEYS` | `dev_key` | Valid API keys (comma-separated) |

## Version Information

**Current Version:** v0.1.4

### Recent Updates (v0.1.4)
- Complete pytest test suite migration
- Advanced test runner with category-based execution
- Comprehensive test fixtures and markers
- Individual test execution capabilities
- Enhanced error handling with pytest.skip
- CI/CD ready test configuration
- Production hardening improvements

### Previous Releases
- **v0.1.3:** Advanced batch processing, unified CLI, enhanced API integration
- **v0.1.2:** Gradio web interface, real-time visualization, secure model management
- **v0.1.1:** Core AutoML functionality, hyperparameter optimization
- **v0.1.0:** Initial release with basic ML pipeline

## Support and Contributing

### Getting Help
- **Documentation:** Available in the `docs/` directory
- **Issues:** GitHub Issues for bug reports and feature requests
- **API Documentation:** Interactive docs at `/docs` endpoint

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### License
MIT License - see [LICENSE](../LICENSE) file for details.

---

*For detailed component documentation, refer to the individual documentation files in the respective subdirectories.*
