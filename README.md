# kolosal AutoML 🤖

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-%23B072FF?logo=pypi)](https://github.com/astral-sh/uv)
[![Version](https://img.shields.io/badge/version-v0.1.3-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-partial-yellow)]()

### 🌟 **New Features in v0.1.3**

* **Advanced Batch Processing API** – Hig## 📝 Previous Releases
* **🚀 Unified CLI Interface** – New main.py with interactive mode selection between GUI and API
* **🔧 Enhanced API Integration** – Complete REST API server with health checks for all modules
* **🎯 Improved Error Handling** – Robust error handling and comprehensive logging across all components
* **📊 Better System Integration** – Seamless switching between web interface and API server modes
* **Interactive CLI Mode** – Choose between GUI, API, or system info with simple menu
* **Direct Mode Selection** – Launch specific modes directly via command line flags
* **Version Display** – Easy version checking with --version flag
* **System Analysis** – Built-in hardware and software analysis tools
* **Enhanced Logging** – Comprehensive logging across all components

---

## 📋 Overview

**kolosal AutoML** streamlines the entire machine‑learning lifecycle—from data ingestion to model deployment. Now featuring a modern **Gradio-powered web interface**, intelligent preprocessing, state‑of‑the‑art hyper‑parameter optimisation, device‑aware acceleration, and first‑class experiment tracking.

---

## 🌟 Key Features

### 🖥️ **Modern Web Interface & CLI (NEW in v0.1.2)**
* **Unified CLI Interface** with interactive mode selection
* **Gradio-powered UI** with intuitive tabbed interface
* **Real-time data visualization** and comprehensive data previews
* **Interactive model training** with progress tracking
* **Dedicated inference server** for production deployments
* **Sample dataset integration** with popular ML datasets
* **Secure model management** with encryption support

### 🔄 Flexible Model Training

* Multi‑task support: **classification**, **regression**, **clustering**
* Seamless integration with scikit‑learn, XGBoost, LightGBM & CatBoost
* Automated model selection & tuning

### 🛠️ Supported Algorithms <sup>(partial)</sup>

| Classification               | Regression                  |
| ---------------------------- | --------------------------- |
| Logistic Regression          | Linear Regression           |
| Random Forest Classifier     | Random Forest Regressor     |
| Gradient Boosting Classifier | Gradient Boosting Regressor |
| XGBoost Classifier           | XGBoost Regressor           |
| LightGBM Classifier          | LightGBM Regressor          |
| CatBoost Classifier          | CatBoost Regressor          |
| Support Vector Classifier    | Support Vector Regressor    |
| Neural Network               | Neural Network              |

### 🔍 Advanced Hyper‑parameter Optimisation

* **Grid Search**, **Random Search**, **Bayesian Optimisation**
* **ASHT** (Adaptive Surrogate‑Assisted Hyper‑parameter Tuning)
* **HyperX** (meta‑optimiser for large search spaces)

### 🧠 Smart Pre‑processing

* Auto‑scaling & encoding
* Robust missing‑value & outlier handling
* Feature selection / extraction pipelines
* **Incremental Learning** with partial_fit support

### ⚡ Performance Optimisation

* Device‑aware config & adaptive batching
* **Advanced Batch Processing** with priority queues
* **Dynamic Memory Management** with optimization
* **Asynchronous Processing** for non-blocking operations
* Quantisation & parallel execution
* Memory‑efficient data loaders

### 📊 Monitoring & Reporting

* Real‑time learning curves & metric dashboards
* **Performance Analytics** with detailed insights
* **Job Status Monitoring** for async operations
* Built‑in experiment tracker
* Performance comparison across models
* Feature importance visualizations

---

## 🚀 Installation & Quick Start

### Prerequisites

* **Python 3.10 or newer**

### **Option 1 — Fast Setup with [UV](https://github.com/astral-sh/uv) 🔥 (Recommended)**

```bash
# 1. Clone the repository
git clone https://github.com/Genta-Technology/kolosal_automl.git
cd kolosal_automl

# 2. Install uv (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Create and activate virtual environment with dependencies
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies ultra-fast with uv
uv pip install -r requirements.txt

# Optional: Install GPU-accelerated packages
uv pip install xgboost lightgbm catboost
```

### **Option 2 — Standard `pip`**

```bash
git clone https://github.com/Genta-Technology/kolosal_automl.git
cd kolosal_automl
python -m venv venv && source venv/bin/activate  # create & activate venv
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** For GPU‑accelerated algorithms (XGBoost, LightGBM, CatBoost) install the respective extras:
>
> ```bash
> uv pip install xgboost lightgbm catboost
> # or with pip:
> pip install xgboost lightgbm catboost
> ```

---

## 🎯 Getting Started

### **� Unified CLI Interface (NEW)**

The main entry point for kolosal AutoML system:

```bash
# Interactive mode (recommended for first-time users)
python main.py

# Launch Gradio web interface directly
python main.py --mode gui

# Start API server directly  
python main.py --mode api

# Show version
python main.py --version

# Show system information
python main.py --system-info

# Show help
python main.py --help
```

#### **Available CLI Options:**
```
--mode {gui,api,interactive}    Mode to run (default: interactive)
--version                       Show version and exit
--system-info                   Show system information and exit  
--no-banner                     Skip the banner display
--help                          Show help message and exit
```

#### **CLI Examples:**
```bash
# Interactive mode - choose what to run
python main.py

# Launch web interface in inference-only mode
python main.py --mode gui --inference-only

# Start API server with custom host/port
python main.py --mode api --host 0.0.0.0 --port 8080

# Quick system check
python main.py --system-info --no-banner
```

### **🌐 Option 1: Gradio Web Interface**

Launch the full-featured web interface:

```bash
# Using uv (recommended)
uv run python app.py

# Or with standard Python
python app.py

# Launch in inference-only mode
uv run python app.py --inference-only

# Custom host and port
uv run python app.py --host 0.0.0.0 --port 8080

# Create public shareable link
uv run python app.py --share
```

**Available Web Interface Options:**
- `--inference-only`: Run in inference-only mode (no training capabilities)
- `--model-path`: Path to pre-trained model file (for inference-only mode)
- `--config-path`: Path to model configuration file
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 7860)
- `--share`: Create a public Gradio link

### **🔧 Option 2: API Server**

Start the REST API server:

```bash
# Using uv (recommended)
uv run python start_api.py

# Or using the CLI
python main.py --mode api

# Or directly
uv run python modules/api/app.py
```

- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

#### **🆕 Advanced API Features:**
- **Batch Processing API**: `/api/batch` - High-performance batch operations with adaptive sizing
- **Async Inference**: `/api/inference/predict/async` - Non-blocking predictions with job tracking
- **Performance Metrics**: `/api/inference/metrics` - Real-time performance analytics
- **Health Monitoring**: Complete health checks for all API components

### **💻 Option 3: Python API**

```python
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.engine.batch_processor import BatchProcessor
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy, BatchProcessorConfig
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your data
# X, y = load_your_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure the training engine
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.HYPERX,
    cv_folds=5,
    test_size=0.2,
)

engine = MLTrainingEngine(config)

best_model, metrics = engine.train_model(
    model=RandomForestClassifier(),
    model_name="RandomForest",
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
    X=X_train,
    y=y_train,
)

engine.save_model(best_model)

# 🆕 Advanced Batch Processing
batch_config = BatchProcessorConfig(
    initial_batch_size=32,
    max_batch_size=128,
    enable_priority_queue=True,
    enable_adaptive_batching=True
)

batch_processor = BatchProcessor(batch_config)
batch_processor.start(lambda batch: best_model.predict(batch))

# Async prediction with priority
future = batch_processor.enqueue_predict(X_test[0:1], priority=BatchPriority.HIGH)
predictions = future.result()
```

---

## 🎯 Web Interface Tutorial

### **1. Data Upload & Exploration**
- Upload your CSV, Excel, Parquet, or JSON files
- Or try built-in sample datasets (Iris, Titanic, Boston Housing, etc.)
- View comprehensive data previews with statistics and visualizations
- Explore missing values, data types, and feature distributions

### **2. Configuration**
- Select task type (Classification/Regression)
- Choose optimization strategy (Random Search, Grid Search, Bayesian, HyperX)
- Configure cross-validation settings
- Set preprocessing options (normalization, feature selection)
- Enable advanced features (quantization, early stopping)

### **3. Model Training**
- Select your target column
- Choose from multiple algorithms (Random Forest, XGBoost, Neural Networks, etc.)
- Monitor training progress in real-time
- View training metrics and feature importance

### **4. Predictions & Evaluation**
- Make predictions on new data
- Compare model performance across different algorithms
- Visualize results with confusion matrices and residual plots
- Test with external datasets

### **5. Model Management**
- Save trained models with optional encryption
- Load previously saved models
- Export models in multiple formats (Pickle, Joblib, ONNX)
- Secure model deployment with access controls

### **6. Inference Server**
- Dedicated inference endpoint for production use
- Real-time predictions with minimal latency
- Support for encrypted model files
- RESTful API compatibility

---

## 🧩 Advanced Configuration Example

```python
from modules.configs import MLTrainingEngineConfig, BatchProcessorConfig, InferenceEngineConfig

# Training Configuration
training_config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.BAYESIAN,
    cv_folds=5,
    test_size=0.2,
    random_state=42,
    enable_quantization=True,
    batch_size=64,
    n_jobs=-1,
    feature_selection=True,
    early_stopping=True,
    early_stopping_rounds=10,
)

# 🆕 Batch Processing Configuration
batch_config = BatchProcessorConfig(
    initial_batch_size=16,
    max_batch_size=256,
    batch_timeout=0.01,
    enable_priority_queue=True,
    enable_adaptive_batching=True,
    enable_monitoring=True,
    max_retries=3,
    processing_strategy=BatchProcessingStrategy.ADAPTIVE
)

# 🆕 Enhanced Inference Configuration
inference_config = InferenceEngineConfig(
    enable_batching=True,
    max_batch_size=128,
    batch_timeout=0.02,
    enable_request_deduplication=True,
    max_cache_entries=2000,
    cache_ttl_seconds=7200,
    enable_quantization=True,
    max_concurrent_requests=200,
    enable_throttling=True
)
```

---

## 📊 Sample Datasets Available

The web interface includes several popular datasets for quick experimentation:

- **Iris**: Classic flower classification dataset
- **Titanic**: Passenger survival classification
- **Boston Housing**: House price regression
- **Wine Quality**: Wine rating prediction
- **Diabetes**: Medical classification dataset
- **Car Evaluation**: Multi-class classification

---

## 🔍 Project Structure

```
kolosal_automl/
├── 📄 main.py                      # 🆕 Main CLI entry point
├── 🌐 app.py                       # Gradio web interface
├── 🔧 start_api.py                 # 🆕 API server launcher
├── 🧪 test_api.py                  # 🆕 API testing script
├── 📁 modules/
│   ├── 📄 __init__.py
│   ├── 📄 configs.py               # Configuration management
│   ├── 📁 api/                     # REST API endpoints
│   │   ├── 📄 __init__.py
│   │   ├── 📄 app.py               # Main API application
│   │   ├── 📄 data_preprocessor_api.py
│   │   ├── 📄 device_optimizer_api.py
│   │   ├── 📄 inference_engine_api.py
│   │   ├── 📄 model_manager_api.py
│   │   ├── 📄 quantizer_api.py
│   │   ├── 📄 train_engine_api.py
│   │   ├── 📄 batch_processor_api.py # 🆕 Batch processing API
│   │   └── 📄 README.md            # 🆕 API documentation
│   ├── 📁 engine/                  # Core ML engines
│   │   ├── 📄 __init__.py
│   │   ├── 📄 batch_processor.py   # 🆕 Advanced batch processing
│   │   ├── 📄 data_preprocessor.py
│   │   ├── 📄 inference_engine.py
│   │   ├── 📄 lru_ttl_cache.py
│   │   ├── 📄 quantizer.py
│   │   └── 📄 train_engine.py
│   ├── 📁 optimizer/               # Optimization algorithms
│   │   ├── 📄 __init__.py
│   │   ├── 📄 configs.py
│   │   ├── 📄 device_optimizer.py  # Device optimization
│   │   └── 📄 model_manager.py     # Secure model management
│   ├── 📁 static/                  # Static assets
│   └── 📁 utils/                   # Utility functions
├── 📁 temp_data/                   # Temporary data storage
├── 📁 tests/                       # Test suites
│   ├── 📄 .gitignore
│   ├── 📁 env/                     # Test environments
│   ├── 📁 functional/              # Functional tests
│   ├── 📁 integration/             # Integration tests
│   ├── 📁 templates/               # Test templates
│   │   ├── 📄 .gitattributes
│   │   └── 📄 .gitignore
│   └── 📁 unit/                    # Unit tests
├── 📄 .gitignore
├── 📄 app.py                       # Alternative app launcher
├── 📄 compose.yaml                 # Docker Compose configuration
├── 📄 Dockerfile                   # Docker containerization
├── 📄 CLI_USAGE.md                 # 🆕 CLI usage documentation
├── 📄 kolosal_api.log               # API logging
├── 📄 LICENSE                      # MIT License
├── 📄 pyproject.toml               # 🆕 Project configuration
├── 📄 README.md                    # Project documentation
└── 📄 requirements.txt             # Dependencies
```

---

## 🧪 Test Status

### Functional

| File                                              | Status   |
| ------------------------------------------------- | -------- |
| tests/functional/test/app_api.py                | ❌ FAILED |
| tests/functional/test/quantizer_api.py          | ❌ FAILED |
| tests/functional/test/data_preprocessor_api.py | ❌ FAILED |
| tests/functional/test/device_optimizer_api.py  | ❌ FAILED |
| tests/functional/test/inference_engine_api.py  | ❌ FAILED |
| tests/functional/test/train_engine_api.py      | ❌ FAILED |
| tests/functional/test/model_manager_api.py     | ❌ FAILED |

### Unit

| File                                   | Status   |
| -------------------------------------- | -------- |
| tests/unit/test/batch_processor.py   | ✅ PASSED |
| tests/unit/test/data_preprocessor.py | ❌ FAILED |
| tests/unit/test/device_optimizer.py  | ❌ FAILED |
| tests/unit/test/inference_engine.py  | ❌ FAILED |
| tests/unit/test/lru_ttl_cache.py    | ✅ PASSED |
| tests/unit/test/model_manager.py     | ❌ FAILED |
| tests/unit/test/optimizer_asht.py    | ❌ FAILED |
| tests/unit/test/optimizer_hyperx.py  | ✅ PASSED |
| tests/unit/test/quantizer.py          | ❌ FAILED |
| tests/unit/test/train_engine.py      | ❌ FAILED |

Run all tests:

```bash
pytest -vv
```

---

## 🆕 What's New in **v0.1.3**

### 🎉 **Major Updates**

* **🚀 Advanced Batch Processing System** – High-performance batch processor with adaptive sizing, priority queues, and memory optimization
* **⚡ Asynchronous Job Management** – Non-blocking task execution with comprehensive job tracking and status monitoring
* **🔧 Enhanced Inference Engine** – Dynamic batching, request deduplication, comprehensive caching, and performance analytics
* **📊 Real-time Performance Monitoring** – Detailed metrics collection with insights for optimization
* **🧠 Memory Optimization Framework** – Advanced memory management with garbage collection and usage monitoring
* **� Robust Error Handling** – Enhanced error recovery, retry mechanisms, and detailed error reporting

### 🔧 **Technical Improvements**

* **Batch Processing API** – Complete REST API for batch operations with configurable strategies
* **Async Inference Endpoints** – Non-blocking prediction requests with job tracking
* **Enhanced Health Monitoring** – Comprehensive health checks for all system components
* **Performance Analytics** – Real-time metrics with detailed performance insights
* **Memory Management** – Advanced memory optimization with automatic garbage collection
* **Request Deduplication** – Intelligent caching to avoid redundant computations

### 🌟 **New Features**

* **Priority-based Processing** – Handle high-priority requests with advanced queue management
* **Adaptive Batch Sizing** – Dynamic batch size adjustment based on system load
* **Feature Importance Analysis** – Built-in feature importance calculation for model interpretability
* **Job Status Tracking** – Complete async job lifecycle management with status monitoring
* **Enhanced Documentation** – Comprehensive API documentation with examples and use cases
* **Performance Profiling** – Detailed performance metrics and optimization recommendations

## � Previous Releases

### **v0.1.2 Highlights**
* **🚀 Gradio Web Interface** – Complete redesign from Streamlit to Gradio
* **🔧 Enhanced UV Integration** – Streamlined installation and dependency management
* **🎯 Dedicated Inference Server** – Production-ready inference endpoint
* **📊 Advanced Data Visualization** – Comprehensive data previews and analysis
* **🔐 Secure Model Management** – Enhanced model encryption and access control

---

## 🚧 Roadmap

1. **Complete Test Suite** & CI green ✨
2. **Enhanced Batch Processing** with distributed computing support
3. **Advanced Async Operations** with streaming and WebSocket support
4. **Docker Containerization** for easy deployment
5. **Model Monitoring** & drift detection with real-time alerts
6. **AutoML Pipeline** with automated feature engineering
7. **Time‑series & anomaly‑detection** modules
8. **Cloud‑native deployment** recipes (AWS, GCP, Azure)
9. **MLOps Integration** with popular platforms
10. **Distributed Training** with multi-node support

---

## 💻 Technology Stack

| Purpose              | Library                       |
| -------------------- | ----------------------------- |
| **CLI Interface**    | argparse / subprocess 🆕      |
| **Web UI**           | Gradio                        |
| **Package Mgmt**     | UV                            |
| **API Server**       | FastAPI / Uvicorn 🆕          |
| **Batch Processing** | Custom BatchProcessor 🆕      |
| **Async Jobs**       | asyncio / ThreadPoolExecutor 🆕 |
| **Data Ops**         | Pandas / NumPy                |
| **Core ML**          | scikit‑learn                  |
| **Boosting**         | XGBoost / LightGBM / CatBoost |
| **Visuals**          | Matplotlib / Seaborn          |
| **Serialisation**    | Joblib / Pickle               |
| **Optimization**     | Optuna / Hyperopt             |
| **Memory Mgmt**      | psutil / gc 🆕                |

---

## 🎯 Usage Modes

### 1. **Interactive CLI Mode** 🆕
- Menu-driven interface for mode selection
- Perfect for first-time users
- Built-in help and guidance

### 2. **Web Interface Mode**
- Full-featured Gradio UI
- Visual data exploration and training
- Real-time progress monitoring

### 3. **API Server Mode** 🆕
- Production-ready REST API
- Programmatic access to all features
- Comprehensive health monitoring

### 4. **Direct Python Integration**
- Import modules directly in code
- Maximum flexibility and control
- Advanced customization options
- **Batch Processing Integration** 🆕

---

## 🎯 Advanced Batch Processing (NEW in v0.1.3)

### **High-Performance Batch Operations**

The new Batch Processing system provides enterprise-grade performance for ML workloads:

```python
from modules.engine.batch_processor import BatchProcessor
from modules.configs import BatchProcessorConfig, BatchProcessingStrategy, BatchPriority

# Configure high-performance batch processing
config = BatchProcessorConfig(
    initial_batch_size=32,
    max_batch_size=256,
    enable_priority_queue=True,
    enable_adaptive_batching=True,
    enable_monitoring=True,
    processing_strategy=BatchProcessingStrategy.ADAPTIVE
)

processor = BatchProcessor(config)

# Start processing with your ML model
processor.start(lambda batch: model.predict(batch))

# Submit high-priority requests
future = processor.enqueue_predict(
    data, 
    priority=BatchPriority.HIGH, 
    timeout=30.0
)

result = future.result()  # Get results asynchronously
```

### **Key Features**
- **Adaptive Batch Sizing**: Automatically adjusts batch size based on system load
- **Priority Queues**: Handle urgent requests with configurable priorities
- **Memory Optimization**: Intelligent memory management with garbage collection
- **Performance Monitoring**: Real-time metrics and performance analytics
- **Error Recovery**: Robust retry mechanisms and fault tolerance
- **Async Processing**: Non-blocking operations with future-based results

### **REST API Integration**
```bash
# Configure batch processor
curl -X POST "http://localhost:8000/api/batch/configure" \
  -H "Content-Type: application/json" \
  -d '{"max_batch_size": 128, "enable_priority_queue": true}'

# Submit batch processing job
curl -X POST "http://localhost:8000/api/batch/process-batch" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"data": [1,2,3], "priority": "high"}]}'

# Monitor batch processor status
curl "http://localhost:8000/api/batch/status"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Verify tests pass: `uv run pytest -q`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

---

## 📚 Documentation

For comprehensive documentation and tutorials:
- **CLI Usage Guide**: [CLI_USAGE.md](CLI_USAGE.md) 🆕
- **API Reference**: [modules/api/README.md](modules/api/README.md) 🆕
- **Batch Processing Guide**: [docs/engine/batch_processor_docs.md](docs/engine/batch_processor_docs.md) 🆕
- **Inference Engine Guide**: [docs/engine/inference_engine_docs.md](docs/engine/inference_engine_docs.md) 🆕
- **Device Optimizer Guide**: [docs/device_optimizer_docs.md](docs/device_optimizer_docs.md)
- **Configuration Guide**: [docs/configuration.md](docs/configuration.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 🎉 Getting Started

Ready to explore advanced machine learning? Try our quickstart:

```bash
# Clone and setup
git clone https://github.com/Genta-Technology/kolosal_automl.git
cd kolosal_automl

# Quick install with UV
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Launch with interactive CLI (NEW!)
python main.py

# Or directly launch the web interface
uv run python app.py

# Open http://localhost:7860 in your browser and start experimenting! 🚀
```

### 🚀 Three Ways to Get Started:

1. **🎯 Interactive CLI** (Recommended)
   ```bash
   python main.py
   # Choose from menu: Web Interface, API Server, or System Info
   ```

2. **🌐 Direct Web Interface**
   ```bash
   python main.py --mode gui
   # or: uv run python app.py
   ```

3. **🔧 API Server**
   ```bash
   python main.py --mode api
   # or: uv run python start_api.py
   ```

---

<div align="center">

**Built with ❤️ by the kolosal AI Team**

[🌟 Star us on GitHub](https://github.com/Genta-Technology/kolosal_automl) | [📖 Documentation](docs/) | [🐛 Report Issues](https://github.com/Genta-Technology/kolosal_automl/issues) | [📝 CLI Guide](CLI_USAGE.md)

</div>