# Advanced ML Training Engine 🤖

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-%23B072FF?logo=pypi)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-partial-yellow.svg)]()

---

## 📋 Overview

The **Advanced ML Training Engine** streamlines the entire machine‑learning lifecycle—from data ingestion to model deployment. Now featuring a modern **Gradio-powered web interface**, intelligent preprocessing, state‑of‑the‑art hyper‑parameter optimisation, device‑aware acceleration, and first‑class experiment tracking.

---

## 🌟 Key Features

### 🖥️ **Modern Web Interface (NEW in v0.1.2)**
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

### ⚡ Performance Optimisation

* Device‑aware config & adaptive batching
* Quantisation & parallel execution
* Memory‑efficient data loaders

### 📊 Monitoring & Reporting

* Real‑time learning curves & metric dashboards
* Built‑in experiment tracker
* Performance comparison across models
* Feature importance visualizations

---

## 🚀 Installation

### Prerequisites

* **Python 3.10 or newer**

### **Option 1 — Fast Setup with [UV](https://github.com/astral-sh/uv) 🔥 (Recommended)**

```bash
# 1. Clone the repository
git clone https://github.com/Genta-Technology/kolosal_automl.git
cd kolosal_automl

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on Windows: 
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Create and activate virtual environment with dependencies
uv venv
# Activate virtual environment
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 4. Install dependencies ultra-fast with uv
uv pip install -r requirements.txt

# Optional: Install GPU-accelerated packages
uv pip install xgboost lightgbm catboost
```

### Option 2 — Standard `pip`

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

## 💻 Quick Start

### **Option 1: Modern Gradio Web Interface (Recommended)**

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

**Available Command Line Options:**
- `--inference-only`: Run in inference-only mode (no training capabilities)
- `--model-path`: Path to pre-trained model file (for inference-only mode)
- `--config-path`: Path to model configuration file
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 7860)
- `--share`: Create a public Gradio link

### **Option 2: Python API**

```python
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your data
# X, y = load_your_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure the engine
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
predictions = engine.predict(X_test)
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
config = MLTrainingEngineConfig(
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
├── 📄 main.py                      # Main application entry point
├── 🌐 app.py                       # 🆕 Gradio web interface
├── 📁 modules/
│   ├── 📄 __init__.py
│   ├── 📄 configs.py               # Configuration management
│   ├── 📁 api/                     # 🆕 API endpoints
│   │   ├── 📄 __init__.py
│   │   ├── 📄 app.py
│   │   ├── 📄 data_preprocessor_api.py
│   │   ├── 📄 device_optimizer_api.py
│   │   ├── 📄 inference_engine_api.py
│   │   ├── 📄 model_manager_api.py
│   │   ├── 📄 quantizer_api.py
│   │   └── 📄 train_engine_api.py
│   ├── 📁 engine/                  # Core ML engines
│   │   ├── 📄 __init__.py
│   │   ├── 📄 batch_processor.py
│   │   ├── 📄 data_preprocessor.py
│   │   ├── 📄 inference_engine.py
│   │   ├── 📄 lru_ttl_cache.py
│   │   ├── 📄 quantizer.py
│   │   └── 📄 train_engine.py
│   ├── 📁 optimizer/               # Optimization algorithms
│   │   ├── 📄 __init__.py
│   │   ├── 📄 configs.py
│   │   ├── 📄 device_optimizer.py  # 🆕 Device optimization
│   │   └── 📄 model_manager.py     # 🆕 Secure model management
│   ├── 📁 static/                  # 🆕 Static assets
│   └── 📁 utils/                   # Utility functions
├── 📁 temp_data/                   # 🆕 Temporary data storage
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
├── 📄 compose.yaml                 # 🆕 Docker Compose configuration
├── 📄 Dockerfile                   # 🆕 Docker containerization
├── 📄 kolosal_apilog               # API logging
├── 📄 LICENSE                      # MIT License
├── 📄 python-version               # Python version specification
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

## 🆕 What's New in **v0.1.2**

### 🎉 **Major Updates**

* **🚀 Gradio Web Interface** – Complete redesign from Streamlit to Gradio for better performance and user experience
* **🔧 Enhanced UV Integration** – Streamlined installation and dependency management with UV package manager
* **🎯 Dedicated Inference Server** – Production-ready inference endpoint with minimal latency
* **📊 Advanced Data Visualization** – Comprehensive data previews with correlation matrices and distribution plots
* **🔐 Secure Model Management** – Enhanced model encryption and access control features

### 🔧 **Technical Improvements**

* **Sample Dataset Integration** – Built-in access to popular ML datasets (Iris, Titanic, Boston Housing, etc.)
* **Real-time Training Progress** – Live updates during model training with detailed metrics
* **Performance Comparison Dashboard** – Side-by-side model evaluation and ranking
* **Enhanced Device Optimization** – Better GPU detection and memory management
* **Improved Error Handling** – More robust error messages and debugging information

### 🌟 **New Features**

* **Multiple Export Formats** – Support for Pickle, Joblib, and ONNX model exports
* **Command Line Interface** – Flexible CLI options for different deployment scenarios
* **Interactive Data Exploration** – In-browser data analysis with statistical summaries
* **Feature Importance Visualization** – Automated generation of feature importance plots
* **Model Encryption** – Secure model storage with password protection

### 💪 **Performance Enhancements**

* **Faster Model Loading** – Optimized model serialization and deserialization
* **Memory Optimization** – Reduced memory footprint during training and inference
* **Parallel Processing** – Enhanced multi-core utilization for training workflows
* **Caching System** – Intelligent caching for faster repeated operations

---

## 🚧 Roadmap

1. **Complete Test Suite** & CI green ✨
2. **REST API Endpoints** for programmatic access
3. **Docker Containerization** for easy deployment
4. **Model Monitoring** & drift detection
5. **AutoML Pipeline** with automated feature engineering
6. **Time‑series & anomaly‑detection** modules
7. **Cloud‑native deployment** recipes (AWS, GCP, Azure)
8. **MLOps Integration** with popular platforms

---

## 💻 Technology Stack

| Purpose           | Library                       |
| ----------------- | ----------------------------- |
| **Web UI**        | Gradio 🆕                     |
| **Package Mgmt**  | UV 🆕                         |
| **Data Ops**      | Pandas / NumPy                |
| **Core ML**       | scikit‑learn                  |
| **Boosting**      | XGBoost / LightGBM / CatBoost |
| **Visuals**       | Matplotlib / Seaborn          |
| **Serialisation** | Joblib / Pickle               |
| **Optimization**  | Optuna / Hyperopt             |

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
- **API Reference**: [docs/api.md](docs/api.md)
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

# Launch the web interface
uv run python app.py

# Open http://localhost:7860 in your browser and start experimenting! 🚀
```

---

<div align="center">

**Built with ❤️ by the Kolosal AI Team**

[🌟 Star us on GitHub](https://github.com/Genta-Technology/kolosal_automl) | [📖 Documentation](docs/) | [🐛 Report Issues](https://github.com/Genta-Technology/kolosal_automl/issues)

</div>