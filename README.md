# Advanced ML Training Engine 🤖

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-partial-yellow.svg)]()

## 📋 Overview

The Advanced ML Training Engine is a sophisticated machine learning framework that streamlines the entire ML model development lifecycle. It provides an integrated suite of tools for data preprocessing, model training, hyperparameter optimization, and inference across various machine learning tasks.

## 🌟 Key Features

### 🔄 Flexible Model Training
- Support for multiple machine learning tasks (Classification, Regression, Clustering)
- Seamless integration with popular ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)
- Intelligent model selection and automated optimization

### 🛠️ Comprehensive Model Support

| Classification | Regression |
|----------------|------------|
| Logistic Regression | Linear Regression |
| Random Forest Classifier | Random Forest Regressor |
| Gradient Boosting Classifier | Gradient Boosting Regressor |
| XGBoost Classifier | XGBoost Regressor |
| LightGBM Classifier | LightGBM Regressor |
| CatBoost Classifier | CatBoost Regressor |
| Support Vector Classification (SVC) | Support Vector Regression (SVR) |

### 🔍 Advanced Hyperparameter Optimization
- **Multiple optimization strategies:**
  - Grid Search
  - Random Search
  - Bayesian Optimization
  - Adaptive Surrogate-Assisted Hyperparameter Tuning (ASHT)
  - HyperX Advanced Optimization

### 🧠 Intelligent Preprocessing
- Automated feature scaling
- Sophisticated missing value handling
- Robust outlier detection and management
- Advanced feature selection techniques

### ⚡ Performance Optimization
- Device-aware configuration
- Adaptive batching
- Quantization support
- Parallel processing capabilities
- Memory-efficient operations

### 📊 Comprehensive Monitoring
- Detailed performance metrics
- Real-time learning curve tracking
- Interactive model performance visualization
- Experiment tracking and reporting

## 🚀 Installation

### Prerequisites
- Python 3.8+

### Option 1: Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/Genta-Technology/kolosal_automl
cd kolosal-automl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Virtual Environment Installation (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/Genta-Technology/kolosal_automl
cd kolosal_automl
```

2. Create a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies within the virtual environment:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Optional: Install additional modules for enhanced functionality:
```bash
# For XGBoost support
pip install xgboost

# For LightGBM support
pip install lightgbm

# For CatBoost support
pip install catboost
```

5. To deactivate the virtual environment when finished:
```bash
deactivate
```

## 💻 Quick Start Example

```python
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare your data
# X, y = load_your_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create configuration
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.HYPERX,
    cv_folds=5,
    test_size=0.2
)

# Initialize training engine
engine = MLTrainingEngine(config)

# Train models
best_model, metrics = engine.train_model(
    model=RandomForestClassifier(),
    model_name='RandomForest',
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    },
    X=X_train, 
    y=y_train
)

# Evaluate and save
engine.save_model(best_model)
predictions = engine.predict(X_test)
```

## 🛠️ Advanced Configuration

The framework offers extensive configuration options through `MLTrainingEngineConfig`:

```python
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,                    # ML task type
    optimization_strategy=OptimizationStrategy.BAYESIAN,  # Optimization method
    cv_folds=5,                                           # Cross-validation folds
    test_size=0.2,                                        # Test set proportion
    random_state=42,                                      # Random seed
    enable_quantization=True,                             # Enable model quantization
    batch_size=64,                                        # Processing batch size
    n_jobs=-1                                             # Parallel jobs (-1 = all cores)
)
```

## 📊 Visualization and Reporting

The framework automatically generates:
- Performance reports with key metrics
- Learning curve visualizations
- Feature importance charts
- Model comparison dashboards

## 🔍 Project Structure

```
ml-training-engine/
├── app.py                 # Main application entry point
├── modules/               # Core functionality modules
│   ├── configs.py         # Configuration classes
│   ├── engine/            # ML engine components
│   │   ├── train_engine.py   # Training engine
│   │   ├── batch_processor.py # Batch processing
│   │   └── inference_engine.py # Model inference
│   ├── optimizer/         # Optimization strategies
│   └── utils/             # Utility functions
├── models/                # Directory for saved models
├── exported_models/       # Directory for exported models
├── tests/                 # Unit tests
└── requirements.txt       # Project dependencies
```

## 🧪 Test Status

### Functional

| Test File                      | Status     |
|-------------------------------|------------|
| tests/functional/test_app_api.py                 | ❌ FAILED |
| tests\functional\test_quantizer_api.py           | ❌ FAILED |
| tests\functional\test_data_preprocessor_api.py   | ❌ FAILED |
| tests\functional\test_device_optimizer_api.py    | ❌ FAILED |
| tests\functional\test_inference_engine_api.py    | ❌ FAILED |
| tests\functional\test_train_engine_api.py        | ❌ FAILED |
| tests\functional\test_model_manager_api.py       | ❌ FAILED |

### Unit

| Test File                      | Status     |
|-------------------------------|------------|
| tests/unit/test_batch_processor.py       | ✅ PASSED |
| tests/unit/test_data_preprocessor.py     | ❌ FAILED |
| tests/unit/test_device_optimizer.py      | ❌ FAILED |
| tests/unit/test_inference_engine.py      | ❌ FAILED |
| tests/unit/test_lru_ttl_cache.py         | ✅ PASSED |
| tests/unit/test_model_manager.py         | ❌ FAILED |
| tests/unit/test_optimizer_asht.py        | ❌ FAILED |
| tests/unit/test_optimizer_hyperx.py      | ✅ PASSED |
| tests/unit/test_quantizer.py             | ❌ FAILED |
| tests/unit/test_train_engine.py          | ❌ FAILED |

Status: 
| ✅ PASSED | ❌ FAILED |⬜ NOT RUN | 🛑 ERROR |

To run all unit tests:
```bash
pytest -vv
```

To run a specific test:
```bash
python -m unittest <file_path>.py
```

## 🚧 Roadmap

1. **Test Suite Completion**: Address and resolve failing tests
2. **UI Enhancements**: Improve user interface and experience
3. **Code Optimization**: Refactor and optimize codebase
4. **Export Format Support**: Add ONNX and PMML model export
5. **Advanced Visualization**: Enhanced model comparison tools
6. **Time Series Support**: Extended time series forecasting functionality
7. **Cloud Integration**: Support for cloud-based deployment and scaling

## 💻 Technologies Used

- **Streamlit**: Interactive frontend interface
- **Pandas & NumPy**: Efficient data processing
- **Scikit-learn**: Core ML algorithms and pipelines
- **XGBoost, LightGBM, CatBoost**: Advanced gradient boosting frameworks
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
