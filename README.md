# Advanced ML Training Engine 🤖

## Overview

The Advanced ML Training Engine is a sophisticated machine learning training and optimization framework designed to streamline the ML model development process. It provides a comprehensive suite of tools for data preprocessing, model training, hyperparameter optimization, and inference across various machine learning tasks.

## 🌟 Key Features

### 1. Flexible Model Training
- Support for multiple machine learning tasks (Classification, Regression, Clustering)
- Compatible with various ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)
- Automatic model selection and optimization

### 2. Supported Models

#### Classification
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- Support Vector Classification (SVC)

#### Regression
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Support Vector Regression (SVR)

### 3. Advanced Hyperparameter Optimization
- Multiple optimization strategies:
  - Grid Search
  - Random Search
  - Bayesian Optimization
  - Adaptive Surrogate-Assisted Hyperparameter Tuning (ASHT)
  - HyperX Advanced Optimization

### 4. Intelligent Preprocessing
- Automated feature scaling
- Missing value handling
- Outlier detection and management
- Feature selection techniques

### 5. Performance Optimization
- Device-aware configuration
- Adaptive batching
- Quantization support
- Parallel processing
- Memory optimization

### 6. Comprehensive Monitoring
- Detailed performance metrics
- Learning curve tracking
- Model performance visualization
- Experiment tracking and reporting

## 🚀 Installation

### Prerequisites
- Python 3.8+
- scikit-learn
- numpy
- pandas
- matplotlib
- Optional: XGBoost, LightGBM, CatBoost

### Install 
1. Clone the repository:
```bash
git clone https://github.com/Genta-Technology/ml-training-engine
cd ml-training-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Quick Start Example

```python
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy

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

## 🛠 Advanced Configuration

The framework offers extensive configuration options through `MLTrainingEngineConfig`:
- Task Type Selection
- Optimization Strategies
- Preprocessing Settings
- Quantization Options
- Performance Tuning Parameters

## 📊 Visualization and Reporting

- Automatic generation of performance reports
- Learning curve plots
- Feature importance visualization
- Model comparison metrics

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

Current unit test results:
- **tests/test_batch_processor.py**: PASSED
- **tests/test_lru_ttl_cache.py**: PASSED
- **tests/test_quantizer.py**: FAILED
- **tests/test_data_preprocessor.py**: FAILED
- **tests/test_engine.py**: FAILED

## 🚧 Planned Improvements

1. **Complete Testing**: Address and resolve failing tests
2. **UI Enhancements**: Improve user interface
3. **Code Optimization**: Refactor and optimize codebase
4. **Additional Export Formats**: Support ONNX and PMML model export
5. **Advanced Visualization**: Enhanced model comparison tools
6. **Time Series Support**: Extended time series forecasting functionality

## 💻 Technologies Used

- **Streamlit**: Frontend interface
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: ML algorithms and pipelines
- **XGBoost, LightGBM, CatBoost**: Advanced gradient boosting frameworks
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## 🧪 Running Tests

To run all unit tests:
```bash
pytest -vv
```

To run a specific test:
```bash
python -m unittest tests\test_<filename>.py
```

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