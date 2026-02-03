# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-02-03

### Added

#### Rust Core Library (kolosal-core)

- **Preprocessing Engine**
  - `StandardScaler`, `MinMaxScaler`, `RobustScaler` for feature scaling
  - `OneHotEncoder`, `LabelEncoder`, `TargetEncoder` for categorical encoding
  - `MeanImputer`, `MedianImputer`, `ModeImputer` for missing values
  - `DataPreprocessor` pipeline for composable transformations

- **Training Engine**
  - K-Fold and Stratified K-Fold cross-validation
  - Metrics: R², MSE, RMSE, MAE, Accuracy, Precision, Recall, F1
  - Model registry with factory pattern

- **Models**
  - Linear Regression with L2 regularization (Ridge)
  - Logistic Regression
  - Decision Tree (classifier and regressor)
  - Random Forest (classifier and regressor)

- **Inference Engine**
  - Single and batch prediction
  - Quantization type definitions

- **Optimizer (HyperOptX)**
  - Search space with Float, Int, Categorical parameters
  - TPE (Tree-structured Parzen Estimator) sampler
  - Random and Grid samplers
  - Median, Percentile, and Hyperband pruners
  - ASHT (Adaptive Successive Halving)

- **Data I/O**
  - CSV loading with type inference
  - Parquet loading
  - Chunked/streaming for large files
  - File info inspection

- **Performance Utilities**
  - SIMD operations (sum, mean, variance, dot product, softmax)
  - Memory pool with allocation tracking
  - LRU cache with TTL support
  - Arena allocator (bumpalo)
  - Parallel execution with rayon

#### Python Bindings (kolosal-python)

- PyO3 bindings for all core modules
- NumPy array interop
- Pandas DataFrame ↔ Polars conversion
- Error handling with Python exceptions

#### Unified Python Package

- `kolosal` package with automatic backend selection
- `KOLOSAL_USE_RUST` environment variable for backend control
- Fallback to legacy Python code when Rust unavailable
- `get_backend_info()` for runtime introspection

### Changed

- Moved original Python code to `legacy/` directory
- Updated Polars API to 0.46 (Series → Column conversion)

### Technical Details

- Rust Edition 2021
- MSRV: 1.75.0
- Polars 0.46
- PyO3 0.20 with abi3-py39

## [0.1.0] - Initial Release

- Original Python implementation
- FastAPI REST API
- Gradio UI
- scikit-learn, XGBoost, LightGBM models
- Optuna hyperparameter optimization
