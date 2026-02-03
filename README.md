# Kolosal AutoML

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance AutoML framework with Rust core and Python bindings.

## Quick Start

```bash
# Install with Rust backend (recommended)
pip install kolosal[rust]

# Or install with Python backend (fallback)
pip install kolosal[python]
```

```python
from kolosal import DataPreprocessor, TrainEngine, get_backend_info

# Check which backend is active
print(get_backend_info())  # {'backend': 'rust', ...}

# Preprocessing (10x faster than sklearn)
preprocessor = DataPreprocessor(scaling="standard")
X_transformed = preprocessor.fit_transform(X_train)

# Training with cross-validation
engine = TrainEngine(task_type="classification", cv_folds=5)
result = engine.train(X_transformed, y_train, model_type="random_forest")
print(f"CV Accuracy: {result['mean_score']:.4f}")

# Inference
predictions = engine.predict(X_test)
```

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Data Preprocessing** | Scalers, Encoders, Imputers | ✅ |
| **Training Engine** | Cross-validation, Metrics | ✅ |
| **Models** | Linear, Logistic, Trees, Random Forest | ✅ |
| **HyperOptX** | TPE, Random, Grid samplers | ✅ |
| **Pruners** | Median, Percentile, Hyperband | ✅ |
| **SIMD Operations** | Vectorized math operations | ✅ |
| **Python Bindings** | PyO3 with NumPy/Pandas | ✅ |

## Performance

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| StandardScaler (1M×10) | 500ms | 45ms | **11x** |
| Random Forest fit | 15s | 1.8s | **8x** |
| Batch Inference (10K) | 50ms | 4ms | **12x** |
| Memory Usage | 1GB | 400MB | **60% less** |

## Project Structure

```
kolosal_automl/
├── rust/                     # Rust implementation
│   ├── kolosal-core/         # Core ML library (pure Rust)
│   └── kolosal-python/       # PyO3 Python bindings
├── python/                   # Unified Python package
│   └── kolosal/              # Main package with backend switching
├── legacy/                   # Original Python implementation
└── docs/                     # Documentation
```

## Building from Source

### Prerequisites

- Rust 1.75+ ([rustup](https://rustup.rs))
- Python 3.9+
- maturin (`pip install maturin`)

### Build Rust Library

```bash
cd rust
cargo build --release
cargo test --workspace
```

### Build Python Wheel

```bash
cd rust/kolosal-python
maturin build --release
# Or for development:
maturin develop --release
```

## Backend Selection

By default, the Rust backend is used. To use the Python fallback:

```bash
export KOLOSAL_USE_RUST=0
```

Or in Python:
```python
import os
os.environ["KOLOSAL_USE_RUST"] = "0"
from kolosal import DataPreprocessor  # Uses Python backend
```

## Documentation

- [Rust API Documentation](rust/README.md)
- [Changelog](CHANGELOG.md)
- [Migration Plan](docs/migration/RUST_MIGRATION_PLAN.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
