# Kolosal AutoML

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance AutoML framework with Rust core and Python bindings.

## Project Structure

```
kolosal_automl/
├── rust/                          # Rust implementation
│   ├── Cargo.toml                 # Workspace configuration
│   ├── kolosal-core/              # Core ML library
│   │   ├── src/
│   │   │   ├── lib.rs             # Library entry point
│   │   │   ├── error.rs           # Error types
│   │   │   ├── preprocessing/     # Data preprocessing
│   │   │   ├── training/          # Model training
│   │   │   ├── inference/         # Model inference
│   │   │   ├── optimizer/         # HyperOptX optimizer
│   │   │   └── utils/             # Utilities
│   │   └── benches/               # Performance benchmarks
│   └── kolosal-python/            # Python bindings (PyO3)
│       └── src/
│           ├── lib.rs             # Python module
│           ├── preprocessing.rs   # Preprocessing bindings
│           ├── training.rs        # Training bindings
│           ├── inference.rs       # Inference bindings
│           └── optimizer.rs       # Optimizer bindings
├── legacy/                        # Original Python implementation
│   ├── modules/                   # Python modules
│   ├── tests/                     # Python tests
│   └── ...
└── docs/
    └── migration/
        └── RUST_MIGRATION_PLAN.md # Detailed migration plan
```

## Features

### Core Engine (Rust)
- **Data Preprocessing**: Imputation, scaling, encoding with parallel processing
- **Training**: Multiple model types with cross-validation
- **Inference**: High-performance batch prediction with streaming support
- **HyperOptX**: Advanced hyperparameter optimization (TPE, Random, Bayesian)

### Python Bindings
- Drop-in replacement for legacy Python modules
- NumPy/Pandas integration
- Automatic type conversion

## Quick Start

### Building

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cd rust
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Python Installation

```bash
# Install maturin
pip install maturin

# Build and install Python package
cd rust/kolosal-python
maturin develop --release
```

### Usage (Python)

```python
import kolosal_automl as kml

# Preprocessing
config = kml.PreprocessingConfig().with_scaler(kml.ScalerType.Standard)
preprocessor = kml.DataPreprocessor(config)
df_processed = preprocessor.fit_transform(df)

# Training
train_config = kml.TrainingConfig(kml.TaskType.Regression, "target")
engine = kml.TrainEngine(train_config)
engine.fit(df_train)
predictions = engine.predict(df_test)

# Hyperparameter Optimization
search_space = kml.SearchSpace() \
    .float("learning_rate", 0.001, 0.1) \
    .int("n_estimators", 10, 500)

opt_config = kml.OptimizationConfig().with_n_trials(100).minimize()
optimizer = kml.HyperOptX(opt_config, search_space)

def objective(params):
    # Your training logic here
    return validation_score

result = optimizer.optimize(objective)
print(f"Best params: {result['best_params']}")
```

### Usage (Rust)

```rust
use kolosal_core::prelude::*;

// Preprocessing
let config = PreprocessingConfig::new()
    .with_scaler(ScalerType::Standard);
let mut preprocessor = DataPreprocessor::with_config(config);
let df_processed = preprocessor.fit_transform(&df)?;

// Training
let config = TrainingConfig::new(TaskType::Regression, "target");
let mut engine = TrainEngine::new(config);
engine.fit(&df_train)?;
let predictions = engine.predict(&df_test)?;

// Optimization
let search_space = SearchSpace::new()
    .float("learning_rate", 0.001, 0.1)
    .int("n_estimators", 10, 500);

let config = OptimizationConfig::new()
    .with_n_trials(100)
    .with_direction(OptimizeDirection::Minimize);

let mut optimizer = HyperOptX::new(config, search_space);
let study = optimizer.optimize(|params| {
    // Training logic
    Ok(validation_score)
})?;
```

## Performance

Target improvements over Python implementation:
- **10x+ faster** preprocessing
- **5-20x faster** training (depending on model)
- **50% less** memory usage
- **Native parallelism** via Rayon

## Development

### Running Tests

```bash
# All tests
cargo test

# Specific module
cargo test --package kolosal-core preprocessing

# With output
cargo test -- --nocapture
```

### Running Benchmarks

```bash
# All benchmarks
cargo bench

# Specific benchmark
cargo bench --bench preprocessing
```

## License

MIT License - see [LICENSE](LICENSE) for details.
