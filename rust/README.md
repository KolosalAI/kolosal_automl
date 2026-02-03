# Rust Workspace for Kolosal AutoML

This directory contains the Rust implementation of Kolosal AutoML. The migration is organized as a Cargo workspace with multiple crates for modularity and maintainability.

## Workspace Structure

```
rust/
├── Cargo.toml                   # Workspace configuration
├── kolosal-core/                # Core data structures and utilities
│   ├── src/
│   │   ├── cache/              # Caching implementations
│   │   ├── memory/             # Memory management
│   │   ├── simd/               # SIMD optimizations
│   │   ├── config/             # Configuration management
│   │   └── errors/             # Error types
│   └── Cargo.toml
├── kolosal-engine/              # Training and inference engines
│   ├── src/
│   │   ├── training/           # Training engine
│   │   ├── inference/          # Inference engine
│   │   ├── batch/              # Batch processing
│   │   ├── quantization/       # Model quantization
│   │   └── compiler/           # JIT compilation
│   └── Cargo.toml
├── kolosal-optimizer/           # Hyperparameter optimization
│   ├── src/
│   │   ├── search/             # Search algorithms
│   │   ├── asht/               # ASHT algorithm
│   │   └── adaptive/           # Adaptive HPO
│   └── Cargo.toml
├── kolosal-preprocessing/       # Data preprocessing
│   ├── src/
│   │   ├── transformers/       # Feature transformers
│   │   ├── encoders/           # Encoders
│   │   └── scalers/            # Scalers
│   └── Cargo.toml
├── kolosal-models/              # Model management
│   ├── src/
│   │   ├── registry/           # Model registry
│   │   └── manager/            # Model lifecycle
│   └── Cargo.toml
├── kolosal-python/              # Python bindings (PyO3)
│   ├── src/
│   │   └── lib.rs              # Python module exports
│   ├── Cargo.toml
│   └── pyproject.toml
├── kolosal-api/                 # REST API (Axum)
│   ├── src/
│   │   ├── routes/             # API routes
│   │   ├── middleware/         # Middleware
│   │   └── security/           # Security
│   └── Cargo.toml
└── kolosal-cli/                 # Command-line interface
    ├── src/
    │   └── main.rs
    └── Cargo.toml
```

## Getting Started

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.9+ (for Python bindings)
- Build tools (gcc, clang)

### Building
```bash
# Build all crates
cargo build

# Build in release mode (optimized)
cargo build --release

# Build specific crate
cargo build -p kolosal-core
```

### Testing
```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p kolosal-core

# Run with output
cargo test -- --nocapture
```

### Benchmarking
```bash
# Run all benchmarks
cargo bench

# Run benchmarks for specific crate
cargo bench -p kolosal-core
```

### Python Bindings
```bash
# Install maturin
pip install maturin

# Build and install Python package
cd kolosal-python
maturin develop --release
```

## Development

### Code Formatting
```bash
cargo fmt
```

### Linting
```bash
cargo clippy -- -D warnings
```

### Documentation
```bash
# Generate and open documentation
cargo doc --open
```

## Migration Status

See [MIGRATION_CHECKLIST.md](../../docs/rust/MIGRATION_CHECKLIST.md) for detailed progress tracking.

Current phase: **Phase 1 - Foundation & Infrastructure**

## Performance Goals

- **Inference:** 10-20x faster than Python
- **Training:** 5-10x faster for large datasets
- **Preprocessing:** 10-50x faster
- **Memory:** 30-50% reduction
- **API:** 20-50x higher throughput

## Contributing

Please refer to the main [RUST_MIGRATION_PLAN.md](../../RUST_MIGRATION_PLAN.md) for architecture and migration strategy.

## License

MIT License - see [LICENSE](../LICENSE) for details.
