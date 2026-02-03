# Getting Started with Rust Development

## Prerequisites

### Install Rust
```bash
# Install rustup (Rust toolchain installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### Install Development Tools
```bash
# Install essential Rust tools
cargo install cargo-watch      # Auto-rebuild on file changes
cargo install cargo-nextest    # Faster test runner
cargo install cargo-flamegraph # Performance profiling
cargo install cargo-deny       # Dependency auditing
cargo install maturin          # Python wheel builder

# Install clippy and rustfmt (usually included with rustup)
rustup component add clippy rustfmt
```

### Set Up IDE

**VS Code:**
```bash
# Install recommended extensions
code --install-extension rust-lang.rust-analyzer
code --install-extension vadimcn.vscode-lldb
code --install-extension serayuzgur.crates
```

**Other IDEs:**
- IntelliJ IDEA: Install Rust plugin
- Vim/Neovim: Install rust.vim and rust-analyzer
- Emacs: Install rustic

## Project Setup

### Create Workspace Structure
```bash
# Create workspace root
mkdir -p rust
cd rust

# Create workspace Cargo.toml
cat > Cargo.toml << 'EOF'
[workspace]
members = [
    "kolosal-core",
    "kolosal-engine",
    "kolosal-optimizer",
    "kolosal-preprocessing",
    "kolosal-models",
    "kolosal-python",
    "kolosal-api",
    "kolosal-cli",
]

[workspace.package]
version = "0.3.0"
edition = "2021"
license = "MIT"
authors = ["Genta Technology <contact@genta.tech>"]
repository = "https://github.com/Genta-Technology/kolosal-automl"

[workspace.dependencies]
# Core dependencies
ndarray = { version = "0.15", features = ["rayon", "serde"] }
polars = { version = "0.36", features = ["lazy", "parquet", "dtype-full"] }
rayon = "1.8"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"

# Numerical computing
num-traits = "0.2"
blas-src = { version = "0.10", default-features = false, features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "static"] }

# Python bindings
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py39"] }
numpy = "0.20"

# Web/API
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["full"] }

# Testing
proptest = "1.4"
criterion = { version = "0.5", features = ["html_reports"] }

[profile.dev]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"

[profile.bench]
inherits = "release"
debug = true

[profile.test]
opt-level = 1
EOF
```

### Create Core Crate
```bash
# Create kolosal-core
cargo new --lib kolosal-core
cd kolosal-core

# Update Cargo.toml
cat > Cargo.toml << 'EOF'
[package]
name = "kolosal-core"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[dependencies]
ndarray.workspace = true
rayon.workspace = true
serde.workspace = true
serde_json.workspace = true
anyhow.workspace = true
thiserror.workspace = true
num-traits.workspace = true

[dev-dependencies]
proptest.workspace = true
criterion.workspace = true

[[bench]]
name = "cache_bench"
harness = false
EOF

# Create basic module structure
mkdir -p src/{cache,memory,simd,config,errors}
```

## Development Workflow

### Building
```bash
# Build all crates
cargo build

# Build in release mode
cargo build --release

# Build specific crate
cargo build -p kolosal-core

# Watch and auto-rebuild
cargo watch -x build
```

### Testing
```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p kolosal-core

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run with nextest (faster)
cargo nextest run

# Run tests in parallel
cargo test -- --test-threads=4
```

### Linting and Formatting
```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy (linter)
cargo clippy

# Fix clippy warnings
cargo clippy --fix

# Strict clippy
cargo clippy -- -D warnings
```

### Benchmarking
```bash
# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench cache_bench

# Generate flamegraph
cargo flamegraph --bench cache_bench
```

### Documentation
```bash
# Generate documentation
cargo doc

# Open documentation in browser
cargo doc --open

# Generate documentation for all dependencies
cargo doc --no-deps --open
```

## Python Binding Development

### Setup Maturin
```bash
# Create Python binding crate
cd rust
maturin new kolosal-python --bindings pyo3

# Develop mode (install in current Python environment)
cd kolosal-python
maturin develop

# Build wheel
maturin build --release

# Build and install
maturin develop --release
```

### Test Python Bindings
```python
# test_bindings.py
import kolosal

# Test your Rust functions from Python
processor = kolosal.DataPreprocessor()
result = processor.fit_transform(data)
```

## Common Tasks

### Add a New Dependency
```bash
# Add to workspace dependencies (edit rust/Cargo.toml)
# Then in your crate Cargo.toml:
[dependencies]
new-crate.workspace = true
```

### Create a New Module
```bash
# In src/
mkdir new_module
touch new_module/mod.rs

# In src/lib.rs or src/main.rs:
# pub mod new_module;
```

### Profile Performance
```bash
# Using cargo flamegraph
cargo flamegraph --bin your-binary

# Using perf (Linux)
cargo build --release
perf record --call-graph dwarf ./target/release/your-binary
perf report

# Using valgrind
cargo build
valgrind --tool=callgrind ./target/debug/your-binary
```

### Check Dependencies
```bash
# Check for outdated dependencies
cargo outdated

# Audit dependencies for security issues
cargo audit

# Check licenses
cargo deny check licenses
```

## Useful Commands

### Cargo
```bash
cargo check              # Fast compilation check
cargo clean              # Remove build artifacts
cargo update             # Update dependencies
cargo tree               # Display dependency tree
cargo bloat              # Find what takes space in binary
cargo expand             # Expand macros
```

### Rustup
```bash
rustup update            # Update Rust toolchain
rustup show              # Show active toolchain
rustup doc               # Open offline documentation
rustup target list       # List available targets
rustup target add <target>  # Add cross-compilation target
```

## Best Practices

### Code Organization
- One module per file or directory
- Public API in `lib.rs`
- Private implementation in submodules
- Tests in `tests/` directory or `#[cfg(test)]` modules

### Error Handling
```rust
// Use Result for fallible operations
fn process_data(data: &[f64]) -> Result<Vec<f64>, ProcessError> {
    // ...
}

// Use anyhow for application errors
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let data = load_data()
        .context("Failed to load data")?;
    Ok(())
}

// Use thiserror for library errors
#[derive(thiserror::Error, Debug)]
pub enum ProcessError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("IO error")]
    Io(#[from] std::io::Error),
}
```

### Performance Tips
- Use `cargo build --release` for benchmarking
- Profile before optimizing
- Use `#[inline]` for small, frequently-called functions
- Prefer `&[T]` over `Vec<T>` for function parameters
- Use iterators instead of manual loops
- Leverage Rayon for parallel iterators

### Testing Tips
- Write unit tests for each module
- Use property-based testing with proptest
- Benchmark critical paths
- Test error conditions
- Use `#[should_panic]` for expected failures

## Resources

### Documentation
- [The Rust Programming Language Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [PyO3 User Guide](https://pyo3.rs/)

### Performance
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/)

### Community
- [Rust Users Forum](https://users.rust-lang.org/)
- [r/rust Subreddit](https://www.reddit.com/r/rust/)
- [Rust Discord](https://discord.gg/rust-lang)

## Troubleshooting

### Compilation Errors
- Read error messages carefully (Rust errors are very helpful)
- Use `cargo check` for faster feedback
- Check compiler suggestions

### Lifetime Issues
- Start with references, add lifetimes when needed
- Use `'static` for data that lives for entire program
- Consider `Arc` or `Rc` for shared ownership

### Performance Issues
- Profile first, optimize second
- Check for unnecessary clones
- Use `cargo bloat` to find binary size issues
- Consider using `jemalloc` allocator

### Python Binding Issues
- Ensure Python GIL is handled correctly
- Use `py.allow_threads()` for long operations
- Minimize Python/Rust boundary crossings
- Check for memory leaks with `valgrind`

## Next Steps

1. Set up your development environment
2. Create the initial workspace structure
3. Implement a simple proof-of-concept
4. Write benchmarks comparing Rust vs Python
5. Iterate and expand

For questions or issues, refer to the main [RUST_MIGRATION_PLAN.md](../../RUST_MIGRATION_PLAN.md).
