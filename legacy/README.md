# Legacy Code

This directory contains legacy/deprecated code that is no longer part of the main project.

## Contents

- **kolosal-python/**: PyO3 Python bindings for kolosal-core (deprecated)
- **tests-python/**: Python integration tests (deprecated)
- **rust-old/**: Old Rust build artifacts

## Note

The main project is now **pure Rust**. These Python bindings are kept for reference only.

If you need Python bindings, please use the Rust library via FFI or rebuild the PyO3 bindings from this legacy code.
