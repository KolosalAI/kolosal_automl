# Kolosal-AutoML

Kolosal-AutoML is a streamlined Auto-ML tool designed to accelerate your machine learning development process. Develop, test, and optimize your machine learning pipelines effortlessly.

## Overview

Kolosal-AutoML provides a flexible framework for automating various stages of machine learning projects, from data preprocessing to model quantization and engine execution. This repository includes tests to ensure each component performs as expected.

## Test Status

The current unit test results are as follows:

- **tests/test_batch_processor.py**: PASSED
- **tests/test_lru_ttl_cache.py**: PASSED
- **tests/test_quantizer.py**: FAILED
- **tests/test_data_preprocessor.py**: FAILED
- **tests/test_engine.py**: FAILED

## Planned Improvements

The roadmap for Kolosal-AutoML includes:

1. **Complete Testing**: Address and resolve failing tests to achieve comprehensive test coverage.
2. **UI Enhancements**: Improve and upgrade the user interface for a smoother user experience.
3. **Code Optimization**: Refactor and optimize the codebase to improve performance and maintainability.

## Running the Tests

To run the unit tests, use the following command:

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
