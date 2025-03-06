# Kolosal-AutoML

Kolosal-AutoML is a powerful, user-friendly automated machine learning platform designed to streamline your ML workflow. With its intuitive interface, you can upload data, train models, evaluate performance, and deploy solutions without deep ML expertise.

## Features

- **Data Upload & Exploration**: Import CSV or Excel files and gain immediate insights with automated visualizations and statistics
- **Intelligent Configuration**: Configure preprocessing, model selection, and optimization strategies through a simple interface
- **Automated Model Training**: Train multiple machine learning models simultaneously with optimized hyperparameters
- **Comprehensive Evaluation**: Compare model performance with detailed metrics and visualizations
- **Prediction Interface**: Make predictions using trained models with both batch upload and interactive input options
- **Export Capabilities**: Export trained models in various formats with sample code for implementation

## Supported Models

### Classification
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- Support Vector Classification (SVC)

### Regression
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Support Vector Regression (SVR)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Genta-Technology/kolosal-automl
cd kolosal-automl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage Guide

### 1. Data Upload & Exploration
- Upload your dataset (CSV or Excel)
- Explore data statistics, correlations, and distributions
- Select your target variable for prediction

### 2. Training Configuration
- Choose between classification and regression tasks
- Configure preprocessing options including normalization and missing value handling
- Select models to train and optimization strategies
- Customize advanced settings for batch processing and quantization

### 3. Model Training
- View dataset information and configuration summary
- Start the training process with a single click
- Monitor training progress in real-time

### 4. Model Evaluation
- Compare performance metrics across all trained models
- Visualize feature importance for better interpretability
- Identify the best-performing model automatically

### 5. Prediction
- Upload new data for batch predictions
- Use the interactive form for individual predictions
- Download prediction results

### 6. Export
- Export your models in various formats (Joblib, Pickle, etc.)
- Get sample code for implementing your model in production
- Export experiment results for documentation

## Project Structure

```
kolosal-automl/
├── app.py                 # Main Streamlit application
├── modules/               # Core functionality modules
│   ├── configs.py         # Configuration classes
│   ├── engine/            # ML engine components
│   │   └── train_engine.py # Training engine implementation
│   ├── preprocessing/     # Data preprocessing components
│   └── utils/             # Utility functions
├── models/                # Directory for saved models
├── exported_models/       # Directory for exported models
├── tests/                 # Unit tests
└── requirements.txt       # Project dependencies
```

## Test Status

The current unit test results are as follows:

- **tests/test_batch_processor.py**: PASSED
- **tests/test_lru_ttl_cache.py**: PASSED
- **tests/test_quantizer.py**: FAILED
- **tests/test_data_preprocessor.py**: FAILED
- **tests/test_engine.py**: FAILED

## Planned Improvements

The roadmap for Kolosal-AutoML includes:

1. **Complete Testing**: Address and resolve failing tests to achieve comprehensive test coverage
2. **UI Enhancements**: Improve and upgrade the user interface for a smoother user experience
3. **Code Optimization**: Refactor and optimize the codebase to improve performance and maintainability
4. **Additional Export Formats**: Support for ONNX and PMML model export
5. **Advanced Visualization**: Enhanced visualization tools for model comparison and evaluation
6. **Time Series Support**: Extended functionality for time series forecasting tasks

## Technologies Used

- **Streamlit**: Frontend interface
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: ML algorithms and pipelines
- **XGBoost, LightGBM, CatBoost**: Advanced gradient boosting frameworks
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## Running Tests

To run the unit tests, use the following command:

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.