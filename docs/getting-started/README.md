# 📖 Getting Started

Welcome to Kolosal AutoML! This guide will get you up and running in just a few minutes.

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
- ✅ Install Kolosal AutoML on your system
- ✅ Configure your development environment  
- ✅ Train your first machine learning model
- ✅ Make predictions using the trained model
- ✅ Access the web interface and API

## 📋 Table of Contents

1. [🔧 Installation](#-installation)
2. [⚙️ Configuration](#️-configuration)
3. [🚀 First Steps](#-first-steps)
4. [🎯 Quick Tutorial](#-quick-tutorial)
5. [🌐 Web Interface](#-web-interface)
6. [🔌 API Access](#-api-access)
7. [❓ Troubleshooting](#-troubleshooting)
8. [📚 Next Steps](#-next-steps)

## 🔧 Installation

### Prerequisites

Before installing Kolosal AutoML, make sure you have:

- **Python 3.10+** (Python 3.11 recommended)
- **4GB+ RAM** (8GB+ recommended for larger datasets)
- **10GB+ available disk space**
- **Git** for cloning the repository

### Option 1: Fast Setup with UV (Recommended) ⚡

UV is a fast Python package manager that significantly speeds up installation:

```bash
# 1. Install UV (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the repository
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# 3. Create and activate virtual environment
uv venv
# Activate:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies (ultra-fast with UV)
uv pip install -r requirements.txt

# 5. Optional: Install GPU-accelerated packages
uv pip install xgboost lightgbm catboost
```

### Option 2: Standard pip Installation

```bash
# 1. Clone repository
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Optional: Install with all features
pip install -e ".[all]"
```

### Option 3: Docker Installation 🐳

For containerized deployment:

```bash
# Clone and build
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# Quick start with Docker Compose
docker-compose up -d

# Access services:
# Web Interface: http://localhost:7860
# API Server: http://localhost:8000
```

## ⚙️ Configuration

### Environment Setup

After installation, configure your environment:

```bash
# Copy example configuration (optional)
cp .env.example .env

# Edit configuration if needed
# nano .env  # Linux/macOS
# notepad .env  # Windows
```

### Performance Optimization (Recommended)

Compile the project for better performance:

```bash
# Compile for 30-60% faster startup times
python main.py --compile

# Or use the compilation script
python compile.py

# Windows users can use:
compile.bat

# Test the performance improvement
python test_performance.py
```

### Enable Auto-Compilation (Optional)

For automatic compilation on startup:

```bash
# Windows (PowerShell):
$env:KOLOSAL_AUTO_COMPILE = "true"

# Linux/macOS:
export KOLOSAL_AUTO_COMPILE=true
```

## 🚀 First Steps

### Verify Installation

Let's make sure everything is working:

```bash
# Check system status
python main.py --system-info

# Run basic tests
python -c "from modules.engine.train_engine import MLTrainingEngine; print('✅ Installation successful!')"
```

### Launch Kolosal AutoML

You have three ways to run Kolosal AutoML:

```bash
# 1. Interactive mode (recommended for first-time users)
python main.py

# 2. Web interface directly
python main.py --mode gui

# 3. API server directly
python main.py --mode api
```

The interactive mode will show you a menu to choose from.

## 🎯 Quick Tutorial

Let's train your first machine learning model!

### Using Python API

```python
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load sample data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2. Configure the training engine
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.BAYESIAN,
    cv_folds=5,
    enable_automl=True
)

# 3. Initialize training engine
engine = MLTrainingEngine(config)

# 4. Train model (this will automatically try multiple algorithms)
best_model, metrics = engine.train_model(
    X=X_train,
    y=y_train,
    model_name="iris_classifier"
)

# 5. Make predictions
predictions = best_model.predict(X_test)
print(f"Model accuracy: {metrics['test_score']:.3f}")

# 6. Save the model
engine.save_model(best_model, "my_first_model.pkl")
```

### Using Command Line

```bash
# Train a model with built-in datasets
python -c "
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType
from sklearn.datasets import load_iris

# Quick model training
config = MLTrainingEngineConfig(task_type=TaskType.CLASSIFICATION)
engine = MLTrainingEngine(config)
data = load_iris()
model, metrics = engine.train_model(data.data, data.target, 'iris_model')
print(f'Trained model with accuracy: {metrics[\"test_score\"]:.3f}')
"
```

## 🌐 Web Interface

The web interface provides an intuitive way to work with machine learning:

### Starting the Web Interface

```bash
# Launch web interface
python main.py --mode gui

# Or directly with the app
python app.py

# With custom settings
python app.py --host 0.0.0.0 --port 7860 --share
```

### Web Interface Features

Once running (default: http://localhost:7860), you can:

1. **📊 Upload Data**: CSV, Excel, Parquet, or JSON files
2. **🔍 Explore Data**: View statistics, missing values, distributions
3. **⚙️ Configure Training**: Select algorithms, optimization strategy
4. **🚂 Train Models**: Monitor progress in real-time
5. **📈 View Results**: Compare models, see feature importance
6. **🔮 Make Predictions**: Test with new data
7. **💾 Save Models**: Export trained models

### Sample Datasets

The interface includes built-in datasets to try:
- **Iris**: Flower classification (150 samples)
- **Titanic**: Passenger survival prediction  
- **Boston Housing**: House price regression
- **Wine Quality**: Wine rating classification

## 🔌 API Access

For programmatic access, use the REST API:

### Starting the API Server

```bash
# Launch API server
python main.py --mode api

# Or directly
python start_api.py

# Access the API documentation
# http://localhost:8000/docs
```

### Basic API Usage

```bash
# Test API health
curl http://localhost:8000/health

# Get system information
curl http://localhost:8000/system/info

# For authenticated endpoints, you'll need an API key
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/models
```

### Python Client Example

```python
import requests

# Create API client
api_url = "http://localhost:8000"
headers = {"X-API-Key": "your-api-key"}

# Health check
response = requests.get(f"{api_url}/health")
print(f"API Status: {response.json()['status']}")

# Train a model via API
training_data = {
    "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
    "target": [0, 0],
    "model_type": "random_forest"
}

response = requests.post(
    f"{api_url}/api/train", 
    json=training_data, 
    headers=headers
)
print(f"Training result: {response.json()}")
```

## ❓ Troubleshooting

### Common Issues

#### 1. **Import Errors**
```bash
# If you see import errors, try:
pip install -r requirements.txt --upgrade

# For missing optional dependencies:
pip install -e ".[all]"
```

#### 2. **Performance Issues**
```bash
# Enable optimizations:
python main.py --compile
export KOLOSAL_AUTO_COMPILE=true

# Check system resources:
python main.py --system-info
```

#### 3. **Port Already in Use**
```bash
# Use different ports:
python app.py --port 7861  # Web interface
python start_api.py --port 8001  # API server

# Or kill existing processes:
# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Linux/macOS:
lsof -ti:7860 | xargs kill -9
```

#### 4. **Memory Issues**
```bash
# For large datasets, reduce memory usage:
python -c "
config = MLTrainingEngineConfig(
    memory_optimization=True,
    low_memory_mode=True,
    chunk_size=1000  # Smaller chunks
)
"
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the console output
2. **Run diagnostics**: `python main.py --system-info`
3. **Check requirements**: Ensure all dependencies are installed
4. **Review documentation**: Check relevant sections in this guide
5. **Ask for help**: Create an issue on GitHub with error details

## 📚 Next Steps

Congratulations! You now have Kolosal AutoML running. Here's what to explore next:

### 🎓 **Learn More**
- [📖 **User Guides**](../user-guides/) - Detailed tutorials for specific tasks
- [🌐 **Web Interface Guide**](../user-guides/web-interface.md) - Master the web interface
- [🔌 **API Usage**](../user-guides/api-usage.md) - Integrate with your applications

### 🚀 **Deploy to Production**
- [🐳 **Docker Deployment**](../deployment/docker.md) - Containerized deployment
- [🏭 **Production Guide**](../deployment/production.md) - Production-ready setup
- [🔒 **Security Setup**](../deployment/security.md) - Secure your deployment

### 🛠️ **Advanced Features**
- [⚡ **Performance Optimization**](../technical/performance.md) - Speed up your workflows
- [📊 **Monitoring Setup**](../deployment/monitoring.md) - Track system performance
- [🧩 **Module Documentation**](../technical/modules/) - Deep dive into components

### 🤝 **Get Involved**
- [👩‍💻 **Development Setup**](../development/setup.md) - Set up development environment
- [🤝 **Contributing**](../development/contributing.md) - Contribute to the project
- [🧪 **Testing**](../development/testing.md) - Help improve quality

## 🎉 Welcome to Kolosal AutoML!

You're now ready to build amazing machine learning applications with Kolosal AutoML. The platform provides:

- ✅ **Easy-to-use interfaces** for all skill levels
- ✅ **Production-ready performance** with optimization features
- ✅ **Flexible deployment options** from laptop to cloud
- ✅ **Comprehensive monitoring** and observability
- ✅ **Enterprise-grade security** features

Happy machine learning! 🚀

---

**Need help?** Check out our [User Guides](../user-guides/) or [create an issue](https://github.com/Genta-Technology/kolosal-automl/issues) on GitHub.

*Getting Started Guide v1.0 | Last updated: January 2025*
