# Advanced ML Training Engine 🤖

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-%23B072FF?logo=pypi)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-partial-yellow.svg)]()

---

## 📋 Overview

The **Advanced ML Training Engine** streamlines the entire machine‑learning lifecycle—from data ingestion to model deployment. It ships with intelligent preprocessing, state‑of‑the‑art hyper‑parameter optimisation, device‑aware acceleration, and first‑class experiment tracking.

---

## 🌟 Key Features

### 🔄 Flexible Model Training

* Multi‑task support: **classification**, **regression**, **clustering**
* Seamless integration with scikit‑learn, XGBoost, LightGBM & CatBoost
* Automated model selection & tuning

### 🛠️ Supported Algorithms <sup>(partial)</sup>

| Classification               | Regression                  |
| ---------------------------- | --------------------------- |
| Logistic Regression          | Linear Regression           |
| Random Forest Classifier     | Random Forest Regressor     |
| Gradient Boosting Classifier | Gradient Boosting Regressor |
| XGBoost Classifier           | XGBoost Regressor           |
| LightGBM Classifier          | LightGBM Regressor          |
| CatBoost Classifier          | CatBoost Regressor          |
| Support Vector Classifier    | Support Vector Regressor    |

### 🔍 Advanced Hyper‑parameter Optimisation

* **Grid Search**, **Random Search**, **Bayesian Optimisation**
* **ASHT** (Adaptive Surrogate‑Assisted Hyper‑parameter Tuning)
* **HyperX** (meta‑optimiser for large search spaces)

### 🧠 Smart Pre‑processing

* Auto‑scaling & encoding
* Robust missing‑value & outlier handling
* Feature selection / extraction pipelines

### ⚡ Performance Optimisation

* Device‑aware config & adaptive batching
* Quantisation & parallel execution
* Memory‑efficient data loaders

### 📊 Monitoring & Reporting

* Real‑time learning curves & metric dashboards
* Built‑in experiment tracker
* One‑click HTML / Markdown reports

---

## 🚀 Installation

### Prerequisites

* **Python 3.10 or newer**

### **Option 1 — Fast Setup with [UV](https://github.com/astral-sh/uv) 🔥 (Recommended)**

```bash
# 1. Clone the repository
git clone https://github.com/Genta-Technology/kolosal_automl.git
cd kolosal_automl

# 2. (Optional) create an isolated environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install uv (one‑time)
pip install --upgrade pip
pip install uv   # or: pipx install uv

# 4. Sync project dependencies (ultra‑fast!)
uv pip sync requirements.lock.txt  # preferred – reproducible
# – or –
uv pip install -r requirements.txt # if you don’t have a lock file yet
```

### Option 2 — Standard `pip`

```bash
git clone https://github.com/Genta-Technology/kolosal_automl.git
cd kolosal_automl
python -m venv venv && source venv/bin/activate  # create & activate venv
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** For GPU‑accelerated algorithms (XGBoost, LightGBM, CatBoost) install the respective extras:
>
> ```bash
> pip install xgboost lightgbm catboost
> ```

---

## 💻 Quick Start

```python
from modules.engine.train_engine import MLTrainingEngine
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationStrategy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your data
# X, y = load_your_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure the engine
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.HYPERX,
    cv_folds=5,
    test_size=0.2,
)

engine = MLTrainingEngine(config)

best_model, metrics = engine.train_model(
    model=RandomForestClassifier(),
    model_name="RandomForest",
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
    X=X_train,
    y=y_train,
)

engine.save_model(best_model)
predictions = engine.predict(X_test)
```

---

## 🧩 Advanced Configuration Example

```python
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    optimization_strategy=OptimizationStrategy.BAYESIAN,
    cv_folds=5,
    test_size=0.2,
    random_state=42,
    enable_quantization=True,
    batch_size=64,
    n_jobs=-1,
)
```

---

## 📊 Visualisation & Reporting

* Performance reports + confusion / ROC / PR curves
* Learning‑curve GIFs
* Feature‑importance bar charts
* HTML & Markdown experiment summaries

---

## 🔍 Project Structure (abridged)

```
kolosal_automl/
├── app.py
├── modules/
│   ├── configs.py
│   ├── engine/
│   │   ├── train_engine.py
│   │   ├── batch_processor.py
│   │   └── inference_engine.py
│   ├── optimizer/
│   └── utils/
├── models/
├── exported_models/
├── tests/
└── requirements.txt
```

---

## 🧪 Test Status

### Functional

| File                                              | Status   |
| ------------------------------------------------- | -------- |
| tests/functional/test/app_api.py                | ❌ FAILED |
| tests/functional/test/quantizer_api.py          | ❌ FAILED |
| tests/functional/test/data_preprocessor_api.py | ❌ FAILED |
| tests/functional/test/device_optimizer_api.py  | ❌ FAILED |
| tests/functional/test/inference_engine_api.py  | ❌ FAILED |
| tests/functional/test/train_engine_api.py      | ❌ FAILED |
| tests/functional/test/model_manager_api.py     | ❌ FAILED |

### Unit

| File                                   | Status   |
| -------------------------------------- | -------- |
| tests/unit/test/batch_processor.py   | ✅ PASSED |
| tests/unit/test/data_preprocessor.py | ❌ FAILED |
| tests/unit/test/device_optimizer.py  | ❌ FAILED |
| tests/unit/test/inference_engine.py  | ❌ FAILED |
| tests/unit/test/lru_ttl_cache.py    | ✅ PASSED |
| tests/unit/test/model_manager.py     | ❌ FAILED |
| tests/unit/test/optimizer_asht.py    | ❌ FAILED |
| tests/unit/test/optimizer_hyperx.py  | ✅ PASSED |
| tests/unit/test/quantizer.py          | ❌ FAILED |
| tests/unit/test/train_engine.py      | ❌ FAILED |

Run all tests:

```bash
pytest -vv
```

---

## 🆕 What’s New in **v0.1.1**

* **Training & Inference Optimisations** – faster epoch times and lower‑latency predictions.
* **Training Engine Fixes** – resolved edge‑case crashes during cross‑validation & improved error messages.
* **Device Optimiser Fixes** – correct GPU detection on hybrid CPU/GPU systems and smarter fallback logic.
* **Report Generation Speed‑ups** – Markdown & HTML reports now render up to 3× faster.
* **Explainability Report Patch** – SHAP/feature‑importance plots now correctly embed and save.
* **Hands‑on Tutorial Notebook** – added *Kolosal\_AutoML\_Tutorial.ipynb* with step‑by‑step examples.
  👉 [Open the notebook on GitHub](https://github.com/Genta-Technology/automl_tutorial)

## 🚧 Roadmap

1. **Complete Test Suite** \&ci green
2. UI/UX enhancements for Streamlit dashboard
3. ONNX & PMML export support
4. Advanced comparison visualiser
5. Time‑series & anomaly‑detection modules
6. Cloud‑native deployment recipes

---

## 💻 Technology Stack

| Purpose       | Library                       |
| ------------- | ----------------------------- |
| UI            | Streamlit                     |
| Data Ops      | Pandas / NumPy                |
| Core ML       | scikit‑learn                  |
| Boosting      | XGBoost / LightGBM / CatBoost |
| Visuals       | Matplotlib / Seaborn          |
| Serialisation | Joblib                        |

---

## 🤝 Contributing

1. Fork → `git checkout -b feature/foo`
2. Make changes & add tests
3. `pytest -q` to verify
4. Commit → push → PR

---

## 📄 License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
