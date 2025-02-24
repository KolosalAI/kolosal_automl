import polars as pl
import logging
import numpy as np
#from modules.hyperopt import HyperparameterOptimizer
from modules.feature_selection import MRMRSelector
from modules.model_selection import HAMSeOptimizer

logging.basicConfig(level=logging.INFO)  # Minimal change: configure logging
logger = logging.getLogger(__name__)

df = pl.read_csv(r"data\iris.csv")  # Minimal change: use raw string to fix the file path
logger.info(f"Loaded dataset with {len(df)} rows")
print(df.head())
data = df.to_numpy()
print(data[:10])
X = data[:, :-1]
y = data[:, -1]
logger.info(f"Extracted features and labels: {X.shape}, {y.shape}")
logger.info(f"Features: {X[:10]}")
logger.info(f"Labels: {y[:10]}")
logger.info(f"Unique labels: {np.unique(y)}")

# Minimal change: pass logger to hyperopt
selector_fixed = MRMRSelector(task="classification")
selector_fixed.fit(X=X, y=y)
selected_x = selector_fixed.transform(X)
logger.info(f"Selected features: {selector_fixed.selected_features_}")
logger.info(f"Selected features: {selected_x[:10]}")
optimizer = HAMSeOptimizer()
ensemble_predictor = optimizer.fit(X, y)
logger.info(f"Ensemble predictor: {ensemble_predictor}")
