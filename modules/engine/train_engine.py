import os
import sys
import gc
import json
import time
import pickle
import shutil
import logging
import traceback
import threading
import uuid
import numpy as np
import polars as pl

from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import asdict
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.exceptions import NotFittedError

# Conditional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mkl
    MKL_AVAILABLE = True
except ImportError:
    MKL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from modules.configs import CPUTrainingConfig, ModelType, OptimizationStrategy, DataSplitStrategy, TrainingState
from modules.engine.data_preprocessor import DataPreprocessor


class PolarsPreprocessor:
    """Adaptation layer to make sklearn preprocessors work with Polars dataframes."""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.column_order = None
    
    def fit(self, df: pl.DataFrame):
        # Convert to numpy for sklearn preprocessor
        pandas_df = df.to_pandas()
        self.column_order = pandas_df.columns.tolist()
        self.preprocessor.fit(pandas_df)
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # Ensure columns match expected order
        if self.column_order:
            missing_cols = set(self.column_order) - set(df.columns)
            if missing_cols:
                for col in missing_cols:
                    df = df.with_columns(pl.lit(None).alias(col))
            
            # Reorder columns to match expected order
            df = df.select(self.column_order)
        
        # Convert to numpy, transform, convert back to polars
        pandas_df = df.to_pandas()
        transformed = self.preprocessor.transform(pandas_df)
        
        # If result is numpy array, convert to DataFrame with original column names
        if isinstance(transformed, np.ndarray):
            if transformed.shape[1] == len(self.column_order):
                return pl.DataFrame(transformed, schema=self.column_order)
            else:
                # Column count changed - use generic names
                return pl.DataFrame(transformed)
        
        # If result is already a pandas DataFrame, convert to polars
        return pl.from_pandas(transformed)


class TrainingEngine:
    """
    A production-ready engine for CPU-based training of ML models at scale.
    Uses polars for efficient data handling and implements various optimizations
    for memory management, parallel processing, and error handling.
    """

    def __init__(self, config: 'CPUTrainingConfig', logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or self._setup_logger()

        self.state = TrainingState.INITIALIZING
        self.start_time = datetime.now()
        self.end_time = None

        # Datasets in memory - using polars instead of pandas
        self.train_data = None
        self.validation_data = None
        self.test_data = None

        # Feature columns / preprocessor
        self.feature_columns: List[str] = []
        self.preprocessor = None

        # Model references
        self.model = None
        self.best_model = None

        # Evaluation results
        self.evaluation_results = {}

        # For resource monitoring
        self.monitoring_thread = None
        self.stop_monitoring_event = threading.Event()
        self.resource_usage = []

        # Unique ID for this training run
        self.training_id = str(uuid.uuid4())

        # Runtime metrics
        self.runtime_metrics = {
            "preprocessing_time": 0,
            "feature_engineering_time": 0,
            "training_time": 0,
            "evaluation_time": 0,
            "optimization_time": 0,
            "total_time": 0,
            "peak_memory_usage_mb": 0
        }

        # Copy callbacks if provided
        self.callbacks = dict(self.config.callbacks) if self.config.callbacks else {}

        # If using Intel MKL, optionally set threads
        self._configure_resources()

        # Prepare and validate internal components (preprocessor, etc.)
        self._initialize_components()

        self.logger.info(f"Training Engine initialized (ID={self.training_id})")
        self.logger.debug(f"Configuration:\n{json.dumps(config.to_dict(), indent=2, default=str)}")

        self._set_state(TrainingState.READY)

    def _setup_logger(self) -> logging.Logger:
        """Set up a structured logger for the training engine with rotation."""
        logger = logging.getLogger(f"training_engine_{int(time.time())}")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        # Console handler with structured format
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            os.makedirs("logs", exist_ok=True)
            fh = RotatingFileHandler(
                f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            fh.setLevel(logger.level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            # Fall back to regular file handler if RotatingFileHandler fails
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            fh.setLevel(logger.level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.warning(f"Failed to set up rotating file handler: {e}. Using standard file handler.")

        return logger

    def _configure_resources(self):
        """Configure CPU threads, memory limits, etc., based on config."""
        if self.config.enable_intel_optimization and MKL_AVAILABLE:
            self.logger.info(f"Setting MKL threads to {self.config.num_threads}")
            mkl.set_num_threads(self.config.num_threads)

        # Set environment variables for thread control
        thread_count = str(self.config.num_threads)
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count
        os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
        os.environ["NUMEXPR_NUM_THREADS"] = thread_count
        
        # Configure polars thread pool
        try:
            pl.Config.set_global_threads(self.config.num_threads)
            self.logger.info(f"Polars global threads set to {self.config.num_threads}")
        except Exception as e:
            self.logger.warning(f"Failed to set Polars thread count: {e}")

        if self.config.memory_limit_gb:
            self.logger.info(f"Memory limit set to {self.config.memory_limit_gb} GB")
            # We'll use this in our memory monitoring

        if self.config.enable_distributed_training:
            self.logger.warning(
                f"Distributed training is enabled. Backend: {self.config.distributed_backend}. "
                "Currently, only a placeholder – actual distributed logic not yet implemented."
            )

    def _initialize_components(self):
        """
        Initialize preprocessor, check dependency availability, etc.
        This is also where you'd set up custom data preprocessors or chunk-based readers.
        """
        self.logger.info("Initializing data preprocessor...")
        try:
            base_preprocessor = DataPreprocessor(
                handle_missing=self.config.dataset.handle_missing,
                handle_outliers=self.config.dataset.handle_outliers,
                handle_categorical=self.config.dataset.handle_categorical,
                encoding_method=self.config.feature_engineering.encoding_method,
                enable_scaling=self.config.feature_engineering.enable_scaling,
                scaling_method=self.config.feature_engineering.scaling_method,
            )
            # Wrap with adapter for Polars
            self.preprocessor = PolarsPreprocessor(base_preprocessor)
            self.logger.info("Data preprocessor initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize preprocessor: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise
        
        self._check_dependencies()

    def _check_dependencies(self):
        """Check that required libraries for the chosen model are installed."""
        mt = self.config.model_training.model_type
        if mt == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost model requested but xgboost is not installed.")
        if mt == ModelType.LIGHTGBM and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM model requested but lightgbm is not installed.")
        if mt == ModelType.SKLEARN and not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn requested but is not installed.")

    def _set_state(self, new_state: 'TrainingState'):
        """Update the engine state and trigger appropriate callbacks."""
        old_state = self.state
        self.state = new_state
        self.logger.info(f"State transition: {old_state.name} -> {new_state.name}")

        # Execute callback if defined
        callback_name = f"on_{new_state.name.lower()}"
        if callback_name in self.callbacks:
            try:
                self.callbacks[callback_name](self)
            except Exception as e:
                self.logger.error(f"Error in callback {callback_name}: {str(e)}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())

    @contextmanager
    def _time_context(self, key: str):
        """Context manager to measure execution time of a block and update metrics."""
        start = time.time()
        peak_memory_before = self._get_current_memory_usage()
        try:
            yield
        finally:
            elapsed = time.time() - start
            peak_memory_after = self._get_current_memory_usage()
            peak_memory_increase = max(0, peak_memory_after - peak_memory_before)
            
            self.runtime_metrics[key] += elapsed
            self.runtime_metrics["peak_memory_usage_mb"] = max(
                self.runtime_metrics["peak_memory_usage_mb"], 
                peak_memory_after
            )
            
            self.logger.debug(
                f"{key} took {elapsed:.2f} seconds. "
                f"Memory usage: {peak_memory_after:.1f} MB (+{peak_memory_increase:.1f} MB)"
            )

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                return memory_info.rss / (1024 * 1024)  # Convert to MB
            except Exception:
                return 0
        return 0

    # ------------------
    # MONITORING
    # ------------------
    def start_monitoring(self):
        """Start background resource monitoring if enabled."""
        if not self.config.enable_monitoring or not PSUTIL_AVAILABLE:
            return

        def monitor():
            try:
                self.logger.info("Resource monitoring started")
                while not self.stop_monitoring_event.is_set():
                    try:
                        proc = psutil.Process(os.getpid())
                        with proc.oneshot():  # More efficient batch collection of metrics
                            cpu = proc.cpu_percent(interval=0.1)
                            mem = proc.memory_info()
                            mem_percent = proc.memory_percent()
                            
                            # Also check system-wide resources
                            system_cpu = psutil.cpu_percent()
                            system_mem = psutil.virtual_memory()

                        resource_data = {
                            'timestamp': datetime.now().isoformat(),
                            'cpu_percent': cpu,
                            'system_cpu_percent': system_cpu,
                            'rss_mb': mem.rss / (1024 ** 2),
                            'memory_percent': mem_percent,
                            'system_memory_percent': system_mem.percent,
                            'state': self.state.name
                        }
                        
                        self.resource_usage.append(resource_data)

                        # Log periodically to not flood logs
                        if len(self.resource_usage) % 30 == 0:
                            self.logger.debug(
                                f"Resource usage: Process CPU={cpu:.1f}%, "
                                f"System CPU={system_cpu:.1f}%, "
                                f"RSS={mem.rss/1024**2:.1f}MB ({mem_percent:.1f}%), "
                                f"System Mem={system_mem.percent:.1f}%"
                            )

                        # Check if we're approaching memory limit
                        if self.config.memory_limit_gb and mem.rss > self.config.memory_limit_gb * 0.9 * 1024**3:
                            self.logger.warning(
                                f"Approaching memory limit: {mem.rss/1024**3:.2f}GB used, "
                                f"limit is {self.config.memory_limit_gb}GB"
                            )

                        time.sleep(self.config.monitoring_interval)
                    except Exception as e:
                        self.logger.error(f"Error in monitoring thread: {e}")
                        time.sleep(5)  # Avoid tight loop in case of recurring errors
            except Exception as e:
                self.logger.error(f"Fatal error in monitoring thread: {e}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring_event.set()
            self.monitoring_thread.join(timeout=5)
            if self.monitoring_thread.is_alive():
                self.logger.warning("Resource monitoring thread did not terminate gracefully")
            else:
                self.logger.info("Resource monitoring stopped")

    # ------------------
    # DATA LOADING
    # ------------------
    def load_data(self):
        """
        Load data from local files or streaming sources.
        Handle train/val/test splits if needed, plus optional imbalance correction.
        Optimized for large datasets using Polars.
        """
        self._set_state(TrainingState.PREPROCESSING)
        ds_cfg = self.config.dataset

        with self._time_context("preprocessing_time"):
            if ds_cfg.enable_chunking and ds_cfg.train_path:
                # Use Polars' lazy loading and streaming capabilities
                self.logger.info(f"Lazy/chunk-based data loading from: {ds_cfg.train_path}")
                self.train_data = self._load_chunked_data(ds_cfg.train_path)
            else:
                # Normal direct load
                if ds_cfg.train_path:
                    self.logger.info(f"Loading train dataset from {ds_cfg.train_path}")
                    self.train_data = self._load_single_file(ds_cfg.train_path)

            # Optional separate val/test files
            if ds_cfg.validation_path:
                self.logger.info(f"Loading validation dataset from {ds_cfg.validation_path}")
                self.validation_data = self._load_single_file(ds_cfg.validation_path)

            if ds_cfg.test_path:
                self.logger.info(f"Loading test dataset from {ds_cfg.test_path}")
                self.test_data = self._load_single_file(ds_cfg.test_path)

            # If no separate val/test, we might split below:
            if (not self.validation_data) and (not self.test_data):
                self.logger.info("No separate validation/test sets found. Splitting from training data...")
                self._split_dataset()

            # Now that we have train data, we can set up features and target
            if ds_cfg.target_column is None:
                raise ValueError("No target column set in dataset config.")
            self._extract_features_and_target()

            # Class imbalance handling
            if ds_cfg.handle_imbalance and self.config.dataset.imbalance_strategy:
                self._handle_class_imbalance()

            # Apply preprocessing and feature engineering
            self._apply_preprocessing()

    def _load_single_file(self, path: str) -> pl.DataFrame:
        """Load a single file efficiently using Polars."""
        try:
            if path.endswith('.csv'):
                return pl.read_csv(
                    path,
                    separator=self.config.dataset.csv_separator,
                    encoding=self.config.dataset.encoding,
                    has_header=self.config.dataset.has_header
                )
            elif path.endswith('.parquet'):
                return pl.read_parquet(path)
            elif path.endswith('.ipc') or path.endswith('.arrow'):
                return pl.read_ipc(path)
            elif path.endswith('.json'):
                return pl.read_json(path)
            elif path.endswith('.pkl') or path.endswith('.pickle'):
                with open(path, 'rb') as f:
                    df = pickle.load(f)
                    # If pandas, convert to polars
                    if hasattr(df, 'to_dict'):  # Simple check for pandas-like object
                        return pl.from_pandas(df)
                    return df
            else:
                raise ValueError(f"Unsupported file format: {path}")
        except Exception as e:
            self.logger.error(f"Error loading file {path}: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    def _load_chunked_data(self, path: str) -> pl.DataFrame:
        """
        Load data in chunks using Polars streaming capabilities.
        Useful for large datasets that don't fit in memory.
        """
        chunk_size = self.config.dataset.chunk_size or 100000
        
        try:
            # Use polars lazy evaluation for large files
            if path.endswith('.csv'):
                self.logger.info(f"Streaming CSV in chunks of {chunk_size} rows")
                df = pl.scan_csv(
                    path,
                    separator=self.config.dataset.csv_separator,
                    encoding=self.config.dataset.encoding,
                    has_header=self.config.dataset.has_header
                )
            elif path.endswith('.parquet'):
                self.logger.info(f"Streaming Parquet in chunks")
                df = pl.scan_parquet(path)
            else:
                self.logger.warning(f"Chunked loading not optimized for {path}. Loading entire file.")
                return self._load_single_file(path)
            
            # Execute the lazy query
            return df.collect()
            
        except Exception as e:
            self.logger.error(f"Error loading chunked data from {path}: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    def _split_dataset(self):
        """Split self.train_data into train/validation/test based on config."""
        ds_cfg = self.config.dataset
        
        # Convert to pandas temporarily for sklearn's split functions
        # In future, we could implement these directly with polars
        train_pandas = self.train_data.to_pandas()
        
        if ds_cfg.split_strategy == DataSplitStrategy.RANDOM:
            # Typical random split
            tv_data, test_data = train_test_split(
                train_pandas,
                test_size=ds_cfg.test_size,
                random_state=ds_cfg.random_seed
            )
            val_ratio = ds_cfg.validation_size / (1 - ds_cfg.test_size)
            train_data, val_data = train_test_split(
                tv_data,
                test_size=val_ratio,
                random_state=ds_cfg.random_seed
            )
        elif ds_cfg.split_strategy == DataSplitStrategy.STRATIFIED:
            tv_data, test_data = train_test_split(
                train_pandas,
                test_size=ds_cfg.test_size,
                random_state=ds_cfg.random_seed,
                stratify=train_pandas[ds_cfg.stratify_column]
            )
            val_ratio = ds_cfg.validation_size / (1 - ds_cfg.test_size)
            train_data, val_data = train_test_split(
                tv_data,
                test_size=val_ratio,
                random_state=ds_cfg.random_seed,
                stratify=tv_data[ds_cfg.stratify_column]
            )
        elif ds_cfg.split_strategy == DataSplitStrategy.TIME_SERIES:
            # Time-series split implementation
            if not ds_cfg.time_column:
                raise ValueError("time_column must be specified for TIME_SERIES split strategy")
            
            # Sort by time column
            train_pandas = train_pandas.sort_values(ds_cfg.time_column)
            
            # Calculate split indices
            n = len(train_pandas)
            test_idx = int(n * (1 - ds_cfg.test_size))
            val_idx = int(test_idx * (1 - ds_cfg.validation_size / (1 - ds_cfg.test_size)))
            
            # Split data
            train_data = train_pandas[:val_idx]
            val_data = train_pandas[val_idx:test_idx]
            test_data = train_pandas[test_idx:]
        elif ds_cfg.split_strategy == DataSplitStrategy.GROUP:
            if not ds_cfg.group_column:
                raise ValueError("group_column must be specified for GROUP split strategy")
            
            # This is a simple implementation - more sophisticated group-based splitting
            # would require a custom implementation
            groups = train_pandas[ds_cfg.group_column].unique()
            np.random.seed(ds_cfg.random_seed)
            np.random.shuffle(groups)
            
            n_groups = len(groups)
            test_groups = groups[:int(n_groups * ds_cfg.test_size)]
            remaining_groups = groups[int(n_groups * ds_cfg.test_size):]
            val_groups = remaining_groups[:int(len(remaining_groups) * ds_cfg.validation_size / (1 - ds_cfg.test_size))]
            train_groups = remaining_groups[int(len(remaining_groups) * ds_cfg.validation_size / (1 - ds_cfg.test_size)):]
            
            train_data = train_pandas[train_pandas[ds_cfg.group_column].isin(train_groups)]
            val_data = train_pandas[train_pandas[ds_cfg.group_column].isin(val_groups)]
            test_data = train_pandas[train_pandas[ds_cfg.group_column].isin(test_groups)]
        else:
            raise ValueError(f"Unsupported split strategy: {ds_cfg.split_strategy}")
        
        # Convert back to polars
        self.train_data = pl.from_pandas(train_data)
        self.validation_data = pl.from_pandas(val_data)
        self.test_data = pl.from_pandas(test_data)
        
        self.logger.info(
            f"Data split complete. Train: {len(self.train_data)} rows, "
            f"Validation: {len(self.validation_data)} rows, "
            f"Test: {len(self.test_data)} rows"
        )

    def _extract_features_and_target(self):
        """Identify which columns are features vs. the target."""
        ds_cfg = self.config.dataset

        if ds_cfg.feature_columns:
            self.feature_columns = ds_cfg.feature_columns
        else:
            # Get all column names from the dataframe
            all_cols = self.train_data.columns
            exclude = {ds_cfg.target_column}
            if ds_cfg.id_column:
                exclude.add(ds_cfg.id_column)
            if ds_cfg.weight_column:
                exclude.add(ds_cfg.weight_column)
            if ds_cfg.stratify_column and ds_cfg.stratify_column not in exclude:
                exclude.add(ds_cfg.stratify_column)
            if ds_cfg.time_column and ds_cfg.time_column not in exclude:
                exclude.add(ds_cfg.time_column)
            if ds_cfg.group_column and ds_cfg.group_column not in exclude:
                exclude.add(ds_cfg.group_column)
                
            self.feature_columns = [c for c in all_cols if c not in exclude]

        self.logger.info(f"Number of feature columns: {len(self.feature_columns)} | Target: {ds_cfg.target_column}")
        
        # Log data types for debugging
        if self.config.debug_mode:
            dtypes = {col: str(dtype) for col, dtype in zip(self.train_data.columns, self.train_data.dtypes)}
            self.logger.debug(f"Data types: {dtypes}")

    def _handle_class_imbalance(self):
        """
        Handle class imbalance for classification tasks.
        Note: This requires conversion to pandas as imblearn doesn't support Polars directly.
        """
        if self.config.model_training.problem_type != "classification":
            return

        strategy = self.config.dataset.imbalance_strategy.lower()
        self.logger.info(f"Handling class imbalance via: {strategy} strategy")
        
        # Convert to pandas for resampling
        train_df = self.train_data.to_pandas()
        X = train_df[self.feature_columns]
        y = train_df[self.config.dataset.target_column]
        
        # Log class distribution before
        class_counts_before = y.value_counts().to_dict()
        self.logger.info(f"Class distribution before resampling: {class_counts_before}")
        
        try:
            # Import imblearn conditionally
            try:
                from imblearn.over_sampling import RandomOverSampler, SMOTE
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.combine import SMOTEENN, SMOTETomek
                imblearn_available = True
            except ImportError:
                imblearn_available = False
                self.logger.warning("imblearn not available. Skipping class imbalance handling.")
                return
            
            if not imblearn_available:
                return
                
            if strategy == "oversample":
                sampler = RandomOverSampler(random_state=self.config.dataset.random_seed)
            elif strategy == "undersample":
                sampler = RandomUnderSampler(random_state=self.config.dataset.random_seed)
            elif strategy == "smote":
                sampler = SMOTE(random_state=self.config.dataset.random_seed)
            elif strategy == "smote_enn":
                sampler = SMOTEENN(random_state=self.config.dataset.random_seed)
            elif strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=self.config.dataset.random_seed)
            else:
                self.logger.warning(f"Unknown imbalance strategy: {strategy}. Skipping resampling.")
                return

            # Apply resampling
            X_res, y_res = sampler.fit_resample(X, y)
            
            # Reconstruct DataFrame and convert back to Polars
            resampled_df = pd.DataFrame(X_res, columns=self.feature_columns)
            resampled_df[self.config.dataset.target_column] = y_res
            
            # Preserve any non-feature columns that weren't included in X
            for col in train_df.columns:
                if col not in resampled_df.columns and col != self.config.dataset.target_column:
                    if col in train_df.columns:
                        # For any column that was in original but not in resampled, handle it
                        # This is a simplification - in production you might need a more sophisticated approach
                        resampled_df[col] = np.nan
            
            self.train_data = pl.from_pandas(resampled_df)
            
            # Log class distribution after
            class_counts_after = y_res.value_counts().to_dict()
            self.logger.info(f"Class distribution after resampling: {class_counts_after}")
            self.logger.info(f"Resampled training set shape: {X_res.shape}")
            
        except Exception as e:
            self.logger.error(f"Error during imbalance handling: {str(e)}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            # Continue with original data
            self.logger.info("Continuing with original data due to error in resampling")

    def _apply_preprocessing(self):
        """Fit the preprocessor on train_data, then transform train/val/test."""
        self.logger.info("Applying preprocessing to datasets")
        try:
            # Fit preprocessor on training data
            self.logger.info("Fitting preprocessor on train data")
            self.preprocessor.fit(self.train_data)
            
            # Transform datasets
            self.logger.info("Transforming train data")
            self.train_data = self.preprocessor.transform(self.train_data)
            
            if self.validation_data is not None:
                self.logger.info("Transforming validation data")
                self.validation_data = self.preprocessor.transform(self.validation_data)
                
            if self.test_data is not None:
                self.logger.info("Transforming test data")
                self.test_data = self.preprocessor.transform(self.test_data)
                
            self.logger.info("Preprocessing complete")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    # ------------------
    # FEATURE ENGINEERING
    # ------------------
    def engineer_features(self):
        """
        Perform feature engineering steps on the datasets.
        Uses Polars expressions for efficient computation.
        """
        self._set_state(TrainingState.FEATURE_ENGINEERING)
        with self._time_context("feature_engineering_time"):
            fe_cfg = self.config.feature_engineering
            self.logger.info("Starting feature engineering steps...")

            # Implement polynomial features manually with Polars
            if fe_cfg.enable_polynomial_features and fe_cfg.polynomial_degree > 1:
                self.logger.info(f"Generating polynomial features (degree={fe_cfg.polynomial_degree})")
                self._generate_polynomial_features(fe_cfg.polynomial_degree)

            # Dimensionality reduction
            if fe_cfg.enable_dimensionality_reduction:
                self.logger.info(f"Applying dimensionality reduction: {fe_cfg.dimensionality_reduction_method}")
                self._apply_dimensionality_reduction(fe_cfg.dimensionality_reduction_method)

            # Feature selection
            if fe_cfg.enable_feature_selection:
                self.logger.info(f"Selecting features by method: {fe_cfg.feature_selection_method}")
                self._apply_feature_selection(fe_cfg.feature_selection_method)

            # Custom feature generation
            if fe_cfg.enable_feature_generation:
                self.logger.info("Generating new features")
                self._generate_custom_features()

            self.logger.info("Feature engineering completed")

    def _generate_polynomial_features(self, degree: int):
        """Generate polynomial and interaction features using Polars expressions."""
        try:
            # We'll create combinations of features up to the given degree
            numeric_features = [
                col for col in self.feature_columns
                if pl.col(col).dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
            ]
            
            if not numeric_features:
                self.logger.warning("No numeric features found for polynomial feature generation")
                return
                
            self.logger.info(f"Generating polynomial features from {len(numeric_features)} numeric features")
            
            # Create combinations for each dataset
            for dataset_name in ['train_data', 'validation_data', 'test_data']:
                dataset = getattr(self, dataset_name)
                if dataset is None:
                    continue
                
                # Generate new columns for each power
                new_cols = []
                for i in range(2, degree + 1):
                    for col in numeric_features:
                        new_col_name = f"{col}_pow{i}"
                        new_cols.append(pl.col(col).pow(i).alias(new_col_name))
                
                # Generate interaction terms (2-way only to avoid explosion of features)
                for i, col1 in enumerate(numeric_features):
                    for col2 in numeric_features[i+1:]:
                        new_col_name = f"{col1}_x_{col2}"
                        new_cols.append((pl.col(col1) * pl.col(col2)).alias(new_col_name))
                
                # Apply transformations
                if new_cols:
                    setattr(self, dataset_name, dataset.with_columns(new_cols))
                    
                    # Update feature columns list after first dataset
                    if dataset_name == 'train_data':
                        self.feature_columns = dataset.columns
                        
            self.logger.info(f"Generated polynomial features. New feature count: {len(self.feature_columns)}")
        except Exception as e:
            self.logger.error(f"Error generating polynomial features: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def _apply_dimensionality_reduction(self, method: str):
        """Apply dimensionality reduction to the feature set."""
        try:
            if method.lower() == 'pca':
                from sklearn.decomposition import PCA
                
                # We need original numpy arrays for PCA
                X_train = self.train_data.select(self.feature_columns).to_numpy()
                
                # Determine number of components
                n_components = min(
                    len(self.feature_columns),
                    self.config.feature_engineering.max_components or len(self.feature_columns)
                )
                
                # Initialize and fit PCA
                pca = PCA(n_components=n_components)
                pca.fit(X_train)
                
                # Determine how many components to keep based on explained variance
                var_threshold = self.config.feature_engineering.variance_threshold or 0.95
                cum_var = np.cumsum(pca.explained_variance_ratio_)
                n_components_kept = np.argmax(cum_var >= var_threshold) + 1
                
                self.logger.info(
                    f"PCA: Using {n_components_kept} components to explain "
                    f"{cum_var[n_components_kept-1]*100:.2f}% of variance"
                )
                
                # Transform data and create new dataframes
                for dataset_name in ['train_data', 'validation_data', 'test_data']:
                    dataset = getattr(self, dataset_name)
                    if dataset is None:
                        continue
                    
                    # Get non-feature columns
                    non_feature_cols = [col for col in dataset.columns if col not in self.feature_columns]
                    non_feature_data = dataset.select(non_feature_cols)
                    
                    # Transform feature data
                    X = dataset.select(self.feature_columns).to_numpy()
                    X_reduced = pca.transform(X)[:, :n_components_kept]
                    
                    # Create new column names for PCA components
                    pca_cols = [f"pca_{i}" for i in range(n_components_kept)]
                    
                    # Create new DataFrame with PCA components and original non-feature columns
                    reduced_df = pl.DataFrame(X_reduced, schema=pca_cols)
                    setattr(self, dataset_name, pl.concat([non_feature_data, reduced_df], how="horizontal"))
                
                # Update feature columns list
                self.feature_columns = pca_cols
                self.logger.info(f"Dimensionality reduction complete: {len(self.feature_columns)} features retained")
                
            elif method.lower() == 'tsne' or method.lower() == 'umap':
                self.logger.warning(f"{method} dimensionality reduction is not implemented")
            else:
                self.logger.warning(f"Unsupported dimensionality reduction method: {method}")
        except Exception as e:
            self.logger.error(f"Error in dimensionality reduction: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def _apply_feature_selection(self, method: str):
        """Select most important features based on specified method."""
        try:
            # Extract feature data
            X_train = self.train_data.select(self.feature_columns).to_numpy()
            y_train = self.train_data.select(self.config.dataset.target_column).to_numpy().ravel()
            
            # Feature selection based on specified method
            if method.lower() == 'variance':
                from sklearn.feature_selection import VarianceThreshold
                
                threshold = self.config.feature_engineering.variance_threshold or 0.0
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(X_train)
                mask = selector.get_support()
                
                selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if mask[i]]
                
            elif method.lower() in ['f_classif', 'chi2', 'mutual_info_classif']:
                from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
                
                k = self.config.feature_engineering.num_features_to_select or min(20, len(self.feature_columns))
                
                if method.lower() == 'f_classif':
                    selector = SelectKBest(f_classif, k=k)
                elif method.lower() == 'chi2':
                    # Chi2 requires non-negative features
                    if (X_train < 0).any():
                        self.logger.warning("Chi2 requires non-negative features. Using f_classif instead.")
                        selector = SelectKBest(f_classif, k=k)
                    else:
                        selector = SelectKBest(chi2, k=k)
                else:  # mutual_info_classif
                    selector = SelectKBest(mutual_info_classif, k=k)
                
                selector.fit(X_train, y_train)
                mask = selector.get_support()
                
                selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if mask[i]]
                
            elif method.lower() in ['l1', 'lasso']:
                from sklearn.feature_selection import SelectFromModel
                from sklearn.linear_model import Lasso
                
                alpha = self.config.feature_engineering.alpha or 0.01
                selector = SelectFromModel(Lasso(alpha=alpha))
                selector.fit(X_train, y_train)
                mask = selector.get_support()
                
                selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if mask[i]]
                
            elif method.lower() == 'tree_importance':
                from sklearn.feature_selection import SelectFromModel
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if self.config.model_training.problem_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=self.config.dataset.random_seed)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=self.config.dataset.random_seed)
                
                selector = SelectFromModel(model, threshold='mean')
                selector.fit(X_train, y_train)
                mask = selector.get_support()
                
                selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if mask[i]]
                
            else:
                self.logger.warning(f"Unsupported feature selection method: {method}")
                return
            
            # Apply selection to all datasets
            if selected_features:
                self.logger.info(f"Selected {len(selected_features)} features out of {len(self.feature_columns)}")
                self.feature_columns = selected_features
                
                # Keep only selected feature columns and non-feature columns
                for dataset_name in ['train_data', 'validation_data', 'test_data']:
                    dataset = getattr(self, dataset_name)
                    if dataset is None:
                        continue
                    
                    # Get non-feature columns (ones not in the original feature set)
                    non_feature_cols = [col for col in dataset.columns if col not in self.feature_columns]
                    
                    # Create new DataFrame with only selected features and non-feature columns
                    setattr(self, dataset_name, dataset.select(non_feature_cols + selected_features))
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    def _generate_custom_features(self):
        """Generate custom features based on domain knowledge or feature engineering rules."""
        try:
            fe_cfg = self.config.feature_engineering
            
            # Date-based feature generation
            if fe_cfg.enable_date_features and self.config.dataset.time_column:
                time_col = self.config.dataset.time_column
                self.logger.info(f"Generating date features from {time_col}")
                
                for dataset_name in ['train_data', 'validation_data', 'test_data']:
                    dataset = getattr(self, dataset_name)
                    if dataset is None or time_col not in dataset.columns:
                        continue
                    
                    # Determine if we need to parse the date
                    if dataset.schema[time_col] not in [pl.Date, pl.Datetime]:
                        # Create a new DataFrame with parsed datetime
                        setattr(self, dataset_name, dataset.with_column(
                            pl.col(time_col).str.to_datetime().alias(time_col)))
                        dataset = getattr(self, dataset_name)
                    
                    # Generate date features
                    date_features = [
                        pl.col(time_col).dt.year().alias(f"{time_col}_year"),
                        pl.col(time_col).dt.month().alias(f"{time_col}_month"),
                        pl.col(time_col).dt.day().alias(f"{time_col}_day"),
                        pl.col(time_col).dt.weekday().alias(f"{time_col}_weekday"),
                        pl.col(time_col).dt.hour().alias(f"{time_col}_hour"),
                        pl.col(time_col).dt.quarter().alias(f"{time_col}_quarter"),
                        # Add more date features as needed
                    ]
                    
                    # Apply transformations
                    setattr(self, dataset_name, dataset.with_columns(date_features))
                    
                    # Update feature columns list after first dataset
                    if dataset_name == 'train_data':
                        new_cols = [expr.output_name() for expr in date_features]
                        self.feature_columns.extend(new_cols)
            
            # Text-based feature generation
            if fe_cfg.enable_text_features and fe_cfg.text_columns:
                self.logger.info(f"Text feature generation is not fully implemented")
                # In a production environment, you would add text processing here
                
            # Aggregation-based features
            if fe_cfg.enable_aggregation_features and fe_cfg.aggregation_columns:
                self.logger.info(f"Generating aggregation features")
                
                # Example: For each categorical column, generate stats of numeric columns
                cat_cols = fe_cfg.aggregation_columns
                num_cols = [col for col in self.feature_columns if col not in cat_cols]
                
                for dataset_name in ['train_data', 'validation_data', 'test_data']:
                    dataset = getattr(self, dataset_name)
                    if dataset is None:
                        continue
                    
                    new_features = []
                    for cat_col in cat_cols:
                        if cat_col not in dataset.columns:
                            continue
                            
                        # Get group aggregations (only for training set)
                        if dataset_name == 'train_data':
                            # Group by categorical column and calculate aggregations
                            aggs = {}
                            for num_col in num_cols[:5]:  # Limit to first 5 numeric cols to avoid explosion
                                aggs[f"{cat_col}_{num_col}_mean"] = pl.col(num_col).mean()
                                aggs[f"{cat_col}_{num_col}_std"] = pl.col(num_col).std()
                                
                            # Compute aggregations
                            agg_df = dataset.groupby(cat_col).agg(**aggs).collect()
                            
                            # Join back with original data to get aggregated features
                            setattr(self, dataset_name, dataset.join(agg_df, on=cat_col, how="left"))
                            
                            # Update feature columns after first dataset
                            self.feature_columns.extend(list(aggs.keys()))
            
            self.logger.info(f"Custom feature generation complete. Final feature count: {len(self.feature_columns)}")
        except Exception as e:
            self.logger.error(f"Error generating custom features: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())

    # ------------------
    # MODEL TRAINING
    # ------------------
    def train_model(self):
        """Train the model with the selected strategy (direct, CV, HPO)."""
        self._set_state(TrainingState.TRAINING)
        with self._time_context("training_time"):
            self.logger.info("Initializing and training the model")
            self._initialize_model()

            if self.config.hyperparameter_optimization.enabled:
                self._train_with_hpo()
            else:
                if self.config.model_training.enable_cross_validation:
                    self._train_with_cross_validation()
                else:
                    self._train_direct()

    def _initialize_model(self):
        """Create the model instance from config."""
        cfg = self.config.model_training
        try:
            if cfg.model_type == ModelType.SKLEARN:
                if not cfg.model_class:
                    raise ValueError("model_class must be specified for sklearn model.")
                mod_path, cls_name = cfg.model_class.rsplit('.', 1)
                mod = __import__(mod_path, fromlist=[cls_name])
                ModelClass = getattr(mod, cls_name)
                self.model = ModelClass(**cfg.hyperparameters)
                self.logger.info(f"Initialized {cls_name} model with params: {cfg.hyperparameters}")

            elif cfg.model_type == ModelType.XGBOOST:
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost is not installed")
                    
                params = cfg.hyperparameters.copy()
                if "objective" not in params:
                    params["objective"] = (
                        "binary:logistic" if cfg.problem_type == "classification" else "reg:squarederror"
                    )
                self.model = xgb.XGBModel(**params)
                self.logger.info(f"Initialized XGBoost model with params: {params}")

            elif cfg.model_type == ModelType.LIGHTGBM:
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM is not installed")
                    
                params = cfg.hyperparameters.copy()
                if "objective" not in params:
                    params["objective"] = (
                        "binary" if cfg.problem_type == "classification" else "regression"
                    )
                self.model = lgb.LGBMModel(**params)
                self.logger.info(f"Initialized LightGBM model with params: {params}")

            elif cfg.model_type == ModelType.CUSTOM:
                self.logger.warning("Custom model type specified, expecting an externally assigned self.model")
                if not self.model:
                    raise ValueError("For CUSTOM model type, self.model must be pre-assigned before training")
            else:
                raise ValueError(f"Unsupported ModelType: {cfg.model_type}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    def _train_direct(self):
        """Train model on the entire training set, optionally with early stopping if validation set is available."""
        X_train, y_train, X_val, y_val = self._prepare_data_for_training()
        self._fit_model(self.model, X_train, y_train, X_val, y_val)
        self.best_model = self.model
        self.logger.info("Model training completed (direct training)")

    def _train_with_cross_validation(self):
        """Perform cross-validation on the model, then refit on the entire training set."""
        cfg = self.config.model_training
        X_train, y_train, X_val, y_val = self._prepare_data_for_training()  # for final refit

        # Determine scoring
        scoring = "neg_mean_squared_error" if cfg.problem_type == "regression" else "accuracy"
        if cfg.cv_scoring:
            scoring = cfg.cv_scoring

        # Choose CV type
        if cfg.cv_strategy == "stratified" and cfg.problem_type == "classification":
            cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
        else:
            cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)

        self.logger.info(f"Starting {cfg.cv_folds}-fold cross-validation with scoring='{scoring}'")
        
        try:
            scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=cv, scoring=scoring, 
                n_jobs=cfg.n_jobs
            )
            
            self.logger.info(f"CV Scores: {scores}")
            self.logger.info(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std():.4f})")
            
            # Store CV results
            self.cv_results_ = {
                "scores": scores.tolist(),
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std())
            }

            # Refit on the entire training set
            self._fit_model(self.model, X_train, y_train, X_val, y_val)
            self.best_model = self.model
            self.logger.info("Model training with cross-validation completed")
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    def _train_with_hpo(self):
        """Train model with hyperparameter optimization (GridSearch, RandomSearch, or Bayesian)."""
        self._set_state(TrainingState.OPTIMIZING)
        with self._time_context("optimization_time"):
            hpo = self.config.hyperparameter_optimization
            try:
                if hpo.strategy == OptimizationStrategy.GRID_SEARCH:
                    self._hpo_grid_search()
                elif hpo.strategy == OptimizationStrategy.RANDOM_SEARCH:
                    self._hpo_random_search()
                elif hpo.strategy == OptimizationStrategy.BAYESIAN:
                    if OPTUNA_AVAILABLE:
                        self._hpo_optuna()
                    else:
                        self.logger.warning("Optuna not available, defaulting to RandomSearch")
                        self._hpo_random_search()
                else:
                    raise ValueError(f"Unsupported HPO strategy: {hpo.strategy}")
            except Exception as e:
                self.logger.error(f"Error during hyperparameter optimization: {e}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                raise

        self._set_state(TrainingState.TRAINING)

    # ------------------
    # HELPER TRAIN METHODS
    # ------------------
    def _prepare_data_for_training(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert Polars DataFrames to NumPy arrays for training."""
        ds = self.config.dataset
        X_train = self.train_data.select(self.feature_columns).to_numpy()
        y_train = self.train_data.select(ds.target_column).to_numpy().ravel()
        
        X_val, y_val = None, None
        if self.validation_data is not None:
            X_val = self.validation_data.select(self.feature_columns).to_numpy()
            y_val = self.validation_data.select(ds.target_column).to_numpy().ravel()
            
        return X_train, y_train, X_val, y_val

    def _fit_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """Fit the model with appropriate parameters and early stopping if possible."""
        cfg = self.config.model_training
        fit_params = {}
        
        # For XGBoost and LightGBM, set up early stopping
        if (cfg.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]) and X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            
            if cfg.early_stopping:
                fit_params["early_stopping_rounds"] = cfg.early_stopping_patience
                
            if cfg.num_epochs is not None:
                # unify param for both xgb and lgbm
                fit_params["n_estimators"] = cfg.num_epochs
                
            # Add verbosity control
            fit_params["verbose"] = True if self.config.debug_mode else False
        
        # Add sample weights if configured
        if cfg.sample_weights and self.config.dataset.weight_column:
            try:
                weights = self.train_data.select(self.config.dataset.weight_column).to_numpy().ravel()
                fit_params["sample_weight"] = weights
                self.logger.info("Using sample weights for training")
            except Exception as e:
                self.logger.warning(f"Could not use sample weights: {e}")

        # Fit the model with timing
        start = time.time()
        try:
            model.fit(X_train, y_train, **fit_params)
            end = time.time()
            self.logger.info(f"Model fit completed in {end - start:.2f} seconds")
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    # ------------------
    # GRID SEARCH / RANDOM SEARCH
    # ------------------
    def _hpo_grid_search(self):
        """Perform grid search for hyperparameter optimization."""
        X_train, y_train, _, _ = self._prepare_data_for_training()
        hpo = self.config.hyperparameter_optimization
        
        # Set up scoring function
        scoring = None if hpo.metric == "auto" else hpo.metric
        
        # Set up CV strategy
        if self.config.model_training.cv_strategy == "stratified" and self.config.model_training.problem_type == "classification":
            cv = StratifiedKFold(n_splits=hpo.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
        else:
            cv = KFold(n_splits=hpo.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
            
        # Create grid search object
        grid_cv = GridSearchCV(
            self.model,
            param_grid=hpo.param_grid,
            cv=cv,
            n_jobs=self.config.model_training.n_jobs,
            scoring=scoring,
            verbose=2 if self.config.debug_mode else 0,
            refit=True
        )
        
        # Fit grid search
        self.logger.info(f"Starting grid search with {hpo.cv_folds}-fold CV")
        start = time.time()
        grid_cv.fit(X_train, y_train)
        end = time.time()
        
        self.logger.info(f"Grid search completed in {end - start:.2f} seconds")
        self._assign_best_model(grid_cv)

    def _hpo_random_search(self):
        """Perform random search for hyperparameter optimization."""
        X_train, y_train, _, _ = self._prepare_data_for_training()
        hpo = self.config.hyperparameter_optimization
        
        # Set up scoring function
        scoring = None if hpo.metric == "auto" else hpo.metric
        
        # Set up CV strategy
        if self.config.model_training.cv_strategy == "stratified" and self.config.model_training.problem_type == "classification":
            cv = StratifiedKFold(n_splits=hpo.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
        else:
            cv = KFold(n_splits=hpo.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
            
        # Create random search object
        random_cv = RandomizedSearchCV(
            self.model,
            param_distributions=hpo.param_grid,
            n_iter=hpo.max_trials,
            cv=cv,
            n_jobs=self.config.model_training.n_jobs,
            scoring=scoring,
            verbose=2 if self.config.debug_mode else 0,
            random_state=self.config.dataset.random_seed,
            refit=True
        )
        
        # Fit random search
        self.logger.info(f"Starting random search with {hpo.max_trials} trials, {hpo.cv_folds}-fold CV")
        start = time.time()
        random_cv.fit(X_train, y_train)
        end = time.time()
        
        self.logger.info(f"Random search completed in {end - start:.2f} seconds")
        self._assign_best_model(random_cv)

    def _assign_best_model(self, search_cv):
        """Store the best model and parameters from a search CV object."""
        try:
            self.logger.info(f"Best parameters from search: {search_cv.best_params_}")
            self.logger.info(f"Best CV score: {search_cv.best_score_:.4f}")
            
            # Store results
            self.hpo_results = {
                "best_params": search_cv.best_params_,
                "best_score": float(search_cv.best_score_),
                "cv_results": {
                    "mean_test_score": search_cv.cv_results_["mean_test_score"].tolist(),
                    "std_test_score": search_cv.cv_results_["std_test_score"].tolist(),
                    "params": [str(p) for p in search_cv.cv_results_["params"]]
                }
            }
            
            # Use the best estimator
            self.best_model = search_cv.best_estimator_
            self.model = self.best_model
        except Exception as e:
            self.logger.error(f"Error assigning best model: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    # ------------------
    # BAYESIAN / OPTUNA
    # ------------------
    def _hpo_optuna(self):
        """Perform Bayesian optimization using Optuna."""
        import optuna  # guaranteed by the check above
        from optuna.samplers import TPESampler, RandomSampler
        from optuna.pruners import MedianPruner, HyperbandPruner
        
        hpo = self.config.hyperparameter_optimization
        X_train, y_train, X_val, y_val = self._prepare_data_for_training()
        mt = self.config.model_training.model_type

        def objective(trial):
            # Build param dictionary from param_grid
            trial_params = {}
            for param_name, space in hpo.param_grid.items():
                if isinstance(space, dict):
                    # handle low/high or choices
                    if "choices" in space:
                        trial_params[param_name] = trial.suggest_categorical(param_name, space["choices"])
                    else:
                        low = space.get("low")
                        high = space.get("high")
                        step = space.get("step", None)
                        log = space.get("log", False)
                        # Decide float vs. int
                        if isinstance(low, int) and isinstance(high, int) and not log:
                            trial_params[param_name] = trial.suggest_int(param_name, low, high, step=step or 1)
                        else:
                            trial_params[param_name] = trial.suggest_float(param_name, low, high, step=step, log=log)
                elif isinstance(space, list):
                    trial_params[param_name] = trial.suggest_categorical(param_name, space)
                else:
                    self.logger.warning(f"Unsupported param space for {param_name}: {space}")

            # Build new model with these params
            candidate_model = None
            if mt == ModelType.XGBOOST:
                candidate_model = xgb.XGBModel(**trial_params)
            elif mt == ModelType.LIGHTGBM:
                candidate_model = lgb.LGBMModel(**trial_params)
            elif mt == ModelType.SKLEARN:
                # Must have a valid model_class
                mod_path, cls_name = self.config.model_training.model_class.rsplit('.', 1)
                mod = __import__(mod_path, fromlist=[cls_name])
                CandidateClass = getattr(mod, cls_name)
                candidate_model = CandidateClass(**trial_params)
            else:
                raise ValueError(f"Optuna not supported for model type {mt} or custom logic needed.")

            # Optionally do early stopping if possible
            if X_val is not None and y_val is not None and self.config.model_training.early_stopping:
                candidate_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.config.model_training.early_stopping_patience,
                    verbose=False
                )
                # Evaluate performance on X_val
                pred_val = candidate_model.predict(X_val)
                score = self._compute_optuna_score(y_val, pred_val, hpo)
            else:
                # Or cross-validation
                score = self._cross_val_score(candidate_model, X_train, y_train, hpo)

            return score if hpo.direction == "maximize" else -score

        # Set up sampler/pruner
        sampler = None
        if hpo.sampler == "tpe":
            sampler = TPESampler(seed=self.config.dataset.random_seed)
        elif hpo.sampler == "random":
            sampler = RandomSampler(seed=self.config.dataset.random_seed)

        pruner = None
        if hpo.pruning:
            if hpo.pruner == "median":
                pruner = MedianPruner()
            elif hpo.pruner == "hyperband":
                pruner = HyperbandPruner()

        # Create and run study
        study = optuna.create_study(direction=hpo.direction, sampler=sampler, pruner=pruner)
        
        self.logger.info(
            f"Starting Optuna optimization with {hpo.max_trials} trials, "
            f"timeout={hpo.max_time_minutes} min, "
            f"direction={hpo.direction}"
        )
        
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=hpo.max_trials,
            timeout=(hpo.max_time_minutes * 60 if hpo.max_time_minutes else None),
            n_jobs=hpo.n_parallel_trials or 1,
            show_progress_bar=True
        )
        end_time = time.time()

        self.logger.info(f"Optuna optimization completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Best trial: {study.best_trial.number}, value: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")

        # Store results
        self.hpo_results = {
            "best_params": study.best_params,
            "best_score": float(study.best_value),
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "optimization_time": end_time - start_time
        }

        # Build final best model, fit on entire training set
        final_params = study.best_params
        if mt == ModelType.XGBOOST:
            self.best_model = xgb.XGBModel(**final_params)
        elif mt == ModelType.LIGHTGBM:
            self.best_model = lgb.LGBMModel(**final_params)
        else:
            mod_path, cls_name = self.config.model_training.model_class.rsplit('.', 1)
            mod = __import__(mod_path, fromlist=[cls_name])
            CandidateClass = getattr(mod, cls_name)
            self.best_model = CandidateClass(**final_params)

        # Fit the best model on all data
        self._fit_model(self.best_model, X_train, y_train, X_val, y_val)
        self.model = self.best_model

    def _compute_optuna_score(self, y_true, y_pred, hpo_cfg):
        """Compute a simple score for the objective function in Optuna."""
        if self.config.model_training.problem_type == "regression":
            if hpo_cfg.metric == "rmse":
                return -np.sqrt(mean_squared_error(y_true, y_pred))
            elif hpo_cfg.metric == "mae":
                return -np.mean(np.abs(y_true - y_pred))
            elif hpo_cfg.metric == "r2":
                return r2_score(y_true, y_pred)
            else:
                return -mean_squared_error(y_true, y_pred)  # negative MSE to maximize
        else:
            # Classification metrics
            if hpo_cfg.metric == "accuracy" or hpo_cfg.metric == "auto":
                return accuracy_score(y_true, y_pred)
            elif hpo_cfg.metric == "f1":
                return f1_score(y_true, y_pred, average="weighted")
            elif hpo_cfg.metric == "precision":
                return precision_score(y_true, y_pred, average="weighted")
            elif hpo_cfg.metric == "recall":
                return recall_score(y_true, y_pred, average="weighted")
            elif hpo_cfg.metric == "auc" and len(np.unique(y_true)) == 2:
                # Only for binary classification
                return roc_auc_score(y_true, y_pred)
            else:
                return accuracy_score(y_true, y_pred)

    def _cross_val_score(self, model, X, y, hpo_cfg):
        """Compute a cross-validation score with config-based folds."""
        scoring = None if hpo_cfg.metric == "auto" else hpo_cfg.metric
        
        if self.config.model_training.cv_strategy == "stratified" and self.config.model_training.problem_type == "classification":
            cv = StratifiedKFold(n_splits=hpo_cfg.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
        else:
            cv = KFold(n_splits=hpo_cfg.cv_folds, shuffle=True, random_state=self.config.dataset.random_seed)
            
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            return scores.mean()
        except Exception as e:
            self.logger.warning(f"Error in cross-validation: {e}. Returning -inf.")
            return float("-inf")

    # ------------------
    # EVALUATION
    # ------------------
    def evaluate_model(self):
        """Evaluate the trained model on train, validation and test sets."""
        self._set_state(TrainingState.EVALUATING)
        with self._time_context("evaluation_time"):
            self.logger.info("Evaluating the trained model")
            
            if not self.best_model:
                self.logger.warning("No best_model found; using self.model.")
                model = self.model
            else:
                model = self.best_model

            eval_cfg = self.config.model_evaluation
            results = {}

            if eval_cfg.evaluate_on_train:
                results["train"] = self._evaluate_dataset(model, self.train_data, "train")

            if eval_cfg.evaluate_on_validation and self.validation_data is not None:
                results["validation"] = self._evaluate_dataset(model, self.validation_data, "validation")

            if eval_cfg.evaluate_on_test and self.test_data is not None:
                results["test"] = self._evaluate_dataset(model, self.test_data, "test")

            self.evaluation_results = results
            self._log_evaluation_summary()

            # Additional evaluation for classification
            if self.config.model_training.problem_type == "classification":
                self._additional_classification_artifacts(model)

            # Feature importance
            if self.config.model_training.compute_feature_importance:
                self._compute_feature_importance()

            self.logger.info("Model evaluation completed")

    def _evaluate_dataset(self, model, dataset: pl.DataFrame, ds_name: str):
        """Compute predictions and standard metrics for a dataset."""
        ds_cfg = self.config.dataset
        
        # Extract features and target
        X = dataset.select(self.feature_columns).to_numpy()
        y_true = dataset.select(ds_cfg.target_column).to_numpy().ravel()

        # Classification vs regression
        if self.config.model_training.problem_type == "classification":
            y_pred = model.predict(X)
            y_prob = None
            
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob_raw = model.predict_proba(X)
                    if y_prob_raw.shape[1] == 2:
                        y_prob = y_prob_raw[:, 1]
                    else:
                        y_prob = y_prob_raw
                except Exception as e:
                    self.logger.warning(f"Could not compute prediction probabilities: {e}")
                    
            metrics = self._compute_classification_metrics(y_true, y_pred, y_prob)
        else:
            # regression
            y_pred = model.predict(X)
            metrics = self._compute_regression_metrics(y_true, y_pred)

        self.logger.info(f"Evaluation on {ds_name} - metrics: {metrics}")
        return {"metrics": metrics, "predictions": y_pred}

    def _compute_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Compute classification metrics like accuracy, precision, recall, F1, possibly AUC."""
        eval_cfg = self.config.model_evaluation
        avg_method = eval_cfg.average_method or "binary"
        
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average=avg_method, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=avg_method, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg_method, zero_division=0))

        # ROC AUC if binary or if you handle multi-class carefully
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                self.logger.warning(f"Could not compute ROC AUC: {e}")
                
        return metrics

    def _compute_regression_metrics(self, y_true, y_pred):
        """Compute regression metrics."""
        mse_val = mean_squared_error(y_true, y_pred)
        return {
            "mse": float(mse_val),
            "rmse": float(np.sqrt(mse_val)),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(np.mean(np.abs(y_true - y_pred)))
        }

    def _log_evaluation_summary(self):
        """Log a summary of evaluation metrics for all datasets."""
        self.logger.info("==== Evaluation Summary ====")
        for ds_name, res in self.evaluation_results.items():
            self.logger.info(f"--- {ds_name.upper()} ---")
            for k, v in res["metrics"].items():
                self.logger.info(f"{k}: {v:.4f}")
        self.logger.info("===========================")

    def _additional_classification_artifacts(self, model):
        """Compute confusion matrix, classification report for classification models."""
        eval_cfg = self.config.model_evaluation

        # Confusion matrix on the validation or test set
        if eval_cfg.confusion_matrix:
            # prioritize validation, then test, else train
            ds_for_cm = None
            preds = None
            if (self.validation_data is not None) and ("validation" in self.evaluation_results):
                ds_for_cm = self.validation_data
                preds = self.evaluation_results["validation"]["predictions"]
            elif (self.test_data is not None) and ("test" in self.evaluation_results):
                ds_for_cm = self.test_data
                preds = self.evaluation_results["test"]["predictions"]
            elif "train" in self.evaluation_results:
                ds_for_cm = self.train_data
                preds = self.evaluation_results["train"]["predictions"]

            if ds_for_cm is not None:
                y_true = ds_for_cm.select(self.config.dataset.target_column).to_numpy().ravel()
                cm = confusion_matrix(y_true, preds)
                self.logger.info(f"Confusion Matrix:\n{cm}")
                
                # Store confusion matrix in results
                self.evaluation_results["confusion_matrix"] = cm.tolist()

        # Classification report
        if eval_cfg.classification_report and hasattr(sklearn.metrics, "classification_report"):
            try:
                from sklearn.metrics import classification_report
                # same logic picking which dataset for the report
                if (self.validation_data is not None) and ("validation" in self.evaluation_results):
                    y_true = self.validation_data.select(self.config.dataset.target_column).to_numpy().ravel()
                    y_pred = self.evaluation_results["validation"]["predictions"]
                else:
                    y_true = self.train_data.select(self.config.dataset.target_column).to_numpy().ravel()
                    y_pred = self.evaluation_results["train"]["predictions"]
                    
                rep = classification_report(y_true, y_pred)
                self.logger.info(f"Classification Report:\n{rep}")
                
                # Also get report as dict for JSON serialization
                rep_dict = classification_report(y_true, y_pred, output_dict=True)
                self.evaluation_results["classification_report"] = rep_dict
            except Exception as e:
                self.logger.warning(f"Could not generate classification report: {e}")

    def _compute_feature_importance(self):
        """Compute and log feature importance from the model if available."""
        model = self.best_model if self.best_model else self.model
        
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                # Create a Polars DataFrame for feature importances
                importance_data = {
                    "feature": self.feature_columns,
                    "importance": importances
                }
                fi = pl.DataFrame(importance_data)
                fi = fi.sort("importance", descending=True)
                self.feature_importance_ = fi
                
                # Log top features
                top_features = fi.head(20)
                self.logger.info(f"Top feature importances:\n{top_features}")
                
                # Store as serializable dict
                self.feature_importance_dict = {
                    "feature": fi["feature"].to_list(),
                    "importance": fi["importance"].to_list()
                }
            else:
                self.logger.info("Feature importances not available for this model")
                
                # Try SHAP for feature importance if requested
                if self.config.model_evaluation.compute_shap and self.config.debug_mode:
                    self.logger.info("SHAP feature importance not implemented in this version")
        except Exception as e:
            self.logger.warning(f"Error computing feature importance: {e}")

    # ------------------
    # REGISTRY / SAVING
    # ------------------
    def save_model(self):
        """Save the trained model and associated artifacts to the registry."""
        self._set_state(TrainingState.SAVING)
        self.logger.info("Saving model/artifacts to registry...")
        
        try:
            reg_cfg = self.config.model_registry
            model_name = reg_cfg.model_name
            version = self._resolve_model_version()

            model_dir = os.path.join(reg_cfg.registry_path, model_name, version)
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.best_model or self.model, f)
            self.logger.info(f"Model saved: {model_path}")

            # Save preprocessor
            if reg_cfg.save_preprocessor and self.preprocessor:
                preproc_path = os.path.join(model_dir, "preprocessor.pkl")
                with open(preproc_path, "wb") as f:
                    pickle.dump(self.preprocessor, f)
                self.logger.info(f"Preprocessor saved: {preproc_path}")

            # Save feature columns
            if reg_cfg.save_feature_names:
                fc_path = os.path.join(model_dir, "feature_columns.json")
                with open(fc_path, "w") as f:
                    json.dump(self.feature_columns, f)
                self.logger.info(f"Feature columns saved: {fc_path}")

            # Save metrics
            if reg_cfg.save_metrics and self.evaluation_results:
                metrics_file = os.path.join(model_dir, "metrics.json")
                # Clean up results for JSON serialization
                results_cleaned = {}
                for ds_name, ds_res in self.evaluation_results.items():
                    if ds_name in ["confusion_matrix", "classification_report"]:
                        results_cleaned[ds_name] = ds_res
                    else:
                        met = ds_res.get("metrics", {})
                        results_cleaned[ds_name] = {"metrics": met}
                with open(metrics_file, "w") as f:
                    json.dump(results_cleaned, f, indent=2)
                self.logger.info(f"Metrics saved: {metrics_file}")

            # Save config
            if reg_cfg.save_hyperparameters:
                cfg_path = os.path.join(model_dir, "config.json")
                with open(cfg_path, "w") as f:
                    json.dump(self.config.to_dict(), f, indent=2, default=str)
                self.logger.info(f"Config saved: {cfg_path}")

            # Save feature importance
            if reg_cfg.save_feature_importance and hasattr(self, "feature_importance_dict"):
                fi_path = os.path.join(model_dir, "feature_importance.json")
                with open(fi_path, "w") as f:
                    json.dump(self.feature_importance_dict, f, indent=2)
                self.logger.info(f"Feature importance saved: {fi_path}")

            # Save HPO results if available
            if hasattr(self, "hpo_results") and self.hpo_results:
                hpo_path = os.path.join(model_dir, "hpo_results.json")
                with open(hpo_path, "w") as f:
                    json.dump(self.hpo_results, f, indent=2)
                self.logger.info(f"HPO results saved: {hpo_path}")

            # Save resource usage if monitored
            if self.resource_usage:
                resource_path = os.path.join(model_dir, "resource_usage.json")
                with open(resource_path, "w") as f:
                    # Sample resource usage to prevent huge files
                    if len(self.resource_usage) > 1000:
                        step = len(self.resource_usage) // 1000
                        sampled = self.resource_usage[::step]
                    else:
                        sampled = self.resource_usage
                    json.dump(sampled, f)
                self.logger.info(f"Resource usage saved: {resource_path}")

            # Metadata
            metadata = {
                "model_name": model_name,
                "model_version": version,
                "description": reg_cfg.description,
                "tags": reg_cfg.tags,
                "creation_time": datetime.now().isoformat(),
                "framework": self.config.model_training.model_type.name,
                "python_version": sys.version,
                "training_id": self.training_id,
                "runtime_metrics": self.runtime_metrics
            }
            meta_path = os.path.join(model_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Metadata saved: {meta_path}")

            # Optionally auto-deploy
            if reg_cfg.auto_deploy:
                self._deploy_model(model_dir)

            # Lifecycle management (archiving older versions)
            if reg_cfg.enable_lifecycle_management:
                self._manage_lifecycle(model_name)
                
            self.logger.info(f"Model and artifacts successfully saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            raise

    def _resolve_model_version(self) -> str:
        """Determine the appropriate model version based on configuration."""
        reg_cfg = self.config.model_registry
        if not reg_cfg.auto_version:
            return reg_cfg.model_version
            
        strategy = reg_cfg.version_strategy
        base_dir = os.path.join(reg_cfg.registry_path, reg_cfg.model_name)
        os.makedirs(base_dir, exist_ok=True)

        if strategy == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        elif strategy == "incremental":
            existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not existing:
                return "1"
            try:
                numeric = [int(x) for x in existing if x.isdigit()]
                return str(max(numeric) + 1 if numeric else 1)
            except Exception:
                return "1"
        elif strategy == "semver":
            # parse semver from model_version
            ver = reg_cfg.model_version
            try:
                major, minor, patch = map(int, ver.split("."))
                patch += 1
                return f"{major}.{minor}.{patch}"
            except Exception:
                return ver
        else:
            return reg_cfg.model_version

    def _deploy_model(self, model_dir: str):
        """Deploy the model to the specified target environment."""
        self.logger.info(f"Auto-deployment to: {self.config.model_registry.deployment_target}")
        
        # This is a placeholder. In a production environment, you would add code to:
        # 1. Package the model (Docker, ZIP, etc.)
        # 2. Upload to deployment target (S3, GCS, Azure Blob, etc.)
        # 3. Trigger deployment process (API call, message queue, etc.)
        # 4. Verify deployment status
        
        deployment_target = self.config.model_registry.deployment_target
        if deployment_target == "local":
            # Example local deployment - create a symlink in a deployment directory
            deploy_dir = os.path.join(self.config.model_registry.registry_path, "deployed")
            os.makedirs(deploy_dir, exist_ok=True)
            
            model_name = self.config.model_registry.model_name
            deploy_path = os.path.join(deploy_dir, model_name)
            
            # Remove existing symlink if it exists
            if os.path.islink(deploy_path):
                os.unlink(deploy_path)
                
            # Create new symlink
            try:
                os.symlink(model_dir, deploy_path, target_is_directory=True)
                self.logger.info(f"Model deployed (symlink) to {deploy_path}")
            except Exception as e:
                self.logger.error(f"Failed to create deployment symlink: {e}")
        else:
            self.logger.info(f"Deployment to {deployment_target} not implemented")

    def _manage_lifecycle(self, model_name: str):
        """Manage model lifecycle by archiving old versions."""
        reg_cfg = self.config.model_registry
        base_dir = os.path.join(reg_cfg.registry_path, model_name)
        if not os.path.exists(base_dir):
            return

        if reg_cfg.archive_old_versions:
            versions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            
            # Create a list of (version, timestamp) tuples
            version_times = []
            for v in versions:
                meta_path = os.path.join(base_dir, v, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        try:
                            meta = json.load(f)
                            ctime = datetime.fromisoformat(meta["creation_time"])
                            version_times.append((v, ctime))
                        except Exception:
                            # Fall back to file modification time
                            st_mtime = os.path.getmtime(os.path.join(base_dir, v))
                            version_times.append((v, datetime.fromtimestamp(st_mtime)))
                else:
                    # No metadata, use modification time
                    st_mtime = os.path.getmtime(os.path.join(base_dir, v))
                    version_times.append((v, datetime.fromtimestamp(st_mtime)))
                    
            # Sort by timestamp (newest first)
            version_times.sort(key=lambda x: x[1], reverse=True)

            # Archive old versions
            if len(version_times) > reg_cfg.max_versions_to_keep:
                to_archive = version_times[reg_cfg.max_versions_to_keep:]
                arch_dir = os.path.join(reg_cfg.registry_path, "archived", model_name)
                os.makedirs(arch_dir, exist_ok=True)
                
                for v, _ in to_archive:
                    src = os.path.join(base_dir, v)
                    dst = os.path.join(arch_dir, v)
                    try:
                        shutil.move(src, dst)
                        self.logger.info(f"Archived old model version: {v}")
                    except Exception as e:
                        self.logger.warning(f"Could not archive version {v}: {str(e)}")

    # ------------------
    # MAIN RUN & CLEANUP
    # ------------------
    def run(self) -> Dict[str, Any]:
        """Execute the complete training pipeline from data loading to model saving."""
        self.logger.info("Starting the training pipeline")
        self.start_time = datetime.now()
        
        if self.config.enable_monitoring:
            self.start_monitoring()
            
        try:
            self.load_data()
            self.engineer_features()
            self.train_model()
            self.evaluate_model()
            self.save_model()

            self._set_state(TrainingState.COMPLETED)
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            self.runtime_metrics["total_time"] = duration
            self.logger.info(f"Training pipeline completed in {duration:.2f} seconds")
            
            model_path = os.path.join(
                self.config.model_registry.registry_path,
                self.config.model_registry.model_name,
                self._resolve_model_version()
            )
            
            return {
                "success": True,
                "training_id": self.training_id,
                "model_path": model_path,
                "metrics": self.evaluation_results,
                "runtime_metrics": self.runtime_metrics
            }
        except Exception as ex:
            self.logger.error(f"Error in training pipeline: {ex}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
                
            self._set_state(TrainingState.FAILED)
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            self.runtime_metrics["total_time"] = duration
            
            return {
                "success": False,
                "training_id": self.training_id,
                "error": str(ex),
                "runtime_metrics": self.runtime_metrics
            }
        finally:
            if self.config.enable_monitoring:
                self.stop_monitoring()
            self.cleanup()

    def cleanup(self):
        """Clean up resources, flush logs, etc."""
        # Free up memory
        if self.config.debug_mode:
            self.logger.info("Running cleanup...")
            
        if self.config.memory_optimization.free_memory_on_cleanup:
            del_attrs = [
                "train_data", "validation_data", "test_data",
                "model", "best_model", "preprocessor"
            ]
            for attr in del_attrs:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    setattr(self, attr, None)
                    
            # Force garbage collection
            gc.collect()
            self.logger.debug("Memory cleanup completed")

    def inference(self, data, output_type: str = "dict"):
        """
        Perform inference with the trained model.
        
        Args:
            data: A DataFrame, array, or dictionary of input data
            output_type: Format of output ("dict", "array", "dataframe")
            
        Returns:
            Predictions in the requested format
        """
        if not (self.model or self.best_model):
            raise NotFittedError("Model is not trained yet.")
            
        model = self.best_model if self.best_model else self.model
        
        # Convert input to appropriate format
        if isinstance(data, pl.DataFrame):
            # Make sure we have all required feature columns
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required feature columns: {missing_cols}")
                
            # Apply preprocessor if available
            if self.preprocessor:
                processed_data = self.preprocessor.transform(data)
                X = processed_data.select(self.feature_columns).to_numpy()
            else:
                X = data.select(self.feature_columns).to_numpy()
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            df = pl.DataFrame([data])
            return self.inference(df, output_type)
        elif isinstance(data, (list, np.ndarray)):
            # Assume data is already preprocessed and in correct order
            X = np.array(data)
        else:
            raise ValueError(f"Unsupported data type for inference: {type(data)}")
            
        # Generate predictions
        if self.config.model_training.problem_type == "classification":
            y_pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X)
                    result = {
                        "predictions": y_pred,
                        "probabilities": y_prob
                    }
                except Exception:
                    result = {"predictions": y_pred}
            else:
                result = {"predictions": y_pred}
        else:
            # Regression
            y_pred = model.predict(X)
            result = {"predictions": y_pred}
            
        # Format output
        if output_type == "dict":
            # Convert to standard Python types for JSON serialization
            if isinstance(result["predictions"], np.ndarray):
                result["predictions"] = result["predictions"].tolist()
            if "probabilities" in result and isinstance(result["probabilities"], np.ndarray):
                result["probabilities"] = result["probabilities"].tolist()
            return result
        elif output_type == "array":
            return y_pred
        elif output_type == "dataframe":
            if isinstance(data, pl.DataFrame):
                if "probabilities" in result and result["probabilities"].shape[1] > 1:
                    # Add class probabilities as columns
                    probs = result["probabilities"]
                    prob_cols = {f"probability_{i}": probs[:, i] for i in range(probs.shape[1])}
                    return data.with_columns([
                        pl.lit(result["predictions"]).alias("prediction"),
                        *[pl.lit(v).alias(k) for k, v in prob_cols.items()]
                    ])
                else:
                    return data.with_column(pl.lit(result["predictions"]).alias("prediction"))
            else:
                # Create a new DataFrame
                if "probabilities" in result and len(result["probabilities"].shape) > 1:
                    probs = result["probabilities"]
                    prob_cols = {f"probability_{i}": probs[:, i] for i in range(probs.shape[1])}
                    return pl.DataFrame({
                        "prediction": result["predictions"],
                        **prob_cols
                    })
                else:
                    return pl.DataFrame({"prediction": result["predictions"]})
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

    def get_memory_usage(self, as_dict=False):
        """Return current memory usage of the training engine components."""
        if not PSUTIL_AVAILABLE:
            return None
            
        try:
            # Get current process
            process = psutil.Process(os.getpid())
            
            # Get memory info
            mem_info = process.memory_info()
            
            # Estimate size of key objects
            def obj_size(obj):
                if obj is None:
                    return 0
                try:
                    # For DataFrames
                    if hasattr(obj, 'memory_usage'):
                        try:
                            # Polars memory_usage handles deep=True differently
                            return obj.estimated_size() if hasattr(obj, 'estimated_size') else obj.memory_usage(deep=True).sum()
                        except Exception:
                            pass
                    # For numpy arrays
                    if hasattr(obj, 'nbytes'):
                        return obj.nbytes
                    # For other objects
                    return sys.getsizeof(obj)
                except Exception:
                    return 0
            
            # Calculate memory for key components
            train_data_mem = obj_size(self.train_data)
            valid_data_mem = obj_size(self.validation_data)
            test_data_mem = obj_size(self.test_data)
            model_mem = obj_size(self.model)
            best_model_mem = obj_size(self.best_model)
            
            # Build memory report
            memory_info = {
                "process_rss_bytes": mem_info.rss,
                "process_rss_mb": mem_info.rss / (1024 * 1024),
                "process_vms_mb": mem_info.vms / (1024 * 1024),
                "train_data_mb": train_data_mem / (1024 * 1024),
                "validation_data_mb": valid_data_mem / (1024 * 1024),
                "test_data_mb": test_data_mem / (1024 * 1024),
                "model_mb": model_mem / (1024 * 1024),
                "best_model_mb": best_model_mem / (1024 * 1024),
                "total_tracked_mb": (train_data_mem + valid_data_mem + test_data_mem + model_mem + best_model_mem) / (1024 * 1024),
                "system_memory_percent": psutil.virtual_memory().percent
            }
            
            if not as_dict:
                # Format as string for logging
                mem_str = (
                    f"Memory usage (MB): Process={memory_info['process_rss_mb']:.1f}, "
                    f"Train data={memory_info['train_data_mb']:.1f}, "
                    f"Val data={memory_info['validation_data_mb']:.1f}, "
                    f"Test data={memory_info['test_data_mb']:.1f}, "
                    f"Models={memory_info['model_mb']:.1f}/{memory_info['best_model_mb']:.1f}, "
                    f"System {memory_info['system_memory_percent']}% used"
                )
                return mem_str
            else:
                return memory_info
        except Exception as e:
            self.logger.warning(f"Error getting memory usage: {e}")
            return None

    def get_runtime_metrics(self):
        """Return runtime metrics for the training run."""
        metrics = self.runtime_metrics.copy()
        
        # Add current timestamp and elapsed time if still running
        if self.state != TrainingState.COMPLETED and self.state != TrainingState.FAILED:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            metrics["elapsed_time"] = elapsed
            metrics["total_time"] = elapsed
            
        # Add peak memory usage
        if PSUTIL_AVAILABLE:
            try:
                memory_usage = self.get_memory_usage(as_dict=True)
                if memory_usage:
                    metrics["peak_memory_usage_mb"] = max(
                        metrics.get("peak_memory_usage_mb", 0),
                        memory_usage["process_rss_mb"]
                    )
            except Exception:
                pass
                
        return metrics

    def load_from_registry(self, model_name: str, version: str = None):
        """
        Load a trained model from the registry.
        
        Args:
            model_name: Name of the model to load
            version: Version to load, or None for latest
            
        Returns:
            Boolean indicating success
        """
        reg_cfg = self.config.model_registry
        base_dir = os.path.join(reg_cfg.registry_path, model_name)
        
        if not os.path.exists(base_dir):
            self.logger.error(f"Model {model_name} not found in registry")
            return False
            
        # Determine version to load
        if version is None:
            # Find latest version by creation time in metadata
            versions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not versions:
                self.logger.error(f"No versions found for model {model_name}")
                return False
                
            latest_version = None
            latest_time = None
            
            for v in versions:
                meta_path = os.path.join(base_dir, v, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            created = datetime.fromisoformat(meta["creation_time"])
                            if latest_time is None or created > latest_time:
                                latest_time = created
                                latest_version = v
                    except Exception:
                        pass
                        
            if latest_version is None:
                # Fall back to using the directory with newest modification time
                version_times = [(v, os.path.getmtime(os.path.join(base_dir, v))) for v in versions]
                version_times.sort(key=lambda x: x[1], reverse=True)
                latest_version = version_times[0][0]
                
            version = latest_version
            
        model_dir = os.path.join(base_dir, version)
        if not os.path.exists(model_dir):
            self.logger.error(f"Version {version} not found for model {model_name}")
            return False
            
        # Load the model
        try:
            model_path = os.path.join(model_dir, "model.pkl")
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found at {model_path}")
                return False
                
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
                
            self.best_model = self.model
            
            # Load preprocessor if available
            preproc_path = os.path.join(model_dir, "preprocessor.pkl")
            if os.path.exists(preproc_path):
                with open(preproc_path, "rb") as f:
                    self.preprocessor = pickle.load(f)
            
            # Load feature columns
            fc_path = os.path.join(model_dir, "feature_columns.json")
            if os.path.exists(fc_path):
                with open(fc_path, "r") as f:
                    self.feature_columns = json.load(f)
            
            self.logger.info(f"Loaded model {model_name} version {version} from registry")
            
            # Also load metadata
            meta_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.model_metadata = json.load(f)
                    
            # Set state to reflect loaded model
            self._set_state(TrainingState.LOADED)
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from registry: {e}")
            if self.config.debug_mode:
                self.logger.error(traceback.format_exc())
            return False
    # ------------------
    # MODEL COMPARISON & ANALYSIS
    # ------------------
    def compare_model_versions(self, model_name: str, versions: List[str] = None):
        """
        Compare metrics from different versions of the same model.
        
        Args:
            model_name: Name of the model to compare versions
            versions: List of versions to compare, or None for all versions
            
        Returns:
            Dictionary with comparison results
        """
        reg_cfg = self.config.model_registry
        base_dir = os.path.join(reg_cfg.registry_path, model_name)
        
        if not os.path.exists(base_dir):
            self.logger.error(f"Model {model_name} not found in registry")
            return None
        
        # Get available versions
        available_versions = [d for d in os.listdir(base_dir) 
                              if os.path.isdir(os.path.join(base_dir, d))]
        
        if not available_versions:
            self.logger.error(f"No versions found for model {model_name}")
            return None
            
        # Filter to requested versions if specified
        if versions:
            found_versions = [v for v in versions if v in available_versions]
            if not found_versions:
                self.logger.error(f"None of the requested versions {versions} found")
                return None
            compare_versions = found_versions
        else:
            compare_versions = available_versions
            
        # Load metrics for each version
        version_metrics = {}
        for version in compare_versions:
            metrics_path = os.path.join(base_dir, version, "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    version_metrics[version] = metrics
                except Exception as e:
                    self.logger.warning(f"Could not load metrics for version {version}: {e}")
        
        if not version_metrics:
            self.logger.error("No metrics found for any of the versions")
            return None
            
        # Create comparison DataFrame
        comparison = {}
        
        # Collect all metric names across all versions
        all_metrics = set()
        for ver_data in version_metrics.values():
            for dataset in ["validation", "test"]:
                if dataset in ver_data and "metrics" in ver_data[dataset]:
                    all_metrics.update(ver_data[dataset]["metrics"].keys())
        
        # Prioritize validation, then test metrics
        for metric in all_metrics:
            metric_values = {}
            for version, ver_data in version_metrics.items():
                # Try validation first, then test
                for dataset in ["validation", "test"]:
                    if (dataset in ver_data and "metrics" in ver_data[dataset] and 
                            metric in ver_data[dataset]["metrics"]):
                        metric_values[version] = ver_data[dataset]["metrics"][metric]
                        break
            comparison[metric] = metric_values
            
        # Determine best version for each metric
        best_versions = {}
        for metric, values in comparison.items():
            if not values:
                continue
                
            # Determine if higher is better (most metrics) or lower is better (error metrics)
            lower_is_better = any(m in metric.lower() for m in ["error", "loss", "mse", "rmse", "mae"])
            
            # Find best version
            best_value = None
            best_ver = None
            for ver, val in values.items():
                if best_value is None:
                    best_value = val
                    best_ver = ver
                elif lower_is_better and val < best_value:
                    best_value = val
                    best_ver = ver
                elif not lower_is_better and val > best_value:
                    best_value = val
                    best_ver = ver
                    
            best_versions[metric] = {
                "version": best_ver,
                "value": best_value
            }
            
        summary = {
            "comparison": comparison,
            "best_versions": best_versions,
            "available_versions": available_versions,
            "compared_versions": compare_versions
        }
        
        return summary

    def generate_training_report(self, include_plots: bool = False):
        """
        Generate a comprehensive training report with all relevant information.
        
        Args:
            include_plots: Whether to generate and include plots in the report
            
        Returns:
            Dictionary with full training report
        """
        report = {
            "training_id": self.training_id,
            "model_name": self.config.model_registry.model_name,
            "status": self.state.name,
            "timestamp": datetime.now().isoformat(),
            "training_time": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (self.end_time - self.start_time).total_seconds() 
                                    if self.end_time and self.start_time else None
            },
            "configuration": self.config.to_dict(),
            "runtime_metrics": self.runtime_metrics,
            "evaluation_results": self.evaluation_results,
            "feature_information": {
                "feature_count": len(self.feature_columns),
                "features": self.feature_columns
            }
        }
        
        # Add feature importance if available
        if hasattr(self, "feature_importance_dict"):
            report["feature_importance"] = self.feature_importance_dict
            
        # Add HPO results if available
        if hasattr(self, "hpo_results"):
            report["hyperparameter_optimization"] = self.hpo_results
            
        # Add CV results if available
        if hasattr(self, "cv_results_"):
            report["cross_validation"] = self.cv_results_
            
        # Add dataset statistics
        dataset_stats = {}
        for ds_name, ds in [
            ("train", self.train_data),
            ("validation", self.validation_data),
            ("test", self.test_data)
        ]:
            if ds is not None:
                dataset_stats[ds_name] = {
                    "row_count": len(ds),
                    "column_count": len(ds.columns),
                    "memory_usage_mb": ds.estimated_size() / (1024 * 1024) if hasattr(ds, 'estimated_size') else None
                }
                
                # Add basic statistics if dataset is small enough
                if len(ds) < 100000 and self.config.debug_mode:
                    try:
                        # Get basic statistics using Polars describe method
                        stats_df = ds.describe()
                        # Convert to dict for JSON serialization
                        stats_dict = {}
                        for col in stats_df.columns:
                            if col == "statistic":
                                continue
                            stat_values = {}
                            for i, stat_name in enumerate(stats_df["statistic"]):
                                stat_values[stat_name] = stats_df[col][i]
                            stats_dict[col] = stat_values
                        dataset_stats[ds_name]["statistics"] = stats_dict
                    except Exception as e:
                        self.logger.warning(f"Could not compute statistics for {ds_name} dataset: {e}")
        
        report["dataset_statistics"] = dataset_stats
        
        # Add resource usage information
        if self.resource_usage:
            # Sample resource usage to keep report size reasonable
            if len(self.resource_usage) > 100:
                step = len(self.resource_usage) // 100
                sampled_usage = self.resource_usage[::step]
            else:
                sampled_usage = self.resource_usage
                
            report["resource_usage"] = {
                "samples": sampled_usage,
                "peak_memory_mb": max(r.get("rss_mb", 0) for r in self.resource_usage) if self.resource_usage else None,
                "peak_cpu_percent": max(r.get("cpu_percent", 0) for r in self.resource_usage) if self.resource_usage else None
            }
            
        # Add plots if requested and bokeh is available
        if include_plots:
            report["plots"] = "Plots generation not implemented"
            
        return report

    # ------------------
    # MODEL EXPLAINABILITY
    # ------------------
    def explain_prediction(self, data, method="shap"):
        """
        Generate explanations for model predictions.
        
        Args:
            data: Data for which to explain predictions
            method: Explanation method (shap, lime, etc.)
            
        Returns:
            Dictionary with explanation results
        """
        if method.lower() not in ["shap", "lime", "permutation", "pdp"]:
            self.logger.warning(f"Unsupported explanation method: {method}")
            return None
            
        # Check if we have a model
        if not (self.model or self.best_model):
            raise NotFittedError("Model is not trained yet.")
            
        model = self.best_model if self.best_model else self.model
        
        # Convert input to appropriate format
        if isinstance(data, pl.DataFrame):
            # Apply preprocessor if available
            if self.preprocessor:
                processed_data = self.preprocessor.transform(data)
                X = processed_data.select(self.feature_columns).to_numpy()
            else:
                X = data.select(self.feature_columns).to_numpy()
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            df = pl.DataFrame([data])
            return self.explain_prediction(df, method)
        elif isinstance(data, (list, np.ndarray)):
            # Assume data is already preprocessed and in correct order
            X = np.array(data)
        else:
            raise ValueError(f"Unsupported data type for explanation: {type(data)}")
            
        # SHAP explanations
        if method.lower() == "shap":
            try:
                import shap
                explainer = None
                
                # Choose appropriate explainer based on model type
                mt = self.config.model_training.model_type
                if mt == ModelType.XGBOOST:
                    explainer = shap.TreeExplainer(model)
                elif mt == ModelType.LIGHTGBM:
                    explainer = shap.TreeExplainer(model)
                elif mt == ModelType.SKLEARN:
                    model_type = type(model).__name__
                    if "Random" in model_type or "Forest" in model_type or "Tree" in model_type:
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
                
                if explainer is None:
                    self.logger.warning("Could not determine appropriate SHAP explainer for model")
                    return None
                    
                # Calculate SHAP values
                shap_values = explainer.shap_values(X)
                
                # Format output
                explanation = {
                    "method": "shap",
                    "feature_names": self.feature_columns,
                    "base_value": explainer.expected_value if hasattr(explainer, "expected_value") else None
                }
                
                # Handle different output formats from different explainers
                if isinstance(shap_values, list):
                    # Multi-class case
                    explanation["shap_values"] = [values.tolist() for values in shap_values]
                else:
                    # Binary or regression case
                    explanation["shap_values"] = shap_values.tolist()
                    
                return explanation
            except ImportError:
                self.logger.error("SHAP package not installed. Install with: pip install shap")
                return None
            except Exception as e:
                self.logger.error(f"Error generating SHAP explanations: {e}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                return None
                
        elif method.lower() == "lime":
            self.logger.warning("LIME explanations not implemented")
            return None
            
        elif method.lower() == "permutation":
            try:
                from sklearn.inspection import permutation_importance
                
                # Get training data for background distribution
                X_train = self.train_data.select(self.feature_columns).to_numpy()
                y_train = self.train_data.select(self.config.dataset.target_column).to_numpy().ravel()
                
                # Calculate permutation importance
                r = permutation_importance(
                    model, X_train, y_train, 
                    n_repeats=10, 
                    random_state=self.config.dataset.random_seed
                )
                
                # Format output
                perm_imp = {
                    "method": "permutation_importance",
                    "feature_names": self.feature_columns,
                    "importances_mean": r.importances_mean.tolist(),
                    "importances_std": r.importances_std.tolist()
                }
                
                return perm_imp
            except ImportError:
                self.logger.error("scikit-learn permutation_importance not available")
                return None
            except Exception as e:
                self.logger.error(f"Error calculating permutation importance: {e}")
                if self.config.debug_mode:
                    self.logger.error(traceback.format_exc())
                return None
                
        elif method.lower() == "pdp":
            self.logger.warning("Partial Dependence Plot explanations not implemented")
            return None
            
        return None

    # ------------------
    # API METHODS
    # ------------------
    def get_status(self):
        """Get current status of the training engine."""
        return {
            "training_id": self.training_id,
            "state": self.state.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "model_name": self.config.model_registry.model_name,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "runtime_metrics": self.get_runtime_metrics()
        }
        
    def to_dict(self):
        """Convert important training engine attributes to a dictionary."""
        return {
            "training_id": self.training_id,
            "state": self.state.name,
            "config": self.config.to_dict(),
            "runtime_metrics": self.runtime_metrics,
            "evaluation_results": self.evaluation_results if hasattr(self, "evaluation_results") else None,
            "feature_columns": self.feature_columns
        }
        
    def batch_predict(self, data_path: str, output_path: str, chunk_size: int = 10000):
        """
        Perform batch prediction on a large dataset, processing in chunks.
        
        Args:
            data_path: Path to input data file
            output_path: Path where to save predictions
            chunk_size: Number of rows to process at once
            
        Returns:
            Dictionary with prediction results summary
        """
        if not (self.model or self.best_model):
            raise NotFittedError("Model is not trained yet.")
            
        model = self.best_model if self.best_model else self.model
        
        # Determine file format and appropriate reader
        if data_path.endswith('.csv'):
            reader = pl.scan_csv(
                data_path,
                separator=self.config.dataset.csv_separator,
                has_header=self.config.dataset.has_header
            )
        elif data_path.endswith('.parquet'):
            reader = pl.scan_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format for batch prediction: {data_path}")
            
        # Setup output file
        out_ext = output_path.split('.')[-1] if '.' in output_path else 'csv'
        if out_ext not in ['csv', 'parquet']:
            self.logger.warning(f"Unsupported output format: {out_ext}. Defaulting to CSV.")
            out_ext = 'csv'
            output_path = f"{output_path}.csv"
            
        # Process in batches
        total_rows = 0
        processed_rows = 0
        
        # Get total count for progress reporting
        try:
            # This might be expensive for some file formats
            total_rows = reader.select(pl.count()).collect().item()
            self.logger.info(f"Starting batch prediction on {total_rows} rows")
        except Exception:
            self.logger.info(f"Could not determine total row count. Starting batch prediction.")
        
        # Process chunks
        start_time = time.time()
        
        # First chunk flag to handle headers properly
        first_chunk = True
        
        # Process the data in chunks
        for batch in reader.collect(streaming=True):
            batch_size = len(batch)
            processed_rows += batch_size
            
            # Apply preprocessor if available
            if self.preprocessor:
                processed_batch = self.preprocessor.transform(batch)
                X = processed_batch.select(self.feature_columns).to_numpy()
            else:
                # Make sure we have the required columns
                missing_cols = set(self.feature_columns) - set(batch.columns)
                if missing_cols:
                    self.logger.error(f"Missing columns in batch: {missing_cols}")
                    raise ValueError(f"Missing required feature columns: {missing_cols}")
                X = batch.select(self.feature_columns).to_numpy()
                
            # Generate predictions
            if self.config.model_training.problem_type == "classification":
                y_pred = model.predict(X)
                
                # Add probabilities if available
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X)
                        result_batch = batch.with_column(pl.lit(y_pred).alias("prediction"))
                        
                        # Add probability columns for each class
                        if y_prob.shape[1] > 1:
                            for i in range(y_prob.shape[1]):
                                result_batch = result_batch.with_column(
                                    pl.lit(y_prob[:, i]).alias(f"probability_class_{i}")
                                )
                        else:
                            result_batch = result_batch.with_column(
                                pl.lit(y_prob[:, 0]).alias("probability")
                            )
                    except Exception as e:
                        self.logger.warning(f"Could not generate prediction probabilities: {e}")
                        result_batch = batch.with_column(pl.lit(y_pred).alias("prediction"))
                else:
                    result_batch = batch.with_column(pl.lit(y_pred).alias("prediction"))
            else:
                # Regression
                y_pred = model.predict(X)
                result_batch = batch.with_column(pl.lit(y_pred).alias("prediction"))
                
            # Write results
            if out_ext == 'csv':
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                result_batch.write_csv(output_path, has_header=header, mode=mode)
            elif out_ext == 'parquet':
                if first_chunk:
                    result_batch.write_parquet(output_path)
                else:
                    # Append to existing parquet file
                    # This is not ideal for parquet - better to collect all predictions first
                    # but keeping the pattern consistent with CSV
                    existing = pl.read_parquet(output_path)
                    combined = pl.concat([existing, result_batch])
                    combined.write_parquet(output_path)
                    
            first_chunk = False
            
            # Report progress
            if total_rows > 0:
                progress = processed_rows / total_rows * 100
                self.logger.info(f"Processed {processed_rows}/{total_rows} rows ({progress:.1f}%)")
            else:
                self.logger.info(f"Processed {processed_rows} rows")
                
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info(
            f"Batch prediction completed in {duration:.2f} seconds. "
            f"Processed {processed_rows} rows ({processed_rows/duration:.1f} rows/sec)"
        )
        
        return {
            "success": True,
            "input_path": data_path,
            "output_path": output_path,
            "rows_processed": processed_rows,
            "processing_time_seconds": duration,
            "rows_per_second": processed_rows / duration if duration > 0 else 0
        }

    # ------------------
    # CONTEXT MANAGER
    # ------------------
    def __enter__(self):
        """Allow using TrainingEngine as a context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        self.cleanup()
        
        # Log any unhandled exceptions
        if exc_type is not None:
            self.logger.error(f"Exception during TrainingEngine execution: {exc_type.__name__}: {exc_val}")
            self._set_state(TrainingState.FAILED)
            return False  # Don't suppress the exception
            
        return True
