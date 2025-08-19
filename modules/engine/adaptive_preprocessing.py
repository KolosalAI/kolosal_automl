"""
Enhanced Data Preprocessing Configuration for Large Datasets

This module provides optimized preprocessing configurations that adapt to dataset size
and available system resources, ensuring efficient processing of both small and large datasets.
"""

import psutil
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from enum import Enum

from modules.configs import PreprocessorConfig, NormalizationType
from modules.engine.optimized_data_loader import DatasetSize


class ProcessingMode(Enum):
    """Processing modes optimized for different scenarios"""
    SPEED = "speed"                    # Prioritize speed over memory
    MEMORY = "memory"                  # Prioritize memory efficiency
    BALANCED = "balanced"              # Balance between speed and memory
    QUALITY = "quality"                # Prioritize data quality and accuracy
    LARGE_SCALE = "large_scale"        # Optimized for very large datasets


class PreprocessingStrategy(Enum):
    """Preprocessing strategies for different dataset types"""
    STANDARD = "standard"              # Standard preprocessing
    MEMORY_OPTIMIZED = "memory_optimized"  # Optimized for memory efficiency
    CATEGORICAL_OPTIMIZED = "categorical_optimized"  # Optimized for high cardinality data
    SPARSE_OPTIMIZED = "sparse_optimized"  # Optimized for sparse data


@dataclass
class DatasetCharacteristics:
    """Dataset characteristics for adaptive preprocessing"""
    n_samples: int = 0
    n_features: int = 0
    n_categorical: int = 0
    n_numerical: int = 0
    missing_ratio: float = 0.0
    memory_usage_mb: float = 0.0
    sparsity_ratio: float = 0.0
    outlier_ratio: float = 0.0
    skewness_avg: float = 0.0
    cardinality_avg: float = 0.0


@dataclass
class AdaptivePreprocessorConfig:
    """
    Adaptive preprocessing configuration that adjusts based on dataset characteristics
    """
    # Base configuration
    base_config: Optional[PreprocessorConfig] = None
    
    # Adaptive settings
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    auto_adjust_chunk_size: bool = True
    auto_adjust_workers: bool = True
    auto_select_normalization: bool = True
    
    # Adaptive boolean flags
    auto_normalization: bool = True
    adaptive_missing_handling: bool = True
    smart_categorical_encoding: bool = True
    dynamic_outlier_detection: bool = True
    memory_aware_processing: bool = True
    
    # Thresholds
    large_dataset_threshold: int = 50000
    high_cardinality_threshold: int = 100
    sparse_data_threshold: float = 0.5
    
    # Memory management
    max_memory_usage_pct: float = 75.0
    enable_memory_monitoring: bool = True
    gc_frequency: int = 10  # Trigger GC every N chunks
    
    # Performance settings
    min_chunk_size: int = 1000
    max_chunk_size: int = 100000
    target_chunk_memory_mb: float = 100.0
    
    # Quality vs Performance tradeoffs
    enable_outlier_detection: bool = True
    outlier_detection_sample_size: Optional[int] = None  # Sample for large datasets
    enable_statistical_validation: bool = True
    
    # Large dataset specific
    use_reservoir_sampling: bool = True
    reservoir_size: int = 100000
    streaming_statistics: bool = True
    incremental_fitting: bool = True
    
    def __post_init__(self):
        if self.base_config is None:
            self.base_config = PreprocessorConfig()
    
    def determine_strategy(self, characteristics: DatasetCharacteristics) -> PreprocessingStrategy:
        """Determine the optimal preprocessing strategy based on dataset characteristics"""
        # Large dataset strategy
        if characteristics.memory_usage_mb > 500 or characteristics.n_samples >= 100_000:
            return PreprocessingStrategy.MEMORY_OPTIMIZED
        
        # High cardinality strategy
        if characteristics.cardinality_avg > 100:
            return PreprocessingStrategy.CATEGORICAL_OPTIMIZED
        
        # Sparse data strategy
        if characteristics.sparsity_ratio > 0.7:
            return PreprocessingStrategy.SPARSE_OPTIMIZED
        
        # Default strategy
        return PreprocessingStrategy.STANDARD
    
    def select_normalization_method(self, characteristics: DatasetCharacteristics) -> NormalizationType:
        """Select optimal normalization method based on dataset characteristics"""
        # Use robust scaling for datasets with high outlier ratio or skewness
        if characteristics.outlier_ratio > 0.1 or characteristics.skewness_avg > 2.0:
            return NormalizationType.ROBUST
        
        # Use MinMax for sparse data
        if characteristics.sparsity_ratio > 0.5:
            return NormalizationType.MINMAX
        
        # Default to standard normalization
        return NormalizationType.STANDARD


class ConfigOptimizer:
    """Configuration optimizer for adaptive preprocessing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset_characteristics(self, df: pd.DataFrame) -> DatasetCharacteristics:
        """Analyze dataset to determine its characteristics"""
        n_samples, n_features = df.shape
        
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        n_numerical = len(numerical_cols)
        n_categorical = len(categorical_cols)
        
        # Calculate missing ratio
        missing_ratio = df.isnull().sum().sum() / (n_samples * n_features) if n_samples * n_features > 0 else 0
        
        # Calculate memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Calculate sparsity ratio (for numerical columns)
        sparsity_ratio = 0.0
        if n_numerical > 0:
            numerical_data = df[numerical_cols]
            zero_count = (numerical_data == 0).sum().sum()
            total_numerical_values = numerical_data.size
            sparsity_ratio = zero_count / total_numerical_values if total_numerical_values > 0 else 0
        
        # Calculate outlier ratio using IQR method
        outlier_ratio = 0.0
        if n_numerical > 0:
            outlier_counts = []
            for col in numerical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    outlier_counts.append(outliers)
            
            total_outliers = sum(outlier_counts)
            total_numerical_values = df[numerical_cols].size
            outlier_ratio = total_outliers / total_numerical_values if total_numerical_values > 0 else 0
        
        # Calculate average skewness
        skewness_avg = 0.0
        if n_numerical > 0:
            skewness_values = []
            for col in numerical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 1:
                    skew = col_data.skew()
                    if not np.isnan(skew):
                        skewness_values.append(abs(skew))
            skewness_avg = np.mean(skewness_values) if skewness_values else 0.0
        
        # Calculate average cardinality for categorical columns
        cardinality_avg = 0.0
        if n_categorical > 0:
            cardinalities = [df[col].nunique() for col in categorical_cols]
            cardinality_avg = np.mean(cardinalities) if cardinalities else 0.0
        
        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_categorical=n_categorical,
            n_numerical=n_numerical,
            missing_ratio=missing_ratio,
            memory_usage_mb=memory_usage_mb,
            sparsity_ratio=sparsity_ratio,
            outlier_ratio=outlier_ratio,
            skewness_avg=skewness_avg,
            cardinality_avg=cardinality_avg
        )
    
    def optimize_for_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize preprocessing configuration for a specific dataset"""
        characteristics = self.analyze_dataset_characteristics(df)
        
        # Create adaptive config
        config = AdaptivePreprocessorConfig()
        strategy = config.determine_strategy(characteristics)
        normalization = config.select_normalization_method(characteristics)
        
        # Generate suggestions
        suggestions = self.get_memory_optimization_suggestions(characteristics)
        
        # Calculate optimal chunk size
        chunk_size = self.suggest_chunk_size(characteristics)
        
        # Estimate processing time
        time_estimate = self.estimate_processing_time(characteristics)
        
        return {
            'strategy': strategy,
            'normalization': normalization,
            'characteristics': characteristics,
            'suggestions': suggestions,
            'chunk_size': chunk_size,
            'time_estimate': time_estimate
        }
    
    def suggest_chunk_size(self, characteristics: DatasetCharacteristics) -> int:
        """Suggest optimal chunk size based on dataset characteristics"""
        # Base chunk size
        if characteristics.memory_usage_mb < 100:
            return characteristics.n_samples  # Process all at once
        elif characteristics.memory_usage_mb < 500:
            return max(10000, characteristics.n_samples // 4)
        elif characteristics.memory_usage_mb < 1000:
            return max(5000, characteristics.n_samples // 8)
        else:
            return max(1000, characteristics.n_samples // 16)
    
    def estimate_processing_time(self, characteristics: DatasetCharacteristics) -> Dict[str, float]:
        """Estimate processing time for the dataset"""
        # Simple heuristic based on size and complexity
        base_time_per_sample = 0.001  # 1ms per sample base
        
        # Adjust for complexity
        complexity_factor = 1.0
        if characteristics.n_categorical > 0:
            complexity_factor += 0.5 * (characteristics.cardinality_avg / 100)
        if characteristics.missing_ratio > 0.1:
            complexity_factor += 0.3
        if characteristics.outlier_ratio > 0.05:
            complexity_factor += 0.2
        
        processing_time = characteristics.n_samples * base_time_per_sample * complexity_factor
        loading_time = characteristics.memory_usage_mb * 0.01  # 10ms per MB
        memory_optimization_time = characteristics.memory_usage_mb * 0.005  # 5ms per MB
        
        return {
            'total_seconds': processing_time + loading_time + memory_optimization_time,
            'loading_seconds': loading_time,
            'preprocessing_seconds': processing_time,
            'memory_optimization_seconds': memory_optimization_time
        }
    
    def get_memory_optimization_suggestions(self, characteristics: DatasetCharacteristics) -> List[str]:
        """Get memory optimization suggestions based on dataset characteristics"""
        suggestions = []
        
        if characteristics.memory_usage_mb > 500:
            suggestions.append("Consider using chunked processing for large dataset")
        
        if characteristics.sparsity_ratio > 0.5:
            suggestions.append("Dataset is sparse - consider using sparse matrix formats")
        
        if characteristics.cardinality_avg > 100:
            suggestions.append("High cardinality categorical data - consider feature hashing or target encoding")
        
        if characteristics.missing_ratio > 0.2:
            suggestions.append("High missing data ratio - consider advanced imputation strategies")
        
        if characteristics.outlier_ratio > 0.1:
            suggestions.append("High outlier ratio detected - consider robust preprocessing methods")
        
        return suggestions


class PreprocessorConfigOptimizer:
    """
    Optimizer that creates preprocessing configurations based on dataset characteristics
    and system resources
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': cpu_count,
            'memory_usage_pct': memory.percent
        }
    
    def optimize_for_dataset(self, 
                           dataset_size: DatasetSize,
                           estimated_memory_mb: float,
                           num_features: Optional[int] = None,
                           processing_mode: ProcessingMode = ProcessingMode.BALANCED,
                           target_memory_pct: float = 75.0) -> PreprocessorConfig:
        """
        Create optimized preprocessing configuration for specific dataset characteristics
        
        Args:
            dataset_size: Size category of the dataset
            estimated_memory_mb: Estimated memory usage
            num_features: Number of features in the dataset
            processing_mode: Processing mode preference
            target_memory_pct: Target memory usage percentage
            
        Returns:
            Optimized PreprocessorConfig
        """
        self.logger.info(f"Optimizing preprocessing config for {dataset_size.value} dataset "
                        f"({estimated_memory_mb:.1f}MB) in {processing_mode.value} mode")
        
        # Calculate optimal settings based on dataset size and resources
        config_params = self._calculate_optimal_params(
            dataset_size, estimated_memory_mb, num_features, 
            processing_mode, target_memory_pct
        )
        
        # Create base configuration
        config = PreprocessorConfig(**config_params)
        
        # Apply dataset-specific optimizations
        if dataset_size in [DatasetSize.LARGE, DatasetSize.HUGE]:
            config = self._optimize_for_large_dataset(config, dataset_size, processing_mode)
        elif dataset_size == DatasetSize.TINY:
            config = self._optimize_for_small_dataset(config, processing_mode)
        
        self.logger.info(f"Generated config: chunk_size={config.chunk_size}, "
                        f"n_jobs={config.n_jobs}, "
                        f"normalization={config.normalization}, "
                        f"parallel_processing={config.parallel_processing}")
        
        return config
    
    def _calculate_optimal_params(self, 
                                dataset_size: DatasetSize,
                                estimated_memory_mb: float,
                                num_features: Optional[int],
                                processing_mode: ProcessingMode,
                                target_memory_pct: float) -> Dict[str, Any]:
        """Calculate optimal parameters based on inputs"""
        
        available_memory_mb = self._system_info['available_memory_gb'] * 1024
        target_memory_mb = available_memory_mb * (target_memory_pct / 100)
        
        # Base parameters
        params = {
            'normalization': self._select_normalization(dataset_size, processing_mode),
            'handle_nan': True,
            'handle_inf': True,
            'nan_strategy': 'mean',
            'inf_strategy': 'max_value',
            'copy_X': True,
            'epsilon': 1e-10,
            'debug_mode': False,
            'cache_enabled': processing_mode != ProcessingMode.MEMORY,
            'enable_input_validation': dataset_size != DatasetSize.HUGE,
        }
        
        # Memory and performance parameters
        if processing_mode == ProcessingMode.SPEED:
            params.update(self._get_speed_optimized_params(dataset_size))
        elif processing_mode == ProcessingMode.MEMORY:
            params.update(self._get_memory_optimized_params(dataset_size, target_memory_mb))
        elif processing_mode == ProcessingMode.LARGE_SCALE:
            params.update(self._get_large_scale_params(dataset_size, target_memory_mb))
        elif processing_mode == ProcessingMode.QUALITY:
            params.update(self._get_quality_optimized_params(dataset_size))
        else:  # BALANCED
            params.update(self._get_balanced_params(dataset_size, target_memory_mb))
        
        # Chunk size optimization
        params['chunk_size'] = self._calculate_optimal_chunk_size(
            dataset_size, estimated_memory_mb, target_memory_mb, processing_mode
        )
        
        # Worker count optimization
        params['n_jobs'] = self._calculate_optimal_workers(
            dataset_size, processing_mode, params.get('parallel_processing', True)
        )
        
        return params
    
    def _select_normalization(self, dataset_size: DatasetSize, 
                            processing_mode: ProcessingMode) -> NormalizationType:
        """Select optimal normalization method"""
        if processing_mode == ProcessingMode.SPEED:
            return NormalizationType.MINMAX  # Faster than standard
        elif processing_mode == ProcessingMode.MEMORY:
            return NormalizationType.MINMAX  # More memory efficient
        elif dataset_size == DatasetSize.HUGE:
            return NormalizationType.ROBUST   # More stable for large datasets
        else:
            return NormalizationType.STANDARD  # Default high quality
    
    def _get_speed_optimized_params(self, dataset_size: DatasetSize) -> Dict[str, Any]:
        """Parameters optimized for speed"""
        return {
            'parallel_processing': True,
            'detect_outliers': dataset_size not in [DatasetSize.LARGE, DatasetSize.HUGE],
            'outlier_method': 'zscore',  # Faster than IQR
            'outlier_handling': 'clip',  # Faster than removal
            'cache_preprocessing': True,
            'dtype': 'float32',  # Faster processing than float64
        }
    
    def _get_memory_optimized_params(self, dataset_size: DatasetSize, 
                                   target_memory_mb: float) -> Dict[str, Any]:
        """Parameters optimized for memory efficiency"""
        return {
            'parallel_processing': dataset_size == DatasetSize.SMALL,  # Limited parallelism
            'detect_outliers': dataset_size not in [DatasetSize.LARGE, DatasetSize.HUGE],
            'cache_preprocessing': False,
            'cache_enabled': False,
            'dtype': 'float32',  # Use less memory than float64
            'copy_X': False,  # Avoid unnecessary copies
        }
    
    def _get_large_scale_params(self, dataset_size: DatasetSize, 
                              target_memory_mb: float) -> Dict[str, Any]:
        """Parameters optimized for very large datasets"""
        return {
            'parallel_processing': True,
            'detect_outliers': False,  # Skip for performance
            'cache_preprocessing': False,
            'cache_enabled': False,
            'dtype': 'float32',
            'copy_X': False,
            'enable_input_validation': False,  # Skip validation for performance
        }
    
    def _get_quality_optimized_params(self, dataset_size: DatasetSize) -> Dict[str, Any]:
        """Parameters optimized for data quality"""
        return {
            'parallel_processing': True,
            'detect_outliers': True,
            'outlier_method': 'iqr',  # More robust than zscore
            'outlier_handling': 'winsorize',  # Better than clipping
            'cache_preprocessing': True,
            'dtype': 'float64',  # Higher precision
            'enable_input_validation': True,
        }
    
    def _get_balanced_params(self, dataset_size: DatasetSize, 
                           target_memory_mb: float) -> Dict[str, Any]:
        """Balanced parameters between speed and memory"""
        enable_outliers = dataset_size in [DatasetSize.TINY, DatasetSize.SMALL, DatasetSize.MEDIUM]
        use_cache = dataset_size in [DatasetSize.TINY, DatasetSize.SMALL]
        
        return {
            'parallel_processing': True,
            'detect_outliers': enable_outliers,
            'outlier_method': 'zscore',
            'outlier_handling': 'clip',
            'cache_preprocessing': use_cache,
            'cache_enabled': use_cache,
            'dtype': 'float32' if dataset_size in [DatasetSize.LARGE, DatasetSize.HUGE] else 'float64',
        }
    
    def _calculate_optimal_chunk_size(self, 
                                    dataset_size: DatasetSize,
                                    estimated_memory_mb: float,
                                    target_memory_mb: float,
                                    processing_mode: ProcessingMode) -> Optional[int]:
        """Calculate optimal chunk size for processing"""
        
        if dataset_size in [DatasetSize.TINY, DatasetSize.SMALL]:
            return None  # No chunking needed
        
        # Base chunk sizes by dataset size
        base_chunks = {
            DatasetSize.MEDIUM: 50000,
            DatasetSize.LARGE: 25000,
            DatasetSize.HUGE: 10000
        }
        
        base_chunk_size = base_chunks.get(dataset_size, 10000)
        
        # Adjust based on available memory
        if estimated_memory_mb > target_memory_mb:
            # Reduce chunk size if memory is constrained
            memory_ratio = target_memory_mb / estimated_memory_mb
            base_chunk_size = int(base_chunk_size * memory_ratio)
        
        # Mode-specific adjustments
        if processing_mode == ProcessingMode.SPEED:
            base_chunk_size = int(base_chunk_size * 1.5)  # Larger chunks for speed
        elif processing_mode == ProcessingMode.MEMORY:
            base_chunk_size = int(base_chunk_size * 0.5)  # Smaller chunks for memory
        
        # Ensure reasonable bounds
        min_chunk = 1000
        max_chunk = 100000
        
        return max(min_chunk, min(max_chunk, base_chunk_size))
    
    def _calculate_optimal_workers(self, 
                                 dataset_size: DatasetSize,
                                 processing_mode: ProcessingMode,
                                 parallel_enabled: bool) -> int:
        """Calculate optimal number of worker processes"""
        
        if not parallel_enabled:
            return 1
        
        cpu_count = self._system_info['cpu_count_logical']
        
        if processing_mode == ProcessingMode.SPEED:
            # Use more workers for speed
            return max(1, cpu_count - 1)
        elif processing_mode == ProcessingMode.MEMORY:
            # Use fewer workers to conserve memory
            return max(1, min(2, cpu_count // 2))
        elif dataset_size == DatasetSize.HUGE:
            # Conservative for very large datasets
            return max(1, min(4, cpu_count // 2))
        else:
            # Balanced approach
            return max(1, min(cpu_count // 2, 4))
    
    def _optimize_for_large_dataset(self, 
                                  config: PreprocessorConfig,
                                  dataset_size: DatasetSize,
                                  processing_mode: ProcessingMode) -> PreprocessorConfig:
        """Apply specific optimizations for large datasets"""
        
        # Enable chunked processing
        if config.chunk_size is None:
            config.chunk_size = 10000
        
        # Disable expensive operations
        if dataset_size == DatasetSize.HUGE:
            config.detect_outliers = False
            config.cache_preprocessing = False
            config.cache_enabled = False
            config.enable_input_validation = False
        
        # Use more efficient data types
        if config.dtype == 'float64':
            config.dtype = 'float32'
        
        return config
    
    def _optimize_for_small_dataset(self, 
                                  config: PreprocessorConfig,
                                  processing_mode: ProcessingMode) -> PreprocessorConfig:
        """Apply specific optimizations for small datasets"""
        
        # No chunking needed
        config.chunk_size = None
        
        # Can afford more expensive operations
        config.detect_outliers = True
        config.cache_preprocessing = True
        config.cache_enabled = True
        config.enable_input_validation = True
        
        # Use higher precision if not in memory mode
        if processing_mode != ProcessingMode.MEMORY:
            config.dtype = 'float64'
        
        return config


# Convenience functions for integration
def create_adaptive_config(auto_normalization: bool = True,
                         adaptive_missing_handling: bool = True,
                         smart_categorical_encoding: bool = True,
                         dynamic_outlier_detection: bool = True,
                         memory_aware: bool = True,
                         large_dataset_threshold: int = 50000,
                         high_cardinality_threshold: int = 100,
                         sparse_data_threshold: float = 0.5) -> AdaptivePreprocessorConfig:
    """Create an adaptive preprocessor configuration with specified settings"""
    config = AdaptivePreprocessorConfig()
    config.auto_normalization = auto_normalization
    config.adaptive_missing_handling = adaptive_missing_handling
    config.smart_categorical_encoding = smart_categorical_encoding
    config.dynamic_outlier_detection = dynamic_outlier_detection
    config.memory_aware_processing = memory_aware
    config.large_dataset_threshold = large_dataset_threshold
    config.high_cardinality_threshold = high_cardinality_threshold
    config.sparse_data_threshold = sparse_data_threshold
    return config


def optimize_preprocessing_config(data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Optimize preprocessing configuration for a dataset
    
    Args:
        data: Either a file path to CSV or a pandas DataFrame
        
    Returns:
        Optimized configuration dictionary
    """
    optimizer = ConfigOptimizer()
    
    if isinstance(data, str):
        # Load DataFrame from file path
        df = pd.read_csv(data)
    else:
        df = data
    
    return optimizer.optimize_for_dataset(df)


def create_optimized_preprocessor_config(dataset_size: DatasetSize,
                                       estimated_memory_mb: float,
                                       processing_mode: ProcessingMode = ProcessingMode.BALANCED,
                                       num_features: Optional[int] = None) -> PreprocessorConfig:
    """
    Create an optimized preprocessor configuration
    
    Args:
        dataset_size: Size category of the dataset
        estimated_memory_mb: Estimated memory usage
        processing_mode: Processing mode preference
        num_features: Number of features (optional)
        
    Returns:
        Optimized PreprocessorConfig
    """
    optimizer = PreprocessorConfigOptimizer()
    return optimizer.optimize_for_dataset(
        dataset_size=dataset_size,
        estimated_memory_mb=estimated_memory_mb,
        num_features=num_features,
        processing_mode=processing_mode
    )


def get_recommended_processing_mode(dataset_size: DatasetSize, 
                                  available_memory_gb: float,
                                  priority: str = "balanced") -> ProcessingMode:
    """
    Get recommended processing mode based on dataset and system characteristics
    
    Args:
        dataset_size: Size category of the dataset
        available_memory_gb: Available system memory in GB
        priority: Priority preference ("speed", "memory", "quality", "balanced")
        
    Returns:
        Recommended ProcessingMode
    """
    if priority == "speed":
        return ProcessingMode.SPEED
    elif priority == "memory":
        return ProcessingMode.MEMORY
    elif priority == "quality":
        return ProcessingMode.QUALITY
    elif dataset_size == DatasetSize.HUGE:
        return ProcessingMode.LARGE_SCALE
    elif available_memory_gb < 4.0:  # Low memory system
        return ProcessingMode.MEMORY
    else:
        return ProcessingMode.BALANCED
