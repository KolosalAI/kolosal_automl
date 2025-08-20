"""
Integration Utilities for Kolosal AutoML Optimization System

This module provides easy-to-use functions for integrating the optimization
system with existing codebases and workflows.
"""

import logging
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optimization modules with fallbacks
try:
    from .optimized_data_loader import (
        OptimizedDataLoader, 
        DatasetSize, 
        LoadingStrategy,
        load_data_optimized
    )
    from .adaptive_preprocessing import (
        AdaptivePreprocessorConfig,
        PreprocessorConfigOptimizer,
        ProcessingMode
    )
    from .memory_aware_processor import (
        MemoryAwareDataProcessor,
        create_memory_aware_processor
    )
    from .data_preprocessor import DataPreprocessor
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimization modules not available: {e}")
    OPTIMIZATION_AVAILABLE = False


class OptimizedDataPipeline:
    """
    Complete optimized data pipeline for loading, preprocessing, and training
    """
    
    def __init__(self, 
                 max_memory_pct: float = 75.0,
                 enable_memory_optimization: bool = True,
                 enable_adaptive_preprocessing: bool = True):
        """
        Initialize optimized data pipeline
        
        Args:
            max_memory_pct: Maximum percentage of available memory to use
            enable_memory_optimization: Enable memory optimization features
            enable_adaptive_preprocessing: Enable adaptive preprocessing
        """
        self.max_memory_pct = max_memory_pct
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_adaptive_preprocessing = enable_adaptive_preprocessing
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self._data_loader = None
        self._preprocessor = None
        self._memory_processor = None
        self._config_optimizer = None
        
        if OPTIMIZATION_AVAILABLE:
            self._initialize_optimized_components()
        else:
            self._initialize_fallback_components()
    
    @property
    def data_loader(self):
        """Expose data_loader for testing"""
        return self._data_loader
    
    @data_loader.setter
    def data_loader(self, value):
        """Allow setting data_loader for testing"""
        self._data_loader = value
    
    @data_loader.deleter
    def data_loader(self):
        """Allow deleting data_loader for testing"""
        self._data_loader = None
    
    @property
    def config_optimizer(self):
        """Expose config_optimizer for testing"""
        return self._config_optimizer
    
    @config_optimizer.setter
    def config_optimizer(self, value):
        """Allow setting config_optimizer for testing"""
        self._config_optimizer = value
    
    @config_optimizer.deleter
    def config_optimizer(self):
        """Allow deleting config_optimizer for testing"""
        self._config_optimizer = None
    
    @property
    def memory_processor(self):
        """Expose memory_processor for testing"""
        return self._memory_processor
    
    @memory_processor.setter
    def memory_processor(self, value):
        """Allow setting memory_processor for testing"""
        self._memory_processor = value
    
    @memory_processor.deleter
    def memory_processor(self):
        """Allow deleting memory_processor for testing"""
        self._memory_processor = None
    
    def _initialize_optimized_components(self):
        """Initialize optimized components"""
        try:
            # Data loader with optimization
            self._data_loader = OptimizedDataLoader()
            logger.info("Optimized data loader initialized")
            
            # Memory processor
            if self.enable_memory_optimization:
                self._memory_processor = create_memory_aware_processor()
                logger.info("Memory-aware processor initialized")
            
            # Adaptive preprocessing optimizer
            if self.enable_adaptive_preprocessing:
                self._config_optimizer = PreprocessorConfigOptimizer()
                logger.info("Adaptive preprocessing optimizer initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize optimization components: {e}")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self):
        """Initialize fallback components when optimization is not available"""
        logger.info("Using fallback components (pandas-based)")
        # Fallback components would be initialized here
        pass
    
    def load_data(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load data with automatic optimization
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dictionary with loaded data and optimization info
        """
        result = {
            'success': False,
            'data': None,
            'dataset_info': None,
            'optimization_info': None,
            'error': None
        }
        
        try:
            if OPTIMIZATION_AVAILABLE and self._data_loader:
                try:
                    df, dataset_info = self._data_loader.load_data(file_path, **kwargs)
                    
                    optimization_info = {
                        'optimized_loading': True,
                        'size_category': dataset_info.size_category.value,
                        'loading_strategy': dataset_info.loading_strategy.value,
                        'memory_mb': dataset_info.actual_memory_mb,
                        'loading_time': dataset_info.loading_time,
                        'optimizations_applied': dataset_info.optimization_applied
                    }
                    
                    result.update({
                        'success': True,
                        'data': df,
                        'dataset_info': dataset_info,
                        'optimization_info': optimization_info
                    })
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Optimized loading failed, using fallback: {e}")
            
            # Fallback to pandas
            df = pd.read_csv(file_path, **kwargs) if file_path.endswith('.csv') else pd.read_parquet(file_path, **kwargs)
            
            optimization_info = {
                'optimized_loading': False,
                'fallback_used': True,
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'rows': len(df),
                'columns': len(df.columns)
            }
            
            result.update({
                'success': True,
                'data': df,
                'optimization_info': optimization_info
            })
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to load data: {e}")
        
        return result
    
    def optimize_preprocessing(self, 
                             df: pd.DataFrame, 
                             target_column: Optional[str] = None):
        """
        Create optimized preprocessor for the dataset
        
        Args:
            df: Input dataframe
            target_column: Target column name (optional)
            
        Returns:
            Dictionary with optimization results
        """
        from modules.engine.data_preprocessor import DataPreprocessor
        
        try:
            if OPTIMIZATION_AVAILABLE and self.enable_adaptive_preprocessing:
                # Create preprocessor with optimization
                preprocessor = DataPreprocessor()
                
                # Optimize for dataset if config optimizer is available
                if self._config_optimizer:
                    try:
                        optimized_preprocessor = preprocessor.optimize_for_dataset(df)
                        return {
                            'success': True,
                            'config': optimized_preprocessor.config.__dict__,
                            'preprocessor': optimized_preprocessor,
                            'recommendations': ['Adaptive preprocessing applied']
                        }
                    except Exception as e:
                        logger.warning(f"Config optimization failed: {e}")
                
                return {
                    'success': True,
                    'config': preprocessor.config.__dict__,
                    'preprocessor': preprocessor,
                    'recommendations': ['Default preprocessing configuration']
                }
                
        except Exception as e:
            logger.warning(f"Adaptive preprocessing failed, using default: {e}")
        
        # Fallback to default preprocessor
        fallback_preprocessor = DataPreprocessor()
        return {
            'success': False,
            'config': fallback_preprocessor.config.__dict__,
            'preprocessor': fallback_preprocessor,
            'recommendations': ['Fallback to default configuration due to errors'],
            'error': str(e) if 'e' in locals() else 'Unknown error'
        }
    
    def optimize_memory(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize dataframe memory usage
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with optimization results including success status and optimized data
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        try:
            if OPTIMIZATION_AVAILABLE and self.enable_memory_optimization and self._memory_processor:
                try:
                    optimized_df = self._memory_processor.optimize_dataframe_memory(df)
                except Exception as e:
                    logger.warning(f"Memory optimization failed: {e}")
                    optimized_df = self._basic_memory_optimization(df)
            else:
                # Fallback to basic optimization
                optimized_df = self._basic_memory_optimization(df)
            
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            return {
                'success': True,
                'optimized_data': optimized_df,
                'optimization_info': {
                    'memory': {
                        'original_mb': round(original_memory, 2),
                        'optimized_mb': round(optimized_memory, 2),
                        'reduction_pct': round((original_memory - optimized_memory) / original_memory * 100, 2) if original_memory > 0 else 0
                    }
                }
            }
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {
                'success': False,
                'optimized_data': df,
                'error': str(e),
                'optimization_info': {
                    'memory': {
                        'original_mb': round(original_memory, 2),
                        'optimized_mb': round(original_memory, 2),
                        'reduction_pct': 0
                    }
                }
            }
    
    def _basic_memory_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic memory optimization using pandas"""
        optimized_df = df.copy()
        
        # Optimize integers
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        # Optimize floats  
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize categories
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df[col]) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def process_complete_pipeline(self, 
                                file_path: str,
                                target_column: Optional[str] = None,
                                fit_preprocessor: bool = True) -> Dict[str, Any]:
        """
        Run complete optimized pipeline
        
        Args:
            file_path: Path to data file
            target_column: Target column name
            fit_preprocessor: Whether to fit the preprocessor
            
        Returns:
            Dictionary with all pipeline results
        """
        results = {
            'success': False,
            'data': None,
            'dataset_info': None,
            'preprocessor': None,
            'optimization_info': {},
            'errors': []
        }
        
        try:
            # Step 1: Load data with optimization
            logger.info("Loading data with optimization...")
            load_result = self.load_data(file_path)
            df = load_result['data']
            dataset_info = load_result.get('dataset_info')
            results['dataset_info'] = dataset_info
            results['optimization_info']['loading'] = dataset_info
            
            # Step 2: Memory optimization
            logger.info("Applying memory optimization...")
            memory_result = self.optimize_memory(df)
            if memory_result['success']:
                df_optimized = memory_result['optimized_data']
                results['optimization_info']['memory'] = memory_result['optimization_info']['memory']
            else:
                df_optimized = df
                results['optimization_info']['memory'] = memory_result['optimization_info']['memory']
                results['errors'].append(f"Memory optimization failed: {memory_result.get('error', 'Unknown error')}")
            
            # Step 3: Optimize preprocessing
            logger.info("Configuring optimized preprocessing...")
            preprocessing_result = self.optimize_preprocessing(df_optimized, target_column)
            preprocessor = preprocessing_result.get('preprocessor')
            results['optimization_info']['preprocessing_config'] = preprocessing_result
            
            if fit_preprocessor and target_column and target_column in df_optimized.columns:
                # Prepare features and target
                X = df_optimized.drop(columns=[target_column])
                y = df_optimized[target_column]
                
                # Fit preprocessor
                preprocessor.fit(X, y)
                results['optimization_info']['preprocessing'] = {
                    'fitted': True,
                    'n_features': X.shape[1],
                    'n_samples': X.shape[0]
                }
            
            results['data'] = df_optimized
            results['preprocessor'] = preprocessor
            results['success'] = True
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and configuration
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            'optimization_available': OPTIMIZATION_AVAILABLE,
            'components_loaded': {
                'data_loader': self._data_loader is not None,
                'memory_processor': self._memory_processor is not None,
                'config_optimizer': self._config_optimizer is not None
            },
            'configuration': {
                'max_memory_pct': self.max_memory_pct,
                'enable_memory_optimization': self.enable_memory_optimization,
                'enable_adaptive_preprocessing': self.enable_adaptive_preprocessing
            }
        }


# Convenience functions for easy integration

def quick_load_optimized(file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick function to load data with optimization
    
    Args:
        file_path: Path to data file
        **kwargs: Additional loading arguments
        
    Returns:
        Tuple of (dataframe, optimization_info)
    """
    pipeline = OptimizedDataPipeline()
    result = pipeline.load_data(file_path, **kwargs)
    
    # Convert the dict result to tuple format expected by the test
    if isinstance(result, dict) and result.get('success'):
        dataframe = result['data']
        optimization_info = {
            'dataset_info': result.get('dataset_info'),
            'optimization_info': result.get('optimization_info')
        }
        return dataframe, optimization_info
    else:
        # Fallback for error cases
        raise RuntimeError(f"Failed to load data: {result.get('error', 'Unknown error')}")


def quick_optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to optimize dataframe memory
    
    Args:
        df: Input dataframe
        
    Returns:
        Memory-optimized dataframe
    """
    pipeline = OptimizedDataPipeline()
    result = pipeline.optimize_memory(df)
    return result['optimized_data']


def quick_preprocessor_optimized(df: pd.DataFrame, target_column: Optional[str] = None) -> DataPreprocessor:
    """
    Quick function to get optimized preprocessor
    
    Args:
        df: Input dataframe
        target_column: Target column name
        
    Returns:
        Optimized preprocessor
    """
    pipeline = OptimizedDataPipeline()
    return pipeline.optimize_preprocessing(df, target_column)


def get_optimization_status() -> Dict[str, Any]:
    """
    Get status of optimization system availability
    
    Returns:
        Dictionary with optimization status information
    """
    status = {
        'optimization_available': OPTIMIZATION_AVAILABLE,
        'modules_loaded': {}
    }
    
    if OPTIMIZATION_AVAILABLE:
        try:
            # Test each module
            status['modules_loaded']['data_loader'] = OptimizedDataLoader is not None
            status['modules_loaded']['adaptive_preprocessing'] = PreprocessorConfigOptimizer is not None  
            status['modules_loaded']['memory_processor'] = MemoryAwareDataProcessor is not None
        except:
            status['optimization_available'] = False
    
    return status


def create_optimized_training_pipeline(
    max_memory_pct: float = 75.0,
    enable_memory_optimization: bool = True,
    enable_adaptive_preprocessing: bool = True
) -> OptimizedDataPipeline:
    """
    Create a complete optimized training pipeline
    
    Args:
        max_memory_pct: Maximum memory usage percentage
        enable_memory_optimization: Enable memory optimization features
        enable_adaptive_preprocessing: Enable adaptive preprocessing
        
    Returns:
        Configured optimization pipeline
    """
    return OptimizedDataPipeline(
        max_memory_pct=max_memory_pct,
        enable_memory_optimization=enable_memory_optimization,
        enable_adaptive_preprocessing=enable_adaptive_preprocessing
    )


# Example usage and integration patterns
def example_integration():
    """Example of how to integrate optimization into existing code"""
    
    # Example 1: Simple data loading with optimization
    file_path = "large_dataset.csv"
    df, info = quick_load_optimized(file_path)
    print(f"Loaded {info['rows']:,} rows with {info['loading_strategy']} strategy")
    
    # Example 2: Memory optimization
    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    df_optimized = quick_optimize_memory(df)
    new_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory reduced from {original_memory:.1f}MB to {new_memory:.1f}MB")
    
    # Example 3: Complete pipeline
    pipeline = create_optimized_training_pipeline()
    results = pipeline.process_complete_pipeline(file_path, target_column='target')
    
    if results['success']:
        print("Pipeline completed successfully!")
        print(f"Optimization info: {results['optimization_info']}")
    else:
        print(f"Pipeline failed: {results['errors']}")


if __name__ == "__main__":
    # Show optimization status
    status = get_optimization_status()
    print(f"Optimization system status: {status}")
    
    # Run example if optimization is available
    if status['optimization_available']:
        print("\nRunning optimization example...")
        example_integration()
    else:
        print("\nOptimization system not available - would use fallback methods")
