"""
Unit tests for the AdaptivePreprocessing module.

This test suite covers:
- Adaptive configuration optimization
- Dataset characteristic analysis  
- Strategy selection logic
- Configuration optimization algorithms
- Integration with existing preprocessing
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from modules.engine.adaptive_preprocessing import (
    AdaptivePreprocessorConfig,
    DatasetCharacteristics,
    PreprocessingStrategy,
    ConfigOptimizer,
    create_adaptive_config,
    optimize_preprocessing_config
)

# Import related modules for testing
try:
    from modules.configs import PreprocessorConfig, NormalizationType
except ImportError:
    # Create mock classes if not available
    class NormalizationType:
        STANDARD = "standard"
        MINMAX = "minmax"
        ROBUST = "robust"
    
    class PreprocessorConfig:
        def __init__(self):
            self.normalization = NormalizationType.STANDARD
            self.handle_nan = True
            self.handle_inf = True


class TestDatasetCharacteristics(unittest.TestCase):
    """Test the DatasetCharacteristics dataclass"""
    
    def test_characteristics_initialization(self):
        """Test DatasetCharacteristics initialization"""
        chars = DatasetCharacteristics(
            n_samples=1000,
            n_features=10,
            n_categorical=3,
            n_numerical=7,
            missing_ratio=0.05,
            memory_usage_mb=50.0,
            sparsity_ratio=0.1,
            outlier_ratio=0.02,
            skewness_avg=1.5,
            cardinality_avg=25.0
        )
        
        self.assertEqual(chars.n_samples, 1000)
        self.assertEqual(chars.n_features, 10)
        self.assertEqual(chars.n_categorical, 3)
        self.assertEqual(chars.n_numerical, 7)
        self.assertAlmostEqual(chars.missing_ratio, 0.05)
        self.assertAlmostEqual(chars.memory_usage_mb, 50.0)
    
    def test_characteristics_validation(self):
        """Test that characteristics values are reasonable"""
        chars = DatasetCharacteristics(
            n_samples=100,
            n_features=5,
            n_categorical=2,
            n_numerical=3
        )
        
        # Check that categorical + numerical = total features
        self.assertEqual(chars.n_categorical + chars.n_numerical, chars.n_features)
        
        # Check that ratios are between 0 and 1
        self.assertGreaterEqual(chars.missing_ratio, 0)
        self.assertLessEqual(chars.missing_ratio, 1)
        self.assertGreaterEqual(chars.sparsity_ratio, 0)
        self.assertLessEqual(chars.sparsity_ratio, 1)


class TestAdaptivePreprocessorConfig(unittest.TestCase):
    """Test the AdaptivePreprocessorConfig class"""
    
    def setUp(self):
        self.config = AdaptivePreprocessorConfig()
    
    def test_default_configuration(self):
        """Test default configuration values"""
        self.assertTrue(self.config.auto_normalization)
        self.assertTrue(self.config.adaptive_missing_handling)
        self.assertTrue(self.config.smart_categorical_encoding)
        self.assertTrue(self.config.dynamic_outlier_detection)
        self.assertTrue(self.config.memory_aware_processing)
        
        # Test thresholds
        self.assertGreater(self.config.large_dataset_threshold, 0)
        self.assertGreater(self.config.high_cardinality_threshold, 0)
        self.assertGreater(self.config.sparse_data_threshold, 0)
    
    def test_strategy_determination(self):
        """Test preprocessing strategy determination"""
        # Test small dataset
        small_chars = DatasetCharacteristics(
            n_samples=500,
            n_features=10,
            n_categorical=2,
            n_numerical=8,
            memory_usage_mb=5.0
        )
        strategy = self.config.determine_strategy(small_chars)
        self.assertEqual(strategy, PreprocessingStrategy.STANDARD)
        
        # Test large dataset
        large_chars = DatasetCharacteristics(
            n_samples=100_000,
            n_features=50,
            n_categorical=10,
            n_numerical=40,
            memory_usage_mb=500.0
        )
        strategy = self.config.determine_strategy(large_chars)
        self.assertEqual(strategy, PreprocessingStrategy.MEMORY_OPTIMIZED)
        
        # Test high cardinality dataset
        high_card_chars = DatasetCharacteristics(
            n_samples=10_000,
            n_features=20,
            n_categorical=5,
            n_numerical=15,
            cardinality_avg=150.0  # High cardinality
        )
        strategy = self.config.determine_strategy(high_card_chars)
        self.assertEqual(strategy, PreprocessingStrategy.CATEGORICAL_OPTIMIZED)
    
    def test_normalization_method_selection(self):
        """Test automatic normalization method selection"""
        # Test dataset with high outlier ratio - should choose robust
        outlier_chars = DatasetCharacteristics(
            n_samples=1000,
            n_features=10,
            outlier_ratio=0.15  # High outlier ratio
        )
        method = self.config.select_normalization_method(outlier_chars)
        self.assertEqual(method, NormalizationType.ROBUST)
        
        # Test dataset with high skewness - should choose robust
        skewed_chars = DatasetCharacteristics(
            n_samples=1000,
            n_features=10,
            skewness_avg=3.5  # High skewness
        )
        method = self.config.select_normalization_method(skewed_chars)
        self.assertEqual(method, NormalizationType.ROBUST)
        
        # Test normal dataset - should choose standard
        normal_chars = DatasetCharacteristics(
            n_samples=1000,
            n_features=10,
            outlier_ratio=0.02,
            skewness_avg=0.5
        )
        method = self.config.select_normalization_method(normal_chars)
        self.assertEqual(method, NormalizationType.STANDARD)


class TestConfigOptimizer(unittest.TestCase):
    """Test the ConfigOptimizer class"""
    
    def setUp(self):
        self.optimizer = ConfigOptimizer()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataframe(self, characteristics: dict) -> pd.DataFrame:
        """Create a test DataFrame with specified characteristics"""
        np.random.seed(42)
        
        n_samples = characteristics.get('n_samples', 1000)
        n_numerical = characteristics.get('n_numerical', 5)
        n_categorical = characteristics.get('n_categorical', 2)
        missing_ratio = characteristics.get('missing_ratio', 0.05)
        outlier_ratio = characteristics.get('outlier_ratio', 0.02)
        
        data = {}
        
        # Create numerical columns
        for i in range(n_numerical):
            col_data = np.random.randn(n_samples)
            
            # Add outliers
            if outlier_ratio > 0:
                n_outliers = int(n_samples * outlier_ratio)
                outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
                col_data[outlier_indices] = np.random.randn(n_outliers) * 10 + 50
            
            data[f'num_col_{i}'] = col_data
        
        # Create categorical columns
        for i in range(n_categorical):
            cardinality = characteristics.get('cardinality', 5)
            categories = [f'cat_{j}' for j in range(cardinality)]
            data[f'cat_col_{i}'] = np.random.choice(categories, n_samples)
        
        df = pd.DataFrame(data)
        
        # Add missing values
        if missing_ratio > 0:
            total_values = df.size
            n_missing = int(total_values * missing_ratio)
            
            for _ in range(n_missing):
                row_idx = np.random.randint(0, n_samples)
                col_idx = np.random.randint(0, len(df.columns))
                df.iloc[row_idx, col_idx] = np.nan
        
        return df
    
    def test_analyze_dataset_characteristics(self):
        """Test dataset characteristics analysis"""
        # Create test dataset
        df = self.create_test_dataframe({
            'n_samples': 1000,
            'n_numerical': 5,
            'n_categorical': 2,
            'missing_ratio': 0.1,
            'outlier_ratio': 0.05
        })
        
        chars = self.optimizer.analyze_dataset_characteristics(df)
        
        # Verify characteristics
        self.assertIsInstance(chars, DatasetCharacteristics)
        self.assertEqual(chars.n_samples, 1000)
        self.assertEqual(chars.n_features, 7)
        self.assertEqual(chars.n_numerical, 5)
        self.assertEqual(chars.n_categorical, 2)
        self.assertGreater(chars.missing_ratio, 0)
        self.assertGreater(chars.memory_usage_mb, 0)
    
    def test_optimize_for_dataset(self):
        """Test optimization for specific dataset"""
        # Create small dataset
        small_df = self.create_test_dataframe({
            'n_samples': 500,
            'n_numerical': 3,
            'n_categorical': 2,
            'missing_ratio': 0.02
        })
        
        config = self.optimizer.optimize_for_dataset(small_df)
        
        self.assertIsInstance(config, dict)
        self.assertIn('normalization', config)
        self.assertIn('strategy', config)
        self.assertIn('characteristics', config)
        
        # Create large dataset
        large_df = self.create_test_dataframe({
            'n_samples': 50_000,
            'n_numerical': 20,
            'n_categorical': 5,
            'missing_ratio': 0.1
        })
        
        config = self.optimizer.optimize_for_dataset(large_df)
        self.assertEqual(config['strategy'], PreprocessingStrategy.MEMORY_OPTIMIZED)
    
    def test_suggest_chunk_size(self):
        """Test chunk size suggestion"""
        # Small dataset
        small_chars = DatasetCharacteristics(
            n_samples=1000,
            memory_usage_mb=10.0
        )
        chunk_size = self.optimizer.suggest_chunk_size(small_chars)
        self.assertGreaterEqual(chunk_size, 1000)  # Should handle all at once
        
        # Large dataset
        large_chars = DatasetCharacteristics(
            n_samples=1_000_000,
            memory_usage_mb=1000.0
        )
        chunk_size = self.optimizer.suggest_chunk_size(large_chars)
        self.assertLess(chunk_size, 1_000_000)  # Should be chunked
        self.assertGreater(chunk_size, 0)
    
    def test_estimate_processing_time(self):
        """Test processing time estimation"""
        chars = DatasetCharacteristics(
            n_samples=10_000,
            n_features=20,
            memory_usage_mb=100.0
        )
        
        time_estimate = self.optimizer.estimate_processing_time(chars)
        
        self.assertIsInstance(time_estimate, dict)
        self.assertIn('total_seconds', time_estimate)
        self.assertIn('loading_seconds', time_estimate)
        self.assertIn('preprocessing_seconds', time_estimate)
        self.assertIn('memory_optimization_seconds', time_estimate)
        
        # All estimates should be positive
        for key, value in time_estimate.items():
            self.assertGreaterEqual(value, 0)
    
    def test_memory_optimization_suggestions(self):
        """Test memory optimization suggestions"""
        # High memory dataset
        high_mem_chars = DatasetCharacteristics(
            n_samples=100_000,
            n_features=50,
            memory_usage_mb=2000.0,
            sparsity_ratio=0.3
        )
        
        suggestions = self.optimizer.get_memory_optimization_suggestions(high_mem_chars)
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Should suggest chunked processing for large datasets
        suggestion_text = ' '.join(suggestions)
        self.assertIn('chunk', suggestion_text.lower())
    
    def test_skewness_calculation(self):
        """Test skewness calculation for dataset characteristics"""
        # Create dataset with known skewness
        df = pd.DataFrame({
            'normal': np.random.randn(1000),
            'skewed': np.random.exponential(2, 1000),  # Positively skewed
            'categorical': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        chars = self.optimizer.analyze_dataset_characteristics(df)
        
        # Should detect skewness in numerical columns
        self.assertIsInstance(chars.skewness_avg, float)
        # Exponential distribution should show positive skewness
        self.assertGreater(chars.skewness_avg, 0)
    
    def test_outlier_detection_analysis(self):
        """Test outlier detection in characteristics analysis"""
        # Create dataset with known outliers
        np.random.seed(42)
        normal_data = np.random.randn(1000)
        outlier_data = np.concatenate([
            normal_data[:950],
            np.array([10, 15, -12, 20, 25])  # Clear outliers
        ])
        
        df = pd.DataFrame({
            'data': outlier_data,
            'category': np.random.choice(['A', 'B'], 1000)
        })
        
        chars = self.optimizer.analyze_dataset_characteristics(df)
        
        # Should detect outliers
        self.assertGreater(chars.outlier_ratio, 0)
        self.assertLess(chars.outlier_ratio, 0.1)  # Should be reasonable


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for easy integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self, characteristics: dict) -> Path:
        """Create a test CSV file with specified characteristics"""
        np.random.seed(42)
        n_samples = characteristics.get('n_samples', 1000)
        
        df = pd.DataFrame({
            'numeric1': np.random.randn(n_samples),
            'numeric2': np.random.uniform(0, 100, n_samples),
            'category1': np.random.choice(['A', 'B', 'C'], n_samples),
            'category2': np.random.choice(['X', 'Y'], n_samples)
        })
        
        # Add missing values if specified
        missing_ratio = characteristics.get('missing_ratio', 0)
        if missing_ratio > 0:
            n_missing = int(n_samples * missing_ratio)
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            df.loc[missing_indices, 'numeric1'] = np.nan
        
        file_path = self.temp_path / "test.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_create_adaptive_config(self):
        """Test create_adaptive_config convenience function"""
        config = create_adaptive_config(
            auto_normalization=True,
            memory_aware=True,
            large_dataset_threshold=50000
        )
        
        self.assertIsInstance(config, AdaptivePreprocessorConfig)
        self.assertTrue(config.auto_normalization)
        self.assertTrue(config.memory_aware_processing)
        self.assertEqual(config.large_dataset_threshold, 50000)
    
    def test_optimize_preprocessing_config(self):
        """Test optimize_preprocessing_config convenience function"""
        csv_file = self.create_test_csv({
            'n_samples': 1000,
            'missing_ratio': 0.05
        })
        
        # Test with file path
        optimized_config = optimize_preprocessing_config(str(csv_file))
        
        self.assertIsInstance(optimized_config, dict)
        self.assertIn('normalization', optimized_config)
        self.assertIn('strategy', optimized_config)
        self.assertIn('characteristics', optimized_config)
        
        # Test with DataFrame
        df = pd.read_csv(csv_file)
        optimized_config_df = optimize_preprocessing_config(df)
        
        self.assertIsInstance(optimized_config_df, dict)
        self.assertEqual(
            optimized_config['strategy'],
            optimized_config_df['strategy']
        )


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios with different dataset types"""
    
    def setUp(self):
        self.optimizer = ConfigOptimizer()
    
    def test_small_clean_dataset(self):
        """Test optimization for small, clean dataset"""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(500),
            'feature2': np.random.uniform(0, 1, 500),
            'category': np.random.choice(['A', 'B'], 500)
        })
        
        config = self.optimizer.optimize_for_dataset(df)
        
        # Should use standard strategy for small, clean data
        self.assertEqual(config['strategy'], PreprocessingStrategy.STANDARD)
        self.assertEqual(config['normalization'], NormalizationType.STANDARD)
    
    def test_large_messy_dataset(self):
        """Test optimization for large dataset with missing values and outliers"""
        np.random.seed(42)
        n_samples = 100_000
        
        # Create messy data
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.exponential(2, n_samples),  # Skewed
            'feature3': np.random.randn(n_samples),
            'high_cardinality': ['item_' + str(i % 1000) for i in range(n_samples)],
            'low_cardinality': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Add outliers
        outlier_indices = np.random.choice(n_samples, 1000, replace=False)
        df.loc[outlier_indices, 'feature1'] = np.random.randn(1000) * 10 + 50
        
        # Add missing values
        missing_indices = np.random.choice(n_samples, 5000, replace=False)
        df.loc[missing_indices, 'feature2'] = np.nan
        
        config = self.optimizer.optimize_for_dataset(df)
        
        # Should use memory-optimized strategy for large datasets
        self.assertEqual(config['strategy'], PreprocessingStrategy.MEMORY_OPTIMIZED)
        
        # Should choose robust normalization due to outliers and skewness
        self.assertEqual(config['normalization'], NormalizationType.ROBUST)
        
        # Should suggest chunked processing
        suggestions = config.get('suggestions', [])
        suggestion_text = ' '.join(suggestions).lower()
        self.assertIn('chunk', suggestion_text)
    
    def test_high_cardinality_dataset(self):
        """Test optimization for dataset with high cardinality categorical features"""
        np.random.seed(42)
        n_samples = 10_000
        
        df = pd.DataFrame({
            'numeric1': np.random.randn(n_samples),
            'numeric2': np.random.randn(n_samples),
            'high_card1': ['cat_' + str(i % 500) for i in range(n_samples)],
            'high_card2': ['item_' + str(i % 300) for i in range(n_samples)],
            'low_card': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        config = self.optimizer.optimize_for_dataset(df)
        
        # Should detect high cardinality and suggest appropriate strategy
        chars = config['characteristics']
        self.assertGreater(chars.cardinality_avg, 100)  # High cardinality detected
        
        # Should suggest categorical optimization strategy
        self.assertEqual(config['strategy'], PreprocessingStrategy.CATEGORICAL_OPTIMIZED)
    
    def test_sparse_dataset(self):
        """Test optimization for sparse dataset"""
        np.random.seed(42)
        n_samples = 5_000
        n_features = 20
        
        # Create mostly zero data (sparse)
        data = np.random.randn(n_samples, n_features)
        # Make 80% of values zero
        zero_mask = np.random.choice([True, False], size=data.shape, p=[0.8, 0.2])
        data[zero_mask] = 0
        
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
        
        config = self.optimizer.optimize_for_dataset(df)
        
        # Should detect sparsity
        chars = config['characteristics']
        self.assertGreater(chars.sparsity_ratio, 0.5)  # High sparsity detected
        
        # Should suggest memory optimization due to sparsity
        suggestions = config.get('suggestions', [])
        suggestion_text = ' '.join(suggestions).lower()
        self.assertIn('sparse', suggestion_text)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        self.optimizer = ConfigOptimizer()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        try:
            chars = self.optimizer.analyze_dataset_characteristics(empty_df)
            self.assertEqual(chars.n_samples, 0)
            self.assertEqual(chars.n_features, 0)
        except ValueError:
            # Acceptable to raise ValueError for empty data
            pass
    
    def test_single_column_dataframe(self):
        """Test handling of single column DataFrame"""
        single_col_df = pd.DataFrame({'col1': range(100)})
        
        chars = self.optimizer.analyze_dataset_characteristics(single_col_df)
        
        self.assertEqual(chars.n_samples, 100)
        self.assertEqual(chars.n_features, 1)
        self.assertEqual(chars.n_numerical, 1)
        self.assertEqual(chars.n_categorical, 0)
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values"""
        df = pd.DataFrame({
            'good_col': range(100),
            'all_missing': [np.nan] * 100
        })
        
        chars = self.optimizer.analyze_dataset_characteristics(df)
        
        # Should handle gracefully
        self.assertEqual(chars.n_samples, 100)
        self.assertEqual(chars.n_features, 2)
        self.assertGreater(chars.missing_ratio, 0.4)  # High missing ratio
    
    def test_invalid_data_types(self):
        """Test handling of unusual data types"""
        df = pd.DataFrame({
            'dates': pd.date_range('2023-01-01', periods=100),
            'complex_numbers': [complex(i, i) for i in range(100)],
            'mixed_types': [1, 'string', 3.14, None] * 25
        })
        
        # Should handle gracefully without crashing
        try:
            chars = self.optimizer.analyze_dataset_characteristics(df)
            self.assertIsInstance(chars, DatasetCharacteristics)
        except Exception as e:
            # Should be a reasonable exception if it fails
            self.assertIsInstance(e, (ValueError, TypeError))


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main()
