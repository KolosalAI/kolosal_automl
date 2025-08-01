"""
Unit tests for the OptimizedDataLoader module.

This test suite covers:
- Dataset size estimation and categorization
- Loading strategy selection
- Memory optimization features
- Different data format handling
- Performance monitoring
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the modules to test
from modules.engine.optimized_data_loader import (
    OptimizedDataLoader,
    LoadingConfig,
    DatasetSize,
    LoadingStrategy,
    DatasetInfo,
    MemoryMonitor,
    create_optimized_loader,
    load_data_optimized
)


class TestMemoryMonitor(unittest.TestCase):
    """Test the MemoryMonitor class"""
    
    def setUp(self):
        self.monitor = MemoryMonitor()
    
    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization"""
        self.assertIsInstance(self.monitor.initial_memory, float)
        self.assertGreater(self.monitor.initial_memory, 0)
        self.assertEqual(self.monitor.peak_memory, self.monitor.initial_memory)
        self.assertEqual(len(self.monitor.memory_history), 0)
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        memory_usage = self.monitor.get_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertGreater(memory_usage, 0)
    
    def test_get_available_memory(self):
        """Test available memory retrieval"""
        available = self.monitor.get_available_memory()
        self.assertIsInstance(available, float)
        self.assertGreater(available, 0)
    
    def test_update_peak(self):
        """Test peak memory update"""
        initial_peak = self.monitor.peak_memory
        current = self.monitor.update_peak()
        
        self.assertIsInstance(current, float)
        self.assertGreaterEqual(self.monitor.peak_memory, initial_peak)
        self.assertEqual(len(self.monitor.memory_history), 1)
    
    def test_memory_pressure_calculation(self):
        """Test memory pressure calculation"""
        pressure = self.monitor.get_memory_pressure()
        self.assertIsInstance(pressure, float)
        self.assertGreaterEqual(pressure, 0)
        self.assertLessEqual(pressure, 200)  # Allow for some overhead
    
    def test_gc_trigger_threshold(self):
        """Test garbage collection trigger logic"""
        # Test with low threshold (should trigger)
        should_trigger = self.monitor.should_trigger_gc(threshold_pct=0.1)
        self.assertIsInstance(should_trigger, bool)
        
        # Test with high threshold (should not trigger)
        should_not_trigger = self.monitor.should_trigger_gc(threshold_pct=99.9)
        self.assertFalse(should_not_trigger)


class TestLoadingConfig(unittest.TestCase):
    """Test the LoadingConfig dataclass"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = LoadingConfig()
        
        self.assertEqual(config.strategy, LoadingStrategy.DIRECT)
        self.assertEqual(config.chunk_size, 10000)
        self.assertEqual(config.max_memory_usage_pct, 70.0)
        self.assertTrue(config.enable_memory_mapping)
        self.assertTrue(config.enable_compression)
        self.assertTrue(config.use_categorical_optimization)
        self.assertTrue(config.dtype_optimization)
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = LoadingConfig(
            strategy=LoadingStrategy.CHUNKED,
            chunk_size=5000,
            max_memory_usage_pct=80.0,
            use_polars=False,
            use_dask=False
        )
        
        self.assertEqual(config.strategy, LoadingStrategy.CHUNKED)
        self.assertEqual(config.chunk_size, 5000)
        self.assertEqual(config.max_memory_usage_pct, 80.0)
        self.assertFalse(config.use_polars)
        self.assertFalse(config.use_dask)


class TestOptimizedDataLoader(unittest.TestCase):
    """Test the main OptimizedDataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = LoadingConfig(
            chunk_size=1000,
            max_memory_usage_pct=70.0,
            use_polars=False,  # Disable for testing
            use_dask=False     # Disable for testing
        )
        self.loader = OptimizedDataLoader(self.config)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self, rows: int = 1000, cols: int = 5, 
                       filename: str = "test.csv") -> Path:
        """Create a test CSV file with specified dimensions"""
        np.random.seed(42)
        data = np.random.randn(rows, cols)
        df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])
        
        # Add some categorical data
        df['category'] = np.random.choice(['A', 'B', 'C'], size=rows)
        
        # Add some missing values
        if rows > 100:
            missing_indices = np.random.choice(rows, size=rows//20, replace=False)
            df.iloc[missing_indices, 0] = np.nan
        
        file_path = self.temp_path / filename
        df.to_csv(file_path, index=False)
        return file_path
    
    def create_test_excel(self, rows: int = 500, filename: str = "test.xlsx") -> Path:
        """Create a test Excel file"""
        np.random.seed(42)
        data = {
            'numbers': np.random.randn(rows),
            'integers': np.random.randint(0, 100, rows),
            'categories': np.random.choice(['X', 'Y', 'Z'], size=rows),
            'booleans': np.random.choice([True, False], size=rows)
        }
        df = pd.DataFrame(data)
        
        file_path = self.temp_path / filename
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        self.assertIsInstance(self.loader.config, LoadingConfig)
        self.assertIsInstance(self.loader.memory_monitor, MemoryMonitor)
        self.assertIsNotNone(self.loader.logger)
        self.assertIn('.csv', self.loader._readers)
        self.assertIn('.xlsx', self.loader._readers)
    
    def test_dataset_size_categorization(self):
        """Test dataset size categorization"""
        test_cases = [
            (5_000, DatasetSize.TINY),
            (50_000, DatasetSize.SMALL),
            (500_000, DatasetSize.MEDIUM),
            (5_000_000, DatasetSize.LARGE),
            (50_000_000, DatasetSize.HUGE)
        ]
        
        for rows, expected_size in test_cases:
            with self.subTest(rows=rows):
                size = self.loader._categorize_dataset_size(rows)
                self.assertEqual(size, expected_size)
    
    def test_csv_size_estimation(self):
        """Test CSV file size estimation"""
        # Create test files of different sizes
        small_file = self.create_test_csv(rows=100, filename="small.csv")
        medium_file = self.create_test_csv(rows=10_000, filename="medium.csv")
        
        # Test small file
        size_cat, memory_est = self.loader.estimate_file_complexity(str(small_file))
        self.assertEqual(size_cat, DatasetSize.TINY)
        self.assertIsInstance(memory_est, float)
        self.assertGreater(memory_est, 0)
        
        # Test medium file
        size_cat, memory_est = self.loader.estimate_file_complexity(str(medium_file))
        self.assertEqual(size_cat, DatasetSize.SMALL)
        self.assertIsInstance(memory_est, float)
        self.assertGreater(memory_est, 0)
    
    def test_excel_size_estimation(self):
        """Test Excel file size estimation"""
        excel_file = self.create_test_excel(rows=500)
        
        size_cat, memory_est = self.loader.estimate_file_complexity(str(excel_file))
        self.assertIn(size_cat, [DatasetSize.TINY, DatasetSize.SMALL])
        self.assertIsInstance(memory_est, float)
        self.assertGreater(memory_est, 0)
    
    def test_loading_strategy_selection(self):
        """Test loading strategy selection logic"""
        # Test tiny dataset - should use DIRECT
        strategy = self.loader.select_loading_strategy(
            DatasetSize.TINY, estimated_memory_mb=10.0
        )
        self.assertEqual(strategy, LoadingStrategy.DIRECT)
        
        # Test large dataset with high memory - should not use DIRECT
        strategy = self.loader.select_loading_strategy(
            DatasetSize.LARGE, estimated_memory_mb=10000.0
        )
        self.assertNotEqual(strategy, LoadingStrategy.DIRECT)
        
        # Test huge dataset - should use advanced strategy
        strategy = self.loader.select_loading_strategy(
            DatasetSize.HUGE, estimated_memory_mb=50000.0
        )
        self.assertIn(strategy, [LoadingStrategy.STREAMING, LoadingStrategy.DISTRIBUTED, 
                                LoadingStrategy.MEMORY_MAPPED])
    
    def test_dtype_optimization(self):
        """Test data type optimization"""
        # Create DataFrame with inefficient dtypes
        df = pd.DataFrame({
            'int_col': pd.Series([1, 2, 3, 4, 5], dtype='int64'),
            'float_col': pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64'),
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = self.loader._optimize_dtypes(df)
        new_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Memory should be reduced or stay the same
        self.assertLessEqual(new_memory, original_memory)
        
        # DataFrame should have same shape and values
        self.assertEqual(optimized_df.shape, df.shape)
        pd.testing.assert_frame_equal(optimized_df.astype(str), df.astype(str))
    
    def test_categorical_optimization(self):
        """Test categorical optimization"""
        # Create DataFrame with repetitive string data
        df = pd.DataFrame({
            'high_cardinality': ['unique_' + str(i) for i in range(100)],
            'low_cardinality': ['category_' + str(i % 3) for i in range(100)],
            'numeric': range(100)
        })
        
        optimized_df = self.loader._optimize_categorical(df)
        
        # High cardinality column should remain object
        self.assertEqual(optimized_df['high_cardinality'].dtype.name, 'object')
        
        # Low cardinality column should become category
        self.assertEqual(optimized_df['low_cardinality'].dtype.name, 'category')
        
        # Numeric column should remain unchanged
        self.assertEqual(optimized_df['numeric'].dtype, df['numeric'].dtype)
    
    def test_missing_data_analysis(self):
        """Test missing data analysis"""
        # Create DataFrame with missing values
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1, np.nan, np.nan, 4, 5],
            'col3': [1, 2, 3, 4, 5]  # No missing values
        })
        
        missing_info = self.loader._analyze_missing_data(df)
        
        self.assertIsInstance(missing_info, dict)
        self.assertIn('total_missing', missing_info)
        self.assertIn('missing_percentage', missing_info)
        self.assertIn('columns_with_missing', missing_info)
        self.assertIn('missing_percentages', missing_info)
        
        # Check values
        self.assertEqual(missing_info['total_missing'], 3)
        self.assertIn('col1', missing_info['columns_with_missing'])
        self.assertIn('col2', missing_info['columns_with_missing'])
        self.assertNotIn('col3', missing_info['columns_with_missing'])
    
    def test_load_small_csv_direct(self):
        """Test loading small CSV file with direct strategy"""
        csv_file = self.create_test_csv(rows=100, cols=3)
        
        df, dataset_info = self.loader.load_data(str(csv_file))
        
        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 4)  # 3 numeric + 1 categorical
        
        # Check dataset info
        self.assertIsInstance(dataset_info, DatasetInfo)
        self.assertEqual(dataset_info.size_category, DatasetSize.TINY)
        self.assertEqual(dataset_info.loading_strategy, LoadingStrategy.DIRECT)
        self.assertEqual(dataset_info.rows, 100)
        self.assertEqual(dataset_info.columns, 4)
        self.assertIsInstance(dataset_info.loading_time, float)
        self.assertGreater(dataset_info.loading_time, 0)
        self.assertIsInstance(dataset_info.optimization_applied, list)
    
    @patch('modules.engine.optimized_data_loader.POLARS_AVAILABLE', False)
    @patch('modules.engine.optimized_data_loader.DASK_AVAILABLE', False)
    def test_load_medium_csv_chunked(self):
        """Test loading medium CSV file with chunked strategy"""
        # Force chunked strategy by setting low memory threshold
        config = LoadingConfig(
            max_memory_usage_pct=0.01,  # Very low threshold
            chunk_size=50
        )
        loader = OptimizedDataLoader(config)
        
        csv_file = self.create_test_csv(rows=200, cols=3)
        
        df, dataset_info = loader.load_data(str(csv_file))
        
        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 200)
        
        # Check dataset info
        self.assertIsInstance(dataset_info, DatasetInfo)
        self.assertEqual(dataset_info.rows, 200)
        self.assertIn('chunked_loading', dataset_info.optimization_applied)
    
    def test_load_excel_file(self):
        """Test loading Excel files"""
        excel_file = self.create_test_excel(rows=50)
        
        df, dataset_info = self.loader.load_data(str(excel_file))
        
        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        self.assertIn('numbers', df.columns)
        self.assertIn('categories', df.columns)
        
        # Check dataset info
        self.assertIsInstance(dataset_info, DatasetInfo)
        self.assertEqual(dataset_info.rows, 50)
    
    def test_file_not_found_error(self):
        """Test handling of non-existent files"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_data("non_existent_file.csv")
    
    def test_unsupported_format_error(self):
        """Test handling of unsupported file formats"""
        # Create a file with unsupported extension
        unsupported_file = self.temp_path / "test.xyz"
        unsupported_file.write_text("some content")
        
        with self.assertRaises(ValueError):
            self.loader.load_data(str(unsupported_file))
    
    def test_post_loading_optimizations(self):
        """Test post-loading optimization pipeline"""
        # Create DataFrame that can benefit from optimization
        df = pd.DataFrame({
            'integers': pd.Series(range(100), dtype='int64'),
            'floats': pd.Series(np.random.randn(100), dtype='float64'),
            'categories': ['cat_' + str(i % 5) for i in range(100)]
        })
        
        optimized_df, optimizations = self.loader._apply_post_loading_optimizations(df)
        
        # Check that optimizations were applied
        self.assertIsInstance(optimizations, list)
        self.assertIn('dtype_optimization', optimizations)
        self.assertIn('categorical_optimization', optimizations)
        
        # Check that DataFrame structure is preserved
        self.assertEqual(optimized_df.shape, df.shape)
        self.assertEqual(list(optimized_df.columns), list(df.columns))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for easy integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self, rows: int = 100) -> Path:
        """Create a simple test CSV file"""
        df = pd.DataFrame({
            'x': range(rows),
            'y': np.random.randn(rows),
            'category': ['A'] * (rows // 2) + ['B'] * (rows - rows // 2)
        })
        
        file_path = self.temp_path / "test.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_create_optimized_loader(self):
        """Test create_optimized_loader convenience function"""
        loader = create_optimized_loader(
            max_memory_pct=80.0,
            chunk_size=5000,
            enable_distributed=False
        )
        
        self.assertIsInstance(loader, OptimizedDataLoader)
        self.assertEqual(loader.config.max_memory_usage_pct, 80.0)
        self.assertEqual(loader.config.chunk_size, 5000)
        self.assertFalse(loader.config.use_dask)
    
    def test_load_data_optimized(self):
        """Test load_data_optimized convenience function"""
        csv_file = self.create_test_csv(rows=50)
        
        df, dataset_info = load_data_optimized(
            str(csv_file),
            max_memory_pct=75.0
        )
        
        # Check results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(dataset_info, DatasetInfo)
        self.assertEqual(len(df), 50)
        self.assertEqual(dataset_info.rows, 50)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        self.loader = OptimizedDataLoader()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_corrupted_csv_handling(self):
        """Test handling of corrupted CSV files"""
        # Create a corrupted CSV file
        corrupted_file = self.temp_path / "corrupted.csv"
        corrupted_file.write_text("header1,header2\nvalue1,value2,extra_value\ninvalid_row")
        
        # Should handle gracefully or raise appropriate exception
        try:
            df, info = self.loader.load_data(str(corrupted_file))
            # If it succeeds, check that we got some data
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            # If it fails, should be a reasonable exception
            self.assertIsInstance(e, (pd.errors.Error, ValueError))
    
    def test_empty_file_handling(self):
        """Test handling of empty files"""
        empty_file = self.temp_path / "empty.csv"
        empty_file.write_text("")
        
        # Should handle gracefully
        try:
            df, info = self.loader.load_data(str(empty_file))
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            # Should be a reasonable exception
            self.assertIsInstance(e, (pd.errors.Error, ValueError))
    
    def test_large_file_memory_management(self):
        """Test memory management with larger files"""
        # Create a moderately large CSV file
        large_data = pd.DataFrame({
            'col1': range(10000),
            'col2': np.random.randn(10000),
            'col3': ['text_' + str(i) for i in range(10000)]
        })
        
        large_file = self.temp_path / "large.csv"
        large_data.to_csv(large_file, index=False)
        
        # Monitor memory before loading
        initial_memory = self.loader.memory_monitor.get_memory_usage()
        
        df, info = self.loader.load_data(str(large_file))
        
        # Check that data was loaded correctly
        self.assertEqual(len(df), 10000)
        self.assertEqual(info.rows, 10000)
        
        # Memory should be reasonable (not excessive growth)
        final_memory = self.loader.memory_monitor.get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Allow up to 500MB growth for this test (generous limit)
        self.assertLess(memory_growth, 500)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main()
