"""
Unit tests for the Optimization Integration module.

This test suite covers:
- OptimizedDataPipeline functionality
- Integration utility functions
- End-to-end optimization workflows
- Cross-module integration
- Error handling and fallbacks
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from modules.engine.optimization_integration import (
    OptimizedDataPipeline,
    create_optimized_training_pipeline,
    quick_load_optimized,
    quick_optimize_memory,
    quick_preprocessor_optimized,
    get_optimization_status
)

# Import related modules
try:
    from modules.engine.optimized_data_loader import DatasetInfo, LoadingStrategy
    from modules.engine.adaptive_preprocessing import DatasetCharacteristics
    OPTIMIZATION_MODULES_AVAILABLE = True
except ImportError:
    OPTIMIZATION_MODULES_AVAILABLE = False


class TestOptimizedDataPipeline(unittest.TestCase):
    """Test the main OptimizedDataPipeline class"""
    
    def setUp(self):
        self.pipeline = OptimizedDataPipeline(
            max_memory_pct=75.0,
            enable_memory_optimization=True,
            enable_adaptive_preprocessing=True
        )
        
        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self, rows: int = 1000, cols: int = 5, 
                       filename: str = "test.csv") -> Path:
        """Create a test CSV file"""
        np.random.seed(42)
        data = {}
        
        # Numerical columns
        for i in range(cols):
            data[f'feature_{i}'] = np.random.randn(rows)
        
        # Categorical column
        data['category'] = np.random.choice(['A', 'B', 'C'], rows)
        
        # Target column
        data['target'] = np.random.randint(0, 2, rows)
        
        df = pd.DataFrame(data)
        file_path = self.temp_path / filename
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.max_memory_pct, 75.0)
        self.assertTrue(self.pipeline.enable_memory_optimization)
        self.assertTrue(self.pipeline.enable_adaptive_preprocessing)
        self.assertIsNotNone(self.pipeline.logger)
    
    @unittest.skipUnless(OPTIMIZATION_MODULES_AVAILABLE, "Optimization modules not available")
    def test_load_data_with_optimization(self):
        """Test data loading with optimization"""
        csv_file = self.create_test_csv(rows=500, cols=3)
        
        result = self.pipeline.load_data(str(csv_file))
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('data', result)
        self.assertIn('dataset_info', result)
        
        if result['success']:
            df = result['data']
            dataset_info = result['dataset_info']
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 500)
            
            if OPTIMIZATION_MODULES_AVAILABLE:
                self.assertIsInstance(dataset_info, DatasetInfo)
                self.assertEqual(dataset_info.rows, 500)
    
    def test_load_data_fallback(self):
        """Test data loading fallback when optimization unavailable"""
        csv_file = self.create_test_csv(rows=100, cols=2)
        
        # Test fallback by mocking unavailable optimizer
        with patch.object(self.pipeline, 'data_loader', None):
            result = self.pipeline.load_data(str(csv_file))
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            
            if result['success']:
                df = result['data']
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(len(df), 100)
    
    @unittest.skipUnless(OPTIMIZATION_MODULES_AVAILABLE, "Optimization modules not available")  
    def test_optimize_preprocessing(self):
        """Test preprocessing optimization"""
        csv_file = self.create_test_csv(rows=1000, cols=5)
        df = pd.read_csv(csv_file)
        
        result = self.pipeline.optimize_preprocessing(df, target_column='target')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('config', result)
            self.assertIn('recommendations', result)
            
            config = result['config']
            self.assertIsInstance(config, dict)
    
    def test_optimize_preprocessing_fallback(self):
        """Test preprocessing optimization fallback"""
        csv_file = self.create_test_csv(rows=100, cols=3)
        df = pd.read_csv(csv_file)
        
        # Test fallback by mocking unavailable config optimizer
        with patch.object(self.pipeline, 'config_optimizer', None):
            result = self.pipeline.optimize_preprocessing(df, target_column='target')
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertTrue(result['success'])  # Should succeed with fallback
    
    def test_optimize_memory(self):
        """Test memory optimization"""
        # Create DataFrame with inefficient dtypes
        df = pd.DataFrame({
            'int_col': pd.Series(range(1000), dtype='int64'),
            'float_col': pd.Series(np.random.randn(1000), dtype='float64'),
            'string_col': ['category_' + str(i % 5) for i in range(1000)]
        })
        
        result = self.pipeline.optimize_memory(df)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('optimized_data', result)
        
        if result['success']:
            optimized_df = result['optimized_data']
            self.assertIsInstance(optimized_df, pd.DataFrame)
            self.assertEqual(optimized_df.shape, df.shape)
            
            # Should provide optimization info
            self.assertIn('optimization_info', result)
            opt_info = result['optimization_info']
            self.assertIn('memory', opt_info)
    
    def test_process_complete_pipeline(self):
        """Test complete pipeline processing"""
        csv_file = self.create_test_csv(rows=500, cols=4)
        
        result = self.pipeline.process_complete_pipeline(
            file_path=str(csv_file),
            target_column='target',
            fit_preprocessor=True
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('data', result)
            self.assertIn('preprocessing_results', result)
            self.assertIn('memory_optimization_results', result)
            
            df = result['data']
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 500)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        # Test with non-existent file
        result = self.pipeline.load_data("non_existent_file.csv")
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_get_pipeline_status(self):
        """Test pipeline status reporting"""
        status = self.pipeline.get_pipeline_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('optimization_available', status)
        self.assertIn('components_loaded', status)
        self.assertIn('configuration', status)
        
        # Configuration should include pipeline settings
        config = status['configuration']
        self.assertEqual(config['max_memory_pct'], 75.0)
        self.assertTrue(config['enable_memory_optimization'])
        self.assertTrue(config['enable_adaptive_preprocessing'])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for quick integration"""
    
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
            'category': ['A'] * (rows // 2) + ['B'] * (rows - rows // 2),
            'target': np.random.randint(0, 2, rows)
        })
        
        file_path = self.temp_path / "test.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_create_optimized_training_pipeline(self):
        """Test create_optimized_training_pipeline function"""
        pipeline = create_optimized_training_pipeline(
            max_memory_pct=80.0,
            enable_memory_optimization=True,
            enable_adaptive_preprocessing=False
        )
        
        self.assertIsInstance(pipeline, OptimizedDataPipeline)
        self.assertEqual(pipeline.max_memory_pct, 80.0)
        self.assertTrue(pipeline.enable_memory_optimization)
        self.assertFalse(pipeline.enable_adaptive_preprocessing)
    
    def test_quick_load_optimized(self):
        """Test quick_load_optimized convenience function"""
        csv_file = self.create_test_csv(rows=50)
        
        result = quick_load_optimized(str(csv_file))
        
        # Should return tuple (df, info)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        df, info = result
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        self.assertIsInstance(info, dict)
    
    def test_quick_optimize_memory(self):
        """Test quick_optimize_memory convenience function"""
        # Create DataFrame with inefficient dtypes
        df = pd.DataFrame({
            'int_col': pd.Series(range(100), dtype='int64'),
            'str_col': ['category_' + str(i % 3) for i in range(100)]
        })
        
        optimized_df = quick_optimize_memory(df)
        
        self.assertIsInstance(optimized_df, pd.DataFrame)
        self.assertEqual(optimized_df.shape, df.shape)
        
        # Should preserve data while optimizing memory
        pd.testing.assert_frame_equal(
            optimized_df.astype(str), 
            df.astype(str), 
            check_dtype=False
        )
    
    def test_quick_preprocessor_optimized(self):
        """Test quick_preprocessor_optimized convenience function"""
        csv_file = self.create_test_csv(rows=100)
        df = pd.read_csv(csv_file)
        
        config = quick_preprocessor_optimized(df, target_column='target')
        
        self.assertIsInstance(config, dict)
        # Should contain optimization recommendations
    
    def test_get_optimization_status(self):
        """Test get_optimization_status function"""
        status = get_optimization_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('optimization_available', status)
        self.assertIn('modules_loaded', status)
        
        # Should be boolean
        self.assertIsInstance(status['optimization_available'], bool)
        self.assertIsInstance(status['modules_loaded'], dict)


class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_dataset(self, scenario: str) -> Path:
        """Create realistic datasets for different scenarios"""
        np.random.seed(42)
        
        if scenario == "small_clean":
            # Small, clean dataset
            n_samples = 500
            df = pd.DataFrame({
                'feature1': np.random.randn(n_samples),
                'feature2': np.random.uniform(0, 100, n_samples),
                'category': np.random.choice(['A', 'B', 'C'], n_samples),
                'target': np.random.randint(0, 2, n_samples)
            })
            
        elif scenario == "medium_messy":
            # Medium dataset with missing values and mixed types
            n_samples = 5000
            df = pd.DataFrame({
                'numeric1': np.random.randn(n_samples),
                'numeric2': np.random.exponential(2, n_samples),
                'high_cardinality': ['item_' + str(i % 500) for i in range(n_samples)],
                'low_cardinality': np.random.choice(['X', 'Y', 'Z'], n_samples),
                'target': np.random.randint(0, 3, n_samples)
            })
            
            # Add missing values
            missing_indices = np.random.choice(n_samples, n_samples // 20, replace=False)
            df.loc[missing_indices, 'numeric1'] = np.nan
            
        elif scenario == "large_sparse":
            # Large dataset with sparse features
            n_samples = 20000
            n_features = 50
            
            # Create mostly zero data
            data = np.random.randn(n_samples, n_features)
            zero_mask = np.random.choice([True, False], size=data.shape, p=[0.8, 0.2])
            data[zero_mask] = 0
            
            df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
            df['target'] = np.random.randint(0, 2, n_samples)
            
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        file_path = self.temp_path / f"{scenario}.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_small_clean_dataset_workflow(self):
        """Test workflow with small, clean dataset"""
        dataset_file = self.create_realistic_dataset("small_clean")
        pipeline = OptimizedDataPipeline()
        
        # Complete workflow
        result = pipeline.process_complete_pipeline(
            file_path=str(dataset_file),
            target_column='target',
            fit_preprocessor=True
        )
        
        self.assertTrue(result['success'])
        
        df = result['data']
        self.assertEqual(len(df), 500)
        
        # Small dataset should use direct loading
        if 'dataset_info' in result and OPTIMIZATION_MODULES_AVAILABLE:
            self.assertEqual(result['dataset_info'].loading_strategy, LoadingStrategy.DIRECT)
    
    def test_medium_messy_dataset_workflow(self):
        """Test workflow with medium, messy dataset"""
        dataset_file = self.create_realistic_dataset("medium_messy")
        pipeline = OptimizedDataPipeline(max_memory_pct=70.0)
        
        # Load data
        load_result = pipeline.load_data(str(dataset_file))
        self.assertTrue(load_result['success'])
        
        df = load_result['data']
        self.assertEqual(len(df), 5000)
        
        # Optimize preprocessing
        preprocess_result = pipeline.optimize_preprocessing(df, target_column='target')
        self.assertTrue(preprocess_result['success'])
        
        # Should provide recommendations for messy data
        if 'recommendations' in preprocess_result:
            recommendations = preprocess_result['recommendations']
            self.assertIsInstance(recommendations, list)
    
    def test_large_sparse_dataset_workflow(self):
        """Test workflow with large, sparse dataset"""
        dataset_file = self.create_realistic_dataset("large_sparse")
        pipeline = OptimizedDataPipeline(
            max_memory_pct=60.0,  # Lower threshold for large data
            enable_memory_optimization=True
        )
        
        # Process complete pipeline
        result = pipeline.process_complete_pipeline(
            file_path=str(dataset_file),
            target_column='target'
        )
        
        self.assertTrue(result['success'])
        
        df = result['data']
        self.assertEqual(len(df), 20000)
        
        # Large dataset should trigger memory optimization
        if 'memory_optimization_results' in result:
            mem_results = result['memory_optimization_results']
            self.assertTrue(mem_results['success'])
            
            # Should show memory reduction
            if 'optimization_info' in mem_results:
                opt_info = mem_results['optimization_info']
                self.assertIn('memory', opt_info)
    
    def test_cross_module_integration(self):
        """Test integration between all optimization modules"""
        dataset_file = self.create_realistic_dataset("medium_messy")
        
        # Test that all modules work together
        pipeline = OptimizedDataPipeline(
            max_memory_pct=75.0,
            enable_memory_optimization=True,
            enable_adaptive_preprocessing=True
        )
        
        # Step 1: Load with optimization
        load_result = pipeline.load_data(str(dataset_file))
        self.assertTrue(load_result['success'])
        
        # Step 2: Optimize preprocessing
        df = load_result['data']
        preprocess_result = pipeline.optimize_preprocessing(df, target_column='target')
        self.assertTrue(preprocess_result['success'])
        
        # Step 3: Optimize memory
        memory_result = pipeline.optimize_memory(df)
        self.assertTrue(memory_result['success'])
        
        # All steps should complete successfully
        optimized_df = memory_result['optimized_data']
        self.assertEqual(optimized_df.shape, df.shape)


class TestErrorHandlingAndFallbacks(unittest.TestCase):
    """Test error handling and fallback mechanisms"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_optimization_modules_fallback(self):
        """Test fallback when optimization modules are not available"""
        pipeline = OptimizedDataPipeline()
        
        # Mock missing modules
        with patch.object(pipeline, 'data_loader', None), \
             patch.object(pipeline, 'memory_processor', None), \
             patch.object(pipeline, 'config_optimizer', None):
            
            # Create simple test data
            df = pd.DataFrame({'x': range(100), 'y': range(100)})
            test_file = self.temp_path / "test.csv"
            df.to_csv(test_file, index=False)
            
            # Should still work with fallbacks
            result = pipeline.load_data(str(test_file))
            self.assertTrue(result['success'])
            
            memory_result = pipeline.optimize_memory(df)
            self.assertTrue(memory_result['success'])
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted files"""
        # Create corrupted file
        corrupted_file = self.temp_path / "corrupted.csv"
        corrupted_file.write_text("invalid,csv,content\nno,proper,structure")
        
        pipeline = OptimizedDataPipeline()
        
        result = pipeline.load_data(str(corrupted_file))
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if not result['success']:
            self.assertIn('error', result)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        pipeline = OptimizedDataPipeline()
        empty_df = pd.DataFrame()
        
        # Memory optimization should handle empty DataFrames
        result = pipeline.optimize_memory(empty_df)
        self.assertTrue(result['success'])
        
        optimized_df = result['optimized_data']
        self.assertEqual(len(optimized_df), 0)
    
    def test_invalid_target_column(self):
        """Test handling of invalid target column"""
        df = pd.DataFrame({'x': range(100), 'y': range(100)})
        pipeline = OptimizedDataPipeline()
        
        # Should handle invalid target column gracefully
        result = pipeline.optimize_preprocessing(df, target_column='nonexistent_column')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        # Depending on implementation, might succeed with warning or fail gracefully
    
    def test_memory_pressure_handling(self):
        """Test handling under high memory pressure"""
        pipeline = OptimizedDataPipeline(max_memory_pct=5.0)  # Very low threshold
        
        # Create moderately large DataFrame
        large_df = pd.DataFrame({
            'data': np.random.randn(10000),
            'strings': ['text_' + str(i) for i in range(10000)]
        })
        
        # Should handle memory constraints gracefully
        result = pipeline.optimize_memory(large_df)
        self.assertTrue(result['success'])


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of the optimization system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_processing_time_tracking(self):
        """Test that processing times are tracked"""
        import time
        
        df = pd.DataFrame({
            'data': np.random.randn(1000),
            'categories': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        pipeline = OptimizedDataPipeline()
        
        start_time = time.time()
        result = pipeline.optimize_memory(df)
        end_time = time.time()
        
        self.assertTrue(result['success'])
        
        # Should complete in reasonable time (< 5 seconds for small dataset)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 5.0)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        # Create DataFrame that should benefit from optimization
        df = pd.DataFrame({
            'int64_col': pd.Series(range(5000), dtype='int64'),
            'float64_col': pd.Series(np.random.randn(5000), dtype='float64'),
            'string_col': ['category_' + str(i % 10) for i in range(5000)]
        })
        
        pipeline = OptimizedDataPipeline()
        
        original_memory = df.memory_usage(deep=True).sum()
        
        result = pipeline.optimize_memory(df)
        self.assertTrue(result['success'])
        
        optimized_df = result['optimized_data']
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should reduce memory usage
        self.assertLessEqual(optimized_memory, original_memory)
        
        # Should provide memory info
        if 'optimization_info' in result:
            opt_info = result['optimization_info']
            self.assertIn('memory', opt_info)
            mem_info = opt_info['memory']
            self.assertIn('original_mb', mem_info)
            self.assertIn('optimized_mb', mem_info)
    
    def test_scalability_characteristics(self):
        """Test scalability with different dataset sizes"""
        pipeline = OptimizedDataPipeline()
        
        sizes = [100, 1000, 5000]  # Different dataset sizes
        processing_times = []
        
        for size in sizes:
            df = pd.DataFrame({
                'data': np.random.randn(size),
                'categories': np.random.choice(['A', 'B', 'C'], size)
            })
            
            start_time = time.time()
            result = pipeline.optimize_memory(df)
            end_time = time.time()
            
            self.assertTrue(result['success'])
            processing_times.append(end_time - start_time)
        
        # Processing time should scale reasonably (not exponentially)
        # Allow for some variation in test environment
        for i in range(1, len(processing_times)):
            time_ratio = processing_times[i] / processing_times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Time should not grow much faster than data size
            self.assertLess(time_ratio, size_ratio * 2)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main()
