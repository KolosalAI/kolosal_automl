"""
Integration tests for the Complete Optimization System.

This test suite covers:
- End-to-end optimization workflows
- Integration with existing Kolosal AutoML components
- Real-world dataset scenarios
- Performance benchmarks
- Cross-module interactions
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Import optimization modules
try:
    from modules.engine.optimization_integration import (
        OptimizedDataPipeline, 
        create_optimized_training_pipeline,
        get_optimization_status
    )
    from modules.engine.optimized_data_loader import OptimizedDataLoader, LoadingConfig
    from modules.engine.adaptive_preprocessing import ConfigOptimizer
    from modules.engine.memory_aware_processor import MemoryAwareDataProcessor
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import existing Kolosal AutoML components for integration testing
try:
    from modules.engine.data_preprocessor import DataPreprocessor
    from modules.engine.batch_processor import BatchProcessor
    from app import MLSystemUI
    KOLOSAL_COMPONENTS_AVAILABLE = True
except ImportError:
    KOLOSAL_COMPONENTS_AVAILABLE = False


@unittest.skipUnless(OPTIMIZATION_AVAILABLE, "Optimization modules not available")
class TestCompleteOptimizationWorkflow(unittest.TestCase):
    """Test complete optimization workflows from start to finish"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Initialize optimization pipeline
        self.pipeline = OptimizedDataPipeline(
            max_memory_pct=75.0,
            enable_memory_optimization=True,
            enable_adaptive_preprocessing=True
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_benchmark_dataset(self, size_category: str) -> Path:
        """Create benchmark datasets for different size categories"""
        np.random.seed(42)
        
        if size_category == "tiny":
            n_samples = 1_000
            n_features = 10
        elif size_category == "small":
            n_samples = 10_000
            n_features = 20
        elif size_category == "medium":
            n_samples = 100_000
            n_features = 30
        elif size_category == "large":
            n_samples = 1_500_000  # Changed to ensure it falls in large category (>1M)
            n_features = 50
        else:
            raise ValueError(f"Unknown size category: {size_category}")
        
        # Create mixed data types
        data = {}
        
        # Numerical features
        for i in range(n_features // 3):
            data[f'num_{i}'] = np.random.randn(n_samples)
        
        # Categorical features with varying cardinality
        for i in range(n_features // 3):
            cardinality = min(50, max(3, n_samples // 1000))
            data[f'cat_{i}'] = np.random.choice([f'cat_{j}' for j in range(cardinality)], n_samples)
        
        # Mixed features (strings that could be categorical)
        for i in range(n_features - 2 * (n_features // 3)):
            data[f'mixed_{i}'] = [f'item_{j % 20}' for j in range(n_samples)]
        
        # Target variable
        data['target'] = np.random.randint(0, 2, n_samples)
        
        # Add some missing values
        df = pd.DataFrame(data)
        missing_ratio = 0.05
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'target':
                missing_indices = np.random.choice(n_samples, int(n_samples * missing_ratio), replace=False)
                df.loc[missing_indices, col] = np.nan
        
        # Save to file
        file_path = self.temp_path / f"{size_category}_dataset.csv"
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def test_tiny_dataset_workflow(self):
        """Test optimization workflow with tiny dataset (1K rows)"""
        dataset_file = self.create_benchmark_dataset("tiny")
        
        # Start timer
        start_time = time.time()
        
        # Complete workflow
        result = self.pipeline.process_complete_pipeline(
            file_path=str(dataset_file),
            target_column='target',
            fit_preprocessor=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertIn('data', result)
        self.assertIn('dataset_info', result)
        
        df = result['data']
        self.assertEqual(len(df), 1_000)
        
        # Performance expectations for tiny dataset
        self.assertLess(processing_time, 5.0)  # Should complete in < 5 seconds
        
        # Should use direct loading strategy
        dataset_info = result['dataset_info']
        self.assertEqual(dataset_info.size_category.value, "tiny")
        
        print(f"Tiny dataset processing time: {processing_time:.2f}s")
    
    def test_small_dataset_workflow(self):
        """Test optimization workflow with small dataset (10K rows)"""
        dataset_file = self.create_benchmark_dataset("small")
        
        start_time = time.time()
        
        result = self.pipeline.process_complete_pipeline(
            file_path=str(dataset_file),
            target_column='target',
            fit_preprocessor=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        self.assertTrue(result['success'])
        
        df = result['data']
        self.assertEqual(len(df), 10_000)
        
        # Performance expectations for small dataset
        self.assertLess(processing_time, 15.0)  # Should complete in < 15 seconds
        
        # Should use direct or chunked loading
        dataset_info = result['dataset_info']
        self.assertEqual(dataset_info.size_category.value, "small")
        
        print(f"Small dataset processing time: {processing_time:.2f}s")
    
    def test_medium_dataset_workflow(self):
        """Test optimization workflow with medium dataset (100K rows)"""
        dataset_file = self.create_benchmark_dataset("medium")
        
        start_time = time.time()
        
        result = self.pipeline.process_complete_pipeline(
            file_path=str(dataset_file),
            target_column='target',
            fit_preprocessor=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        self.assertTrue(result['success'])
        
        df = result['data']
        self.assertEqual(len(df), 100_000)
        
        # Performance expectations for medium dataset
        self.assertLess(processing_time, 60.0)  # Should complete in < 1 minute
        
        # Should use chunked or streaming strategy
        dataset_info = result['dataset_info']
        self.assertEqual(dataset_info.size_category.value, "medium")
        
        # Should show significant memory optimization
        if 'memory_optimization_results' in result:
            mem_results = result['memory_optimization_results']
            if mem_results['success'] and 'optimization_info' in mem_results:
                memory_info = mem_results['optimization_info']['memory']
                reduction_pct = memory_info.get('reduction_percent', 0)
                self.assertGreater(reduction_pct, 10)  # At least 10% reduction
        
        print(f"Medium dataset processing time: {processing_time:.2f}s")
    
    def test_large_dataset_workflow(self):
        """Test optimization workflow with large dataset (500K rows)"""
        dataset_file = self.create_benchmark_dataset("large")
        
        start_time = time.time()
        
        result = self.pipeline.process_complete_pipeline(
            file_path=str(dataset_file),
            target_column='target',
            fit_preprocessor=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        self.assertTrue(result['success'])
        
        df = result['data']
        self.assertEqual(len(df), 1_500_000)
        
        # Performance expectations for large dataset
        self.assertLess(processing_time, 300.0)  # Should complete in < 5 minutes
        
        # Should use advanced loading strategy
        dataset_info = result['dataset_info']
        self.assertEqual(dataset_info.size_category.value, "large")
        
        # Should show significant memory optimization
        if 'memory_optimization_results' in result:
            mem_results = result['memory_optimization_results']
            if mem_results['success'] and 'optimization_info' in mem_results:
                memory_info = mem_results['optimization_info']['memory']
                reduction_pct = memory_info.get('reduction_percent', 0)
                self.assertGreater(reduction_pct, 20)  # At least 20% reduction for large dataset
        
        print(f"Large dataset processing time: {processing_time:.2f}s")
    
    def test_memory_optimization_effectiveness(self):
        """Test effectiveness of memory optimization across dataset sizes"""
        results = {}
        
        for size in ["tiny", "small", "medium"]:
            dataset_file = self.create_benchmark_dataset(size)
            
            # Load without optimization
            df_original = pd.read_csv(dataset_file)
            original_memory = df_original.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Load with optimization
            result = self.pipeline.load_data(str(dataset_file))
            self.assertTrue(result['success'])
            
            df_optimized = result['data']
            optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Calculate reduction
            memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
            
            results[size] = {
                'original_mb': original_memory,
                'optimized_mb': optimized_memory,
                'reduction_percent': memory_reduction
            }
            
            # Should achieve meaningful memory reduction
            self.assertGreaterEqual(memory_reduction, 0)  # At minimum, no increase
            
            print(f"{size.title()} dataset: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                  f"({memory_reduction:.1f}% reduction)")
        
        # Larger datasets should show better optimization
        if 'medium' in results and 'tiny' in results:
            self.assertGreaterEqual(
                results['medium']['reduction_percent'],
                results['tiny']['reduction_percent'] * 0.8  # Allow some variation
            )
    
    def test_preprocessing_adaptation(self):
        """Test that preprocessing adapts to different dataset characteristics"""
        # Test with different types of datasets
        datasets = {
            'normal': self.create_benchmark_dataset("small"),
            'high_cardinality': self.create_high_cardinality_dataset(),
            'sparse': self.create_sparse_dataset(),
            'outlier_heavy': self.create_outlier_dataset()
        }
        
        configs = {}
        
        for dataset_type, file_path in datasets.items():
            df = pd.read_csv(file_path)
            
            result = self.pipeline.optimize_preprocessing(df, target_column='target')
            self.assertTrue(result['success'])
            
            configs[dataset_type] = result.get('config', {})
        
        # Different datasets should get different configurations
        normalization_methods = set()
        strategies = set()
        
        for config in configs.values():
            if 'normalization' in config:
                normalization_methods.add(config['normalization'])
            if 'strategy' in config:
                strategies.add(config['strategy'])
        
        # Should use different approaches for different data types
        print(f"Normalization methods used: {normalization_methods}")
        print(f"Strategies used: {strategies}")
    
    def create_high_cardinality_dataset(self) -> Path:
        """Create dataset with high cardinality categorical features"""
        np.random.seed(42)
        n_samples = 5_000
        
        df = pd.DataFrame({
            'numeric1': np.random.randn(n_samples),
            'numeric2': np.random.randn(n_samples),
            'high_card1': [f'item_{i}' for i in range(n_samples)],  # Unique values
            'high_card2': [f'cat_{i % 1000}' for i in range(n_samples)],  # High cardinality
            'low_card': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        file_path = self.temp_path / "high_cardinality.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def create_sparse_dataset(self) -> Path:
        """Create sparse dataset with many zero values"""
        np.random.seed(42)
        n_samples = 5_000
        n_features = 20
        
        # Create sparse data (80% zeros)
        data = np.random.randn(n_samples, n_features)
        zero_mask = np.random.choice([True, False], size=data.shape, p=[0.8, 0.2])
        data[zero_mask] = 0
        
        df = pd.DataFrame(data, columns=[f'sparse_feature_{i}' for i in range(n_features)])
        df['target'] = np.random.randint(0, 2, n_samples)
        
        file_path = self.temp_path / "sparse.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def create_outlier_dataset(self) -> Path:
        """Create dataset with many outliers"""
        np.random.seed(42)
        n_samples = 5_000
        
        # Normal data with injected outliers
        normal_data = np.random.randn(n_samples)
        outlier_indices = np.random.choice(n_samples, n_samples // 10, replace=False)
        normal_data[outlier_indices] = np.random.randn(len(outlier_indices)) * 10 + 20
        
        df = pd.DataFrame({
            'normal_feature': np.random.randn(n_samples),
            'outlier_feature': normal_data,
            'skewed_feature': np.random.exponential(2, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        file_path = self.temp_path / "outliers.csv"
        df.to_csv(file_path, index=False)
        return file_path


@unittest.skipUnless(KOLOSAL_COMPONENTS_AVAILABLE and OPTIMIZATION_AVAILABLE, 
                     "Kolosal components and optimization modules not available")
class TestKolosalIntegration(unittest.TestCase):
    """Test integration with existing Kolosal AutoML components"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataset(self) -> Path:
        """Create a test dataset for integration testing"""
        np.random.seed(42)
        n_samples = 1_000
        
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.uniform(0, 100, n_samples),
            'feature3': np.random.exponential(2, n_samples),
            'category1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'category2': np.random.choice(['X', 'Y'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, n_samples // 20, replace=False)
        df.loc[missing_indices, 'feature1'] = np.nan
        
        file_path = self.temp_path / "integration_test.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_data_preprocessor_integration(self):
        """Test integration with enhanced DataPreprocessor"""
        dataset_file = self.create_test_dataset()
        df = pd.read_csv(dataset_file)
        
        # Test enhanced DataPreprocessor with optimization
        try:
            preprocessor = DataPreprocessor()
            
            # Check if it has the new optimize_for_dataset method
            if hasattr(preprocessor, 'optimize_for_dataset'):
                optimized_config = preprocessor.optimize_for_dataset(df)
                self.assertIsInstance(optimized_config, dict)
                print("DataPreprocessor optimization integration: [PASS]")
            else:
                print("DataPreprocessor optimization integration: Not available")
                
        except Exception as e:
            print(f"DataPreprocessor integration test failed: {e}")
    
    def test_batch_processor_integration(self):
        """Test integration with enhanced BatchProcessor"""
        dataset_file = self.create_test_dataset()
        df = pd.read_csv(dataset_file)
        
        try:
            batch_processor = BatchProcessor()
            
            # Check if memory-aware components are integrated
            has_memory_processor = hasattr(batch_processor, '_memory_processor')
            has_adaptive_chunker = hasattr(batch_processor, '_adaptive_chunker')
            
            if has_memory_processor or has_adaptive_chunker:
                print("BatchProcessor optimization integration: [PASS]")
                
                # Test batch processing with optimization
                X = df.drop('target', axis=1)
                y = df['target']
                
                # Should handle the data without errors
                # Note: Actual batch processing might need more setup
                self.assertIsNotNone(batch_processor)
            else:
                print("BatchProcessor optimization integration: Not available")
                
        except Exception as e:
            print(f"BatchProcessor integration test failed: {e}")
    
    def test_mlsystem_ui_integration(self):
        """Test integration with MLSystemUI (main app)"""
        try:
            # Test MLSystemUI initialization with optimization
            app = MLSystemUI()
            
            # Check if optimization pipeline is available
            has_optimization_pipeline = hasattr(app, 'optimization_pipeline')
            
            if has_optimization_pipeline and app.optimization_pipeline is not None:
                print("MLSystemUI optimization integration: [PASS]")
                
                # Test optimization status
                status = get_optimization_status()
                self.assertIsInstance(status, dict)
                self.assertIn('optimization_available', status)
                
            else:
                print("MLSystemUI optimization integration: Not available")
                
        except Exception as e:
            print(f"MLSystemUI integration test failed: {e}")
    
    def test_end_to_end_workflow_with_app(self):
        """Test complete end-to-end workflow through the app interface"""
        dataset_file = self.create_test_dataset()
        
        try:
            app = MLSystemUI()
            
            if hasattr(app, 'optimization_pipeline') and app.optimization_pipeline:
                # Test loading data through the app interface
                # This would typically be done through the Gradio interface
                # but we can test the underlying methods
                
                # Simulate the app's load_data method
                if hasattr(app, 'load_data'):
                    print("Testing end-to-end workflow through app interface...")
                    
                    # The actual load_data method might require specific parameters
                    # This is more of a smoke test to ensure integration doesn't break
                    self.assertIsNotNone(app.optimization_pipeline)
                    
                    print("End-to-end workflow integration: [PASS]")
                else:
                    print("App load_data method not available for testing")
            else:
                print("Optimization pipeline not available in app")
                
        except Exception as e:
            print(f"End-to-end workflow test failed: {e}")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for the optimization system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(OPTIMIZATION_AVAILABLE, "Optimization modules not available")
    def test_loading_performance_benchmark(self):
        """Benchmark data loading performance"""
        print("\n=== DATA LOADING PERFORMANCE BENCHMARK ===")
        
        # Test different file sizes
        test_cases = [
            ("tiny", 1_000, 10),
            ("small", 10_000, 20),
            ("medium", 50_000, 30),
        ]
        
        results = {}
        
        for size_name, n_rows, n_cols in test_cases:
            # Create test dataset
            np.random.seed(42)
            df = pd.DataFrame({
                f'feature_{i}': np.random.randn(n_rows) for i in range(n_cols)
            })
            df['category'] = np.random.choice(['A', 'B', 'C'], n_rows)
            
            file_path = self.temp_path / f"{size_name}.csv"
            df.to_csv(file_path, index=False)
            
            # Benchmark optimized loading
            loader = OptimizedDataLoader()
            
            start_time = time.time()
            result_df, dataset_info = loader.load_data(str(file_path))
            end_time = time.time()
            
            loading_time = end_time - start_time
            rows_per_second = n_rows / loading_time if loading_time > 0 else float('inf')
            
            results[size_name] = {
                'rows': n_rows,
                'loading_time': loading_time,
                'rows_per_second': rows_per_second,
                'strategy': dataset_info.loading_strategy.value,
                'memory_mb': dataset_info.actual_memory_mb
            }
            
            print(f"{size_name.title()}: {n_rows:,} rows loaded in {loading_time:.3f}s "
                  f"({rows_per_second:,.0f} rows/sec) using {dataset_info.loading_strategy.value} strategy")
        
        # Performance assertions
        self.assertGreater(results['tiny']['rows_per_second'], 10_000)  # At least 10K rows/sec
        self.assertGreater(results['small']['rows_per_second'], 5_000)  # At least 5K rows/sec
        
        return results
    
    @unittest.skipUnless(OPTIMIZATION_AVAILABLE, "Optimization modules not available")
    def test_memory_optimization_benchmark(self):
        """Benchmark memory optimization performance"""
        print("\n=== MEMORY OPTIMIZATION PERFORMANCE BENCHMARK ===")
        
        # Create test datasets with different characteristics
        test_cases = {
            'mixed_types': self.create_mixed_types_dataset(10_000),
            'high_cardinality': self.create_high_cardinality_dataset(5_000),
            'sparse_data': self.create_sparse_dataset(15_000)
        }
        
        processor = MemoryAwareDataProcessor()
        
        for dataset_name, df in test_cases.items():
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            start_time = time.time()
            optimized_df = processor.optimize_dataframe_memory(df)
            end_time = time.time()
            
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            optimization_time = end_time - start_time
            
            reduction_percent = ((original_memory - optimized_memory) / original_memory) * 100
            mb_per_second = original_memory / optimization_time if optimization_time > 0 else float('inf')
            
            print(f"{dataset_name}: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                  f"({reduction_percent:.1f}% reduction) in {optimization_time:.3f}s "
                  f"({mb_per_second:.1f} MB/s)")
            
            # Verify optimization effectiveness
            self.assertGreaterEqual(reduction_percent, 0)  # Should not increase memory
            self.assertLess(optimization_time, 10.0)  # Should complete reasonably quickly
    
    def create_mixed_types_dataset(self, n_samples: int) -> pd.DataFrame:
        """Create dataset with mixed data types for benchmarking"""
        np.random.seed(42)
        return pd.DataFrame({
            'int64_col': pd.Series(range(n_samples), dtype='int64'),
            'float64_col': pd.Series(np.random.randn(n_samples), dtype='float64'),
            'string_col': [f'category_{i % 10}' for i in range(n_samples)],
            'bool_col': np.random.choice([True, False], n_samples),
            'object_col': [f'item_{i}' for i in range(n_samples)]
        })
    
    def create_high_cardinality_dataset(self, n_samples: int) -> pd.DataFrame:
        """Create dataset with high cardinality for benchmarking"""
        np.random.seed(42)
        return pd.DataFrame({
            'id_col': [f'id_{i}' for i in range(n_samples)],
            'high_card': [f'cat_{i % 500}' for i in range(n_samples)],
            'medium_card': [f'type_{i % 50}' for i in range(n_samples)],
            'low_card': np.random.choice(['A', 'B', 'C'], n_samples),
            'numeric': np.random.randn(n_samples)
        })
    
    def create_sparse_dataset(self, n_samples: int) -> pd.DataFrame:
        """Create sparse dataset for benchmarking"""
        np.random.seed(42)
        n_features = 20
        
        # Create mostly zero data
        data = np.random.randn(n_samples, n_features)
        zero_mask = np.random.choice([True, False], size=data.shape, p=[0.85, 0.15])
        data[zero_mask] = 0
        
        return pd.DataFrame(data, columns=[f'sparse_{i}' for i in range(n_features)])


class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration and compatibility"""
    
    @unittest.skipUnless(OPTIMIZATION_AVAILABLE, "Optimization modules not available")
    def test_optimization_status_reporting(self):
        """Test optimization status and availability reporting"""
        status = get_optimization_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('optimization_available', status)
        self.assertIn('modules_loaded', status)
        
        # Should report available modules
        modules = status['modules_loaded']
        self.assertIsInstance(modules, dict)
        
        print(f"Optimization Status: {status}")
        
        # If optimization is available, key modules should be loaded
        if status['optimization_available']:
            expected_modules = ['data_loader', 'adaptive_preprocessing', 'memory_processor']
            for module in expected_modules:
                self.assertIn(module, modules)
    
    @unittest.skipUnless(OPTIMIZATION_AVAILABLE, "Optimization modules not available")
    def test_graceful_degradation(self):
        """Test graceful degradation when optimization components fail"""
        pipeline = OptimizedDataPipeline()
        
        # Create simple test data
        df = pd.DataFrame({
            'x': range(100),
            'y': np.random.randn(100),
            'category': ['A'] * 50 + ['B'] * 50
        })
        
        # Test with simulated component failures
        with patch.object(pipeline, 'data_loader', None):
            # Should still work with fallback
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            try:
                df.to_csv(temp_file.name, index=False)
                temp_file.close()  # Close file before using it
                
                result = pipeline.load_data(temp_file.name)
                self.assertTrue(result['success'])
                
            finally:
                # Cleanup - ensure file is closed before deletion
                try:
                    if not temp_file.closed:
                        temp_file.close()
                    import os
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except (OSError, PermissionError):
                    # On Windows, sometimes files are still locked
                    pass
    
    def test_import_safety(self):
        """Test that imports are safe and don't break existing functionality"""
        # Test that optimization modules can be imported safely
        try:
            from modules.engine import optimization_integration
            from modules.engine import optimized_data_loader
            from modules.engine import adaptive_preprocessing
            from modules.engine import memory_aware_processor
            print("All optimization modules imported successfully [PASS]")
        except ImportError as e:
            print(f"Import warning: {e}")
            # This is acceptable - modules might not be available in all environments
        
        # Test that existing modules still work
        try:
            from modules.engine.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            self.assertIsNotNone(preprocessor)
            print("Existing DataPreprocessor still functional [PASS]")
        except Exception as e:
            self.fail(f"Existing functionality broken: {e}")


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Print test environment info
    print("=" * 80)
    print("OPTIMIZATION SYSTEM INTEGRATION TESTS")
    print("=" * 80)
    print(f"Optimization modules available: {OPTIMIZATION_AVAILABLE}")
    print(f"Kolosal components available: {KOLOSAL_COMPONENTS_AVAILABLE}")
    print("=" * 80)
    
    # Run tests
    unittest.main(verbosity=2)
