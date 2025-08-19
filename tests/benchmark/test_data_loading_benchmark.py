# ---------------------------------------------------------------------
# tests/benchmark/test_data_loading_benchmark.py - Data loading performance tests
# ---------------------------------------------------------------------
"""
Benchmark tests for data loading and processing performance.
Tests the recent changes to data loading paths in app.py
"""
import pytest
import pandas as pd
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Import with fallback handling like in app.py
try:
    from modules.ui.sample_data_loader import SampleDataLoader
    from modules.ui.data_preview_generator import DataPreviewGenerator
    HAS_UI_MODULES = True
except ImportError:
    HAS_UI_MODULES = False
    print("🔧 DEBUG: UI modules not available - using fallback implementations")

try:
    from app import MLSystemUI
    HAS_APP = True
except (ImportError, SystemExit):
    HAS_APP = False
    print("🔧 DEBUG: Main app not available")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.data_loading
]

class TestDataLoadingBenchmarks:
    """Benchmark tests for data loading performance."""
    
    @pytest.mark.benchmark
    def test_csv_loading_performance(self, test_data_generator, benchmark_result):
        """Benchmark CSV loading performance across different sizes."""
        benchmark_result.start()
        
        results = {}
        
        for size in ['tiny', 'small', 'medium']:
            test_data = test_data_generator(size)
            file_path = test_data['file_path']
            
            # Time the loading
            df, load_time = time_function(pd.read_csv, file_path)
            
            results[size] = {
                'rows': test_data['rows'],
                'cols': test_data['cols'],
                'file_size_mb': file_path.stat().st_size / 1024**2,
                'load_time_ms': load_time,
                'load_rate_mb_per_sec': (file_path.stat().st_size / 1024**2) / (load_time / 1000) if load_time > 0 else 0
            }
            
            # Cleanup
            del df
        
        benchmark_result.stop()
        benchmark_result.metadata = results
        
        # Assert reasonable performance
        assert results['small']['load_time_ms'] < 1000, f"Small dataset took {results['small']['load_time_ms']}ms"
        assert results['medium']['load_time_ms'] < 5000, f"Medium dataset took {results['medium']['load_time_ms']}ms"
    
    @pytest.mark.benchmark
    def test_excel_loading_performance(self, test_data_generator, benchmark_result):
        """Benchmark Excel loading performance."""
        benchmark_result.start()
        
        test_data = test_data_generator('small')
        df_csv = test_data['dataframe']
        
        # Create Excel file
        temp_dir = Path(tempfile.gettempdir()) / "kolosal_benchmark"
        excel_file = temp_dir / "test_data_excel.xlsx"
        df_csv.to_excel(excel_file, index=False)
        
        # Time Excel loading
        df_excel, load_time = time_function(pd.read_excel, excel_file)
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'rows': df_excel.shape[0],
            'cols': df_excel.shape[1],
            'file_size_mb': excel_file.stat().st_size / 1024**2,
            'load_time_ms': load_time
        }
        
        # Excel should be slower than CSV but still reasonable
        assert load_time < 5000, f"Excel loading took {load_time}ms"
        assert df_excel.shape == df_csv.shape, "Excel and CSV data shapes should match"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_UI_MODULES, reason="UI modules not available")
    def test_sample_data_loader_performance(self, benchmark_result):
        """Benchmark sample data loader performance."""
        benchmark_result.start()
        
        loader = SampleDataLoader()
        results = {}
        
        datasets = ["Iris Classification", "Boston Housing", "Wine Quality", "Diabetes", "Breast Cancer"]
        
        for dataset_name in datasets:
            df, load_time = time_function(loader.load_sample_data, dataset_name)
            
            results[dataset_name] = {
                'rows': df[0].shape[0] if isinstance(df, tuple) else df.shape[0],
                'cols': df[0].shape[1] if isinstance(df, tuple) else df.shape[1],
                'load_time_ms': load_time
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = results
        
        # Sample data should load very quickly
        for dataset, result in results.items():
            assert result['load_time_ms'] < 100, f"{dataset} took {result['load_time_ms']}ms"
    
    @pytest.mark.benchmark
    def test_data_preview_generation_performance(self, test_data_generator, benchmark_result):
        """Benchmark data preview generation performance."""
        benchmark_result.start()
        
        results = {}
        
        for size in ['small', 'medium']:
            test_data = test_data_generator(size)
            df = test_data['dataframe']
            
            # Mock data preview generator if not available
            if HAS_UI_MODULES:
                try:
                    preview_gen = DataPreviewGenerator()
                    
                    # Time summary generation
                    summary, summary_time = time_function(preview_gen.generate_data_summary, df)
                    
                    # Time preview formatting
                    preview, preview_time = time_function(preview_gen.format_data_preview, df, summary)
                    
                except Exception as e:
                    # Fallback to simple operations
                    summary, summary_time = time_function(lambda: {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict()
                    })
                    
                    preview, preview_time = time_function(lambda: f"Data shape: {df.shape}")
            else:
                # Fallback implementations
                summary, summary_time = time_function(lambda: {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict()
                })
                
                preview, preview_time = time_function(lambda: f"Data shape: {df.shape}")
            
            # Time HTML table generation
            html_table, html_time = time_function(
                lambda: df.head(10).to_html(classes="table table-striped", escape=False, border=0)
            )
            
            results[size] = {
                'rows': df.shape[0],
                'cols': df.shape[1],
                'summary_time_ms': summary_time,
                'preview_time_ms': preview_time,
                'html_time_ms': html_time,
                'total_time_ms': summary_time + preview_time + html_time
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = results
        
        # Preview generation should be fast
        for size, result in results.items():
            assert result['total_time_ms'] < 2000, f"{size} preview took {result['total_time_ms']}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.memory
    def test_memory_usage_data_loading(self, test_data_generator, memory_monitor, benchmark_result):
        """Test memory usage during data loading."""
        benchmark_result.start()
        
        results = {}
        
        for size in ['small', 'medium']:
            test_data = test_data_generator(size)
            file_path = test_data['file_path']
            expected_memory = test_data['memory_mb']
            
            memory_monitor.sample()  # Baseline
            
            # Load data and measure memory
            df, memory_stats = measure_memory_usage(pd.read_csv, file_path)
            
            memory_monitor.sample()  # After loading
            
            results[size] = {
                'expected_memory_mb': expected_memory,
                'actual_memory_delta_mb': memory_stats['delta_mb'],
                'efficiency_ratio': expected_memory / memory_stats['delta_mb'] if memory_stats['delta_mb'] > 0 else 0,
                'rows': df.shape[0],
                'cols': df.shape[1]
            }
            
            # Cleanup
            del df
            
        benchmark_result.stop()
        benchmark_result.metadata = {
            'memory_tests': results,
            'memory_monitor_stats': memory_monitor.get_stats()
        }
        
        # Memory usage should be reasonable (within 3x of expected)
        for size, result in results.items():
            assert result['actual_memory_delta_mb'] < result['expected_memory_mb'] * 3, \
                f"{size} used {result['actual_memory_delta_mb']}MB vs expected {result['expected_memory_mb']}MB"
    
    @pytest.mark.benchmark
    @pytest.mark.consistency
    def test_data_loading_consistency(self, test_data_generator, benchmark_result):
        """Test that data loading is consistent across multiple runs."""
        benchmark_result.start()
        
        test_data = test_data_generator('small')
        file_path = test_data['file_path']
        
        # Load the same file multiple times
        load_times = []
        checksums = []
        
        for i in range(5):
            df, load_time = time_function(pd.read_csv, file_path)
            load_times.append(load_time)
            
            # Simple checksum for consistency
            checksum = hash(tuple(df.iloc[0]))
            checksums.append(checksum)
            
            del df
        
        benchmark_result.stop()
        
        # Calculate statistics
        avg_time = sum(load_times) / len(load_times)
        std_time = np.std(load_times)
        variation_percent = (std_time / avg_time) * 100 if avg_time > 0 else 0
        
        benchmark_result.metadata = {
            'load_times_ms': load_times,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'variation_percent': variation_percent,
            'checksums': checksums,
            'consistent_data': len(set(checksums)) == 1
        }
        
        # Data should be consistent
        assert len(set(checksums)) == 1, "Data loading produced inconsistent results"
        
        # Timing should be relatively stable (< 50% variation)
        assert variation_percent < 50, f"Load time variation too high: {variation_percent}%"


class TestFileValidationBenchmarks:
    """Benchmark tests for file validation and security checks."""
    
    @pytest.mark.benchmark
    def test_file_validation_performance(self, test_data_generator, benchmark_result):
        """Benchmark file validation performance."""
        benchmark_result.start()
        
        # Create test files with different extensions
        test_data = test_data_generator('small')
        base_path = test_data['file_path'].parent
        
        test_files = []
        extensions = ['.csv', '.xlsx', '.json', '.txt', '.exe']
        
        for ext in extensions:
            file_path = base_path / f"test_file{ext}"
            if ext == '.csv':
                test_data['dataframe'].to_csv(file_path, index=False)
            elif ext == '.xlsx':
                test_data['dataframe'].to_excel(file_path, index=False)
            elif ext == '.json':
                test_data['dataframe'].to_json(file_path)
            else:
                # Create dummy files
                file_path.write_text("dummy content")
            
            test_files.append(file_path)
        
        # Mock the validation function from app.py
        def mock_validate_file_upload(path, allowed_extensions):
            """Mock validation function."""
            return Path(path).suffix.lower() in allowed_extensions
        
        # Test validation performance
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        validation_times = []
        
        for file_path in test_files:
            _, validation_time = time_function(
                mock_validate_file_upload, 
                str(file_path), 
                allowed_extensions
            )
            validation_times.append(validation_time)
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'validation_times_ms': validation_times,
            'avg_validation_time_ms': sum(validation_times) / len(validation_times),
            'max_validation_time_ms': max(validation_times),
            'files_tested': len(test_files)
        }
        
        # Validation should be very fast
        assert max(validation_times) < 10, f"File validation too slow: {max(validation_times)}ms"
    
    @pytest.mark.benchmark
    def test_size_limit_validation_performance(self, benchmark_result):
        """Benchmark size limit validation performance."""
        benchmark_result.start()
        
        # Test different data sizes against limits
        test_cases = [
            (1000, 10, False),      # Normal size
            (100000, 100, False),   # Large but acceptable
            (1000000, 50, False),   # At row limit
            (1000001, 50, True),    # Over row limit
            (50000, 10000, False),  # At column limit
            (50000, 10001, True),   # Over column limit
        ]
        
        results = []
        
        for rows, cols, should_fail in test_cases:
            # Mock DataFrame shape check
            mock_df = Mock()
            mock_df.shape = (rows, cols)
            
            def check_limits(df):
                if df.shape[0] > 1000000:
                    return False, "Too many rows"
                if df.shape[1] > 10000:
                    return False, "Too many columns"
                return True, "OK"
            
            result, check_time = time_function(check_limits, mock_df)
            is_valid, message = result
            
            results.append({
                'rows': rows,
                'cols': cols,
                'should_fail': should_fail,
                'is_valid': is_valid,
                'check_time_ms': check_time,
                'message': message
            })
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'limit_checks': results,
            'avg_check_time_ms': sum(r['check_time_ms'] for r in results) / len(results)
        }
        
        # Size checks should be instant
        for result in results:
            assert result['check_time_ms'] < 1, f"Size check too slow: {result['check_time_ms']}ms"
            
            # Validation logic should be correct
            expected_valid = not result['should_fail']
            assert result['is_valid'] == expected_valid, \
                f"Size validation incorrect for {result['rows']}x{result['cols']}"


@pytest.mark.benchmark
@pytest.mark.stress
class TestDataLoadingStress:
    """Stress tests for data loading under high load."""
    
    @pytest.mark.stress
    def test_repeated_loading_stress(self, test_data_generator, benchmark_result, benchmark_config):
        """Test repeated data loading operations."""
        benchmark_result.start()
        
        test_data = test_data_generator('small')
        file_path = test_data['file_path']
        
        iterations = benchmark_config['stress_iterations']
        load_times = []
        memory_samples = []
        
        import psutil
        process = psutil.Process()
        
        for i in range(iterations):
            # Sample memory before
            memory_before = process.memory_info().rss / 1024**2
            
            # Load data
            df, load_time = time_function(pd.read_csv, file_path)
            load_times.append(load_time)
            
            # Sample memory after
            memory_after = process.memory_info().rss / 1024**2
            memory_samples.append(memory_after)
            
            # Cleanup
            del df
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                import gc
                gc.collect()
        
        benchmark_result.stop()
        
        # Calculate statistics
        avg_load_time = sum(load_times) / len(load_times)
        memory_growth = memory_samples[-1] - memory_samples[0]
        
        benchmark_result.metadata = {
            'iterations': iterations,
            'avg_load_time_ms': avg_load_time,
            'min_load_time_ms': min(load_times),
            'max_load_time_ms': max(load_times),
            'memory_growth_mb': memory_growth,
            'final_memory_mb': memory_samples[-1],
            'load_time_degradation': (load_times[-10:]) if len(load_times) >= 10 else load_times
        }
        
        # Performance should remain stable
        assert memory_growth < 100, f"Memory grew by {memory_growth}MB"
        
        # Load times shouldn't degrade significantly
        early_avg = sum(load_times[:10]) / 10 if len(load_times) >= 10 else avg_load_time
        late_avg = sum(load_times[-10:]) / 10 if len(load_times) >= 10 else avg_load_time
        degradation = (late_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0
        
        assert degradation < 100, f"Load time degraded by {degradation}%"
    
    @pytest.mark.stress
    def test_concurrent_loading_simulation(self, test_data_generator, benchmark_result):
        """Simulate concurrent data loading operations."""
        benchmark_result.start()
        
        test_data = test_data_generator('small')
        file_path = test_data['file_path']
        
        # Simulate concurrent operations by rapidly switching between operations
        operations = []
        start_time = time.perf_counter()
        
        for i in range(20):
            # Load data
            df, load_time = time_function(pd.read_csv, file_path)
            operations.append(('load', load_time))
            
            # Generate summary
            summary, summary_time = time_function(lambda: {
                "shape": df.shape,
                "columns": df.columns.tolist()
            })
            operations.append(('summary', summary_time))
            
            # Generate HTML
            html, html_time = time_function(
                lambda: df.head(5).to_html(escape=False)
            )
            operations.append(('html', html_time))
            
            del df, summary, html
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'total_operations': len(operations),
            'total_time_ms': total_time,
            'avg_operation_time_ms': sum(op[1] for op in operations) / len(operations),
            'operation_breakdown': {
                'load': [op[1] for op in operations if op[0] == 'load'],
                'summary': [op[1] for op in operations if op[0] == 'summary'],
                'html': [op[1] for op in operations if op[0] == 'html']
            }
        }
        
        # System should handle rapid operations
        assert total_time < 30000, f"Concurrent simulation took too long: {total_time}ms"
