# ---------------------------------------------------------------------
# tests/benchmark/test_ui_performance_benchmark.py - UI performance tests  
# ---------------------------------------------------------------------
"""
Benchmark tests for UI components and Gradio interface performance.
Tests the recent changes to UI components in app.py
"""
import pytest
import pandas as pd
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import threading
import queue

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

# Test if main components are available (defer import to avoid collection errors)
def _get_app_components():
    """Safely import app components."""
    try:
        # Force environment variable for testing
        os.environ["FORCE_FALLBACK_IMPLEMENTATIONS"] = "1"
        from app import MLSystemUI
        return MLSystemUI, True
    except (ImportError, SystemExit, Exception) as e:
        print(f"🔧 DEBUG: Main app import failed: {e}")
        return None, False

HAS_APP = False
try:
    MLSystemUI, HAS_APP = _get_app_components()
except:
    HAS_APP = False
    MLSystemUI = None

# Test if Gradio is available (defer import to avoid collection errors)
def _get_gradio():
    """Safely import Gradio."""
    try:
        import gradio as gr
        return gr, True
    except (ImportError, SystemExit, Exception):
        return None, False

HAS_GRADIO = False
try:
    gr, HAS_GRADIO = _get_gradio()
except:
    HAS_GRADIO = False
    gr = None

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.ui,
    pytest.mark.skipif(True, reason="UI tests disabled due to Pydantic/FastAPI compatibility issues")
]

def get_ui_system():
    """Safely get MLSystemUI class."""
    if HAS_APP and MLSystemUI:
        return MLSystemUI
    else:
        # Get it fresh if needed
        ui_class, available = _get_app_components()
        if available:
            return ui_class
        else:
            pytest.skip("MLSystemUI not available")

class TestUIComponentBenchmarks:
    """Benchmark tests for UI component performance."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_mlsystem_ui_initialization_performance(self, benchmark_result):
        """Benchmark UI initialization time."""
        benchmark_result.start()
        
        # Get UI class safely
        UIClass = get_ui_system()
        
        # Test with different initialization modes
        init_times = {}
        
        # Test inference-only mode
        ui_inference, init_time = time_function(lambda: UIClass(inference_only=True))
        init_times['inference_only'] = init_time
        
        # Test full mode (if available)
        try:
            ui_full, init_time = time_function(lambda: UIClass(inference_only=False))
            init_times['full_mode'] = init_time
        except Exception as e:
            init_times['full_mode'] = f"Failed: {e}"
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'initialization_times_ms': init_times,
            'has_fallbacks': hasattr(ui_inference, 'current_data_info') if 'ui_inference' in locals() else False
        }
        
        # Initialization should be reasonably fast
        assert init_times['inference_only'] < 5000, f"UI init took {init_times['inference_only']}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_data_loading_ui_performance(self, test_data_generator, benchmark_result):
        """Benchmark UI data loading operations."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        
        # Test sample data loading
        sample_results = {}
        sample_datasets = ["Iris Classification", "Boston Housing", "Wine Quality"]
        
        for dataset in sample_datasets:
            try:
                result, load_time = time_function(ui.load_sample_data, dataset)
                sample_results[dataset] = {
                    'load_time_ms': load_time,
                    'result_length': len(result) if hasattr(result, '__len__') else 0,
                    'status': 'success' if result else 'failed'
                }
            except Exception as e:
                sample_results[dataset] = {
                    'load_time_ms': 0,
                    'error': str(e),
                    'status': 'error'
                }
        
        # Test file data loading (mock file object)
        test_data = test_data_generator('small')
        mock_file = Mock()
        mock_file.name = str(test_data['file_path'])
        
        try:
            file_result, file_load_time = time_function(ui.load_data, mock_file)
            file_test_result = {
                'load_time_ms': file_load_time,
                'result_length': len(file_result) if hasattr(file_result, '__len__') else 0,
                'status': 'success'
            }
        except Exception as e:
            file_test_result = {
                'load_time_ms': 0,
                'error': str(e),
                'status': 'error'
            }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'sample_data_loading': sample_results,
            'file_data_loading': file_test_result
        }
        
        # UI operations should be fast
        for dataset, result in sample_results.items():
            if result['status'] == 'success':
                assert result['load_time_ms'] < 1000, f"{dataset} UI loading took {result['load_time_ms']}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_data_preview_ui_performance(self, test_data_generator, benchmark_result):
        """Benchmark data preview generation in UI context."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        test_data = test_data_generator('medium')
        
        # Set current data
        ui.current_data = test_data['dataframe']
        
        # Test refresh data preview
        preview_times = []
        
        for i in range(5):
            try:
                result, preview_time = time_function(ui.refresh_data_preview)
                preview_times.append(preview_time)
            except Exception as e:
                preview_times.append(f"Error: {e}")
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'data_shape': test_data['dataframe'].shape,
            'preview_times_ms': preview_times,
            'avg_preview_time_ms': sum(t for t in preview_times if isinstance(t, (int, float))) / len([t for t in preview_times if isinstance(t, (int, float))]) if any(isinstance(t, (int, float)) for t in preview_times) else 0,
            'errors': [t for t in preview_times if isinstance(t, str)]
        }
        
        # Preview should be fast even for medium datasets
        valid_times = [t for t in preview_times if isinstance(t, (int, float))]
        if valid_times:
            assert max(valid_times) < 3000, f"Data preview took {max(valid_times)}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_unified_data_loading_performance(self, test_data_generator, benchmark_result):
        """Benchmark the new unified data loading function."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        
        results = {}
        
        # Test 1: Sample dataset only
        sample_result, sample_time = time_function(
            ui.load_data_unified, None, "Iris Classification"
        )
        results['sample_only'] = {
            'time_ms': sample_time,
            'result_length': len(sample_result) if hasattr(sample_result, '__len__') else 0
        }
        
        # Test 2: File only
        test_data = test_data_generator('small')
        mock_file = Mock()
        mock_file.name = str(test_data['file_path'])
        
        file_result, file_time = time_function(
            ui.load_data_unified, mock_file, "Select a dataset..."
        )
        results['file_only'] = {
            'time_ms': file_time,
            'result_length': len(file_result) if hasattr(file_result, '__len__') else 0
        }
        
        # Test 3: Both provided (file should take priority)
        both_result, both_time = time_function(
            ui.load_data_unified, mock_file, "Iris Classification"
        )
        results['both_provided'] = {
            'time_ms': both_time,
            'result_length': len(both_result) if hasattr(both_result, '__len__') else 0
        }
        
        # Test 4: Neither provided
        neither_result, neither_time = time_function(
            ui.load_data_unified, None, "Select a dataset..."
        )
        results['neither_provided'] = {
            'time_ms': neither_time,
            'result_length': len(neither_result) if hasattr(neither_result, '__len__') else 0
        }
        
        benchmark_result.stop()
        benchmark_result.metadata = results
        
        # All operations should be fast
        for test_name, result in results.items():
            assert result['time_ms'] < 2000, f"{test_name} took {result['time_ms']}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.memory
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_ui_memory_usage(self, test_data_generator, memory_monitor, benchmark_result):
        """Test memory usage of UI operations."""
        benchmark_result.start()
        
        memory_monitor.sample()  # Baseline
        
        # Initialize UI
        ui, init_memory = measure_memory_usage(lambda: MLSystemUI(inference_only=True))
        memory_monitor.sample()
        
        # Load sample data
        sample_result, sample_memory = measure_memory_usage(
            ui.load_sample_data, "Iris Classification"
        )
        memory_monitor.sample()
        
        # Load file data
        test_data = test_data_generator('small')
        mock_file = Mock()
        mock_file.name = str(test_data['file_path'])
        
        file_result, file_memory = measure_memory_usage(ui.load_data, mock_file)
        memory_monitor.sample()
        
        # Generate multiple previews
        for i in range(3):
            ui.refresh_data_preview()
            memory_monitor.sample()
        
        benchmark_result.stop()
        
        monitor_stats = memory_monitor.get_stats()
        
        benchmark_result.metadata = {
            'init_memory_delta_mb': init_memory['delta_mb'],
            'sample_load_memory_delta_mb': sample_memory['delta_mb'],
            'file_load_memory_delta_mb': file_memory['delta_mb'],
            'memory_monitor_stats': monitor_stats,
            'peak_memory_mb': monitor_stats.get('peak_mb', 0)
        }
        
        # Memory usage should be reasonable
        assert monitor_stats.get('peak_mb', 0) < 500, f"Peak memory usage: {monitor_stats.get('peak_mb', 0)}MB"


class TestUIStressTests:
    """Stress tests for UI components."""
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_rapid_data_loading_stress(self, test_data_generator, benchmark_result, benchmark_config):
        """Test rapid consecutive data loading operations."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        iterations = min(benchmark_config['stress_iterations'], 20)  # Limit for UI tests
        
        # Prepare test data
        test_data = test_data_generator('tiny')
        mock_file = Mock()
        mock_file.name = str(test_data['file_path'])
        
        sample_datasets = ["Iris Classification", "Boston Housing", "Wine Quality"]
        
        operation_times = []
        memory_samples = []
        
        import psutil
        process = psutil.Process()
        
        for i in range(iterations):
            memory_before = process.memory_info().rss / 1024**2
            
            # Alternate between different operations
            if i % 3 == 0:
                # Sample data loading
                result, op_time = time_function(
                    ui.load_sample_data, 
                    sample_datasets[i % len(sample_datasets)]
                )
                op_type = 'sample'
            elif i % 3 == 1:
                # File data loading
                result, op_time = time_function(ui.load_data, mock_file)
                op_type = 'file'
            else:
                # Unified loading
                result, op_time = time_function(
                    ui.load_data_unified, 
                    mock_file, 
                    sample_datasets[i % len(sample_datasets)]
                )
                op_type = 'unified'
            
            operation_times.append((op_type, op_time))
            
            memory_after = process.memory_info().rss / 1024**2
            memory_samples.append(memory_after)
            
            # Periodic cleanup
            if i % 5 == 0:
                import gc
                gc.collect()
        
        benchmark_result.stop()
        
        # Analyze results
        by_type = {}
        for op_type, op_time in operation_times:
            if op_type not in by_type:
                by_type[op_type] = []
            by_type[op_type].append(op_time)
        
        memory_growth = memory_samples[-1] - memory_samples[0] if memory_samples else 0
        
        benchmark_result.metadata = {
            'iterations': iterations,
            'operation_times': operation_times,
            'performance_by_type': {
                op_type: {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times),
                    'max_ms': max(times),
                    'min_ms': min(times)
                }
                for op_type, times in by_type.items()
            },
            'memory_growth_mb': memory_growth,
            'final_memory_mb': memory_samples[-1] if memory_samples else 0
        }
        
        # Performance should remain stable
        assert memory_growth < 200, f"Memory grew by {memory_growth}MB during stress test"
        
        # Operations should complete reasonably quickly even under stress
        for op_type, times in by_type.items():
            avg_time = sum(times) / len(times)
            assert avg_time < 3000, f"{op_type} operations averaged {avg_time}ms"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")  
    def test_data_preview_refresh_stress(self, test_data_generator, benchmark_result):
        """Test rapid data preview refresh operations."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        
        # Load test data
        test_data = test_data_generator('small')
        ui.current_data = test_data['dataframe']
        
        refresh_times = []
        error_count = 0
        
        # Rapid refresh operations
        for i in range(30):
            try:
                result, refresh_time = time_function(ui.refresh_data_preview)
                refresh_times.append(refresh_time)
            except Exception as e:
                error_count += 1
                refresh_times.append(f"Error: {e}")
        
        benchmark_result.stop()
        
        valid_times = [t for t in refresh_times if isinstance(t, (int, float))]
        
        benchmark_result.metadata = {
            'total_refreshes': len(refresh_times),
            'successful_refreshes': len(valid_times),
            'error_count': error_count,
            'refresh_times_ms': refresh_times,
            'avg_refresh_time_ms': sum(valid_times) / len(valid_times) if valid_times else 0,
            'max_refresh_time_ms': max(valid_times) if valid_times else 0
        }
        
        # Most operations should succeed
        assert error_count < len(refresh_times) * 0.1, f"Too many errors: {error_count}/{len(refresh_times)}"
        
        # Performance should be consistent
        if valid_times:
            assert max(valid_times) < 5000, f"Slowest refresh: {max(valid_times)}ms"


@pytest.mark.benchmark
@pytest.mark.consistency
class TestUIConsistencyTests:
    """Test UI behavior consistency."""
    
    @pytest.mark.consistency
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_data_loading_consistency(self, test_data_generator, benchmark_result):
        """Test that UI data loading produces consistent results."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        
        # Test sample data consistency
        dataset_name = "Iris Classification"
        sample_results = []
        
        for i in range(5):
            result = ui.load_sample_data(dataset_name)
            if isinstance(result, tuple) and len(result) >= 4:
                # Extract data info from result
                data_info = result[0] if result[0] else ""
                metadata = result[1] if isinstance(result[1], dict) else {}
                
                sample_results.append({
                    'iteration': i,
                    'data_info_length': len(data_info),
                    'metadata_keys': sorted(metadata.keys()) if metadata else [],
                    'has_data': bool(data_info),
                    'result_length': len(result)
                })
        
        # Test file loading consistency
        test_data = test_data_generator('tiny')
        mock_file = Mock()
        mock_file.name = str(test_data['file_path'])
        
        file_results = []
        
        for i in range(3):
            result = ui.load_data(mock_file)
            if isinstance(result, tuple) and len(result) >= 4:
                data_info = result[0] if result[0] else ""
                metadata = result[1] if isinstance(result[1], dict) else {}
                
                file_results.append({
                    'iteration': i,
                    'data_info_length': len(data_info),
                    'metadata_keys': sorted(metadata.keys()) if metadata else [],
                    'has_data': bool(data_info),
                    'result_length': len(result)
                })
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'sample_data_results': sample_results,
            'file_data_results': file_results,
            'sample_consistency': len(set(str(r) for r in sample_results)) == 1,
            'file_consistency': len(set(str(r) for r in file_results)) == 1
        }
        
        # Results should be consistent
        if len(sample_results) > 1:
            first_sample = sample_results[0]
            for result in sample_results[1:]:
                assert result['data_info_length'] == first_sample['data_info_length'], \
                    "Sample data loading inconsistent"
        
        if len(file_results) > 1:
            first_file = file_results[0]
            for result in file_results[1:]:
                assert result['data_info_length'] == first_file['data_info_length'], \
                    "File data loading inconsistent"
    
    @pytest.mark.consistency
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_persistent_state_consistency(self, test_data_generator, benchmark_result):
        """Test that persistent UI state is maintained correctly."""
        benchmark_result.start()
        
        ui = MLSystemUI(inference_only=True)
        
        # Load data and check persistent state
        test_data = test_data_generator('small')
        mock_file = Mock()
        mock_file.name = str(test_data['file_path'])
        
        # Initial state
        initial_state = {
            'data_info': getattr(ui, 'current_data_info', ''),
            'data_preview': getattr(ui, 'current_data_preview', ''),
            'data_summary': getattr(ui, 'current_data_summary', {}),
            'current_data': getattr(ui, 'current_data', None)
        }
        
        # Load data
        ui.load_data(mock_file)
        
        # Check state after loading
        after_load_state = {
            'data_info': getattr(ui, 'current_data_info', ''),
            'data_preview': getattr(ui, 'current_data_preview', ''),
            'data_summary': getattr(ui, 'current_data_summary', {}),
            'current_data': getattr(ui, 'current_data', None)
        }
        
        # Refresh preview multiple times
        refresh_states = []
        for i in range(3):
            ui.refresh_data_preview()
            refresh_states.append({
                'iteration': i,
                'data_info': getattr(ui, 'current_data_info', ''),
                'data_preview': getattr(ui, 'current_data_preview', ''),
                'data_summary': getattr(ui, 'current_data_summary', {}),
            })
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'initial_state': {k: str(v)[:100] if v else '' for k, v in initial_state.items()},
            'after_load_state': {k: str(v)[:100] if v else '' for k, v in after_load_state.items()},
            'refresh_states': refresh_states,
            'state_changed_after_load': initial_state != after_load_state,
            'state_stable_after_refresh': len(set(str(s) for s in refresh_states)) <= 1
        }
        
        # State should change after loading
        assert initial_state != after_load_state, "UI state should change after data loading"
        
        # State should be stable after refresh operations
        if len(refresh_states) > 1:
            # Allow some variation in state but core data should be stable
            first_refresh = refresh_states[0]
            for state in refresh_states[1:]:
                # Check that data structure is maintained
                assert len(state['data_info']) == len(first_refresh['data_info']), \
                    "Data info length should be stable"
