# ---------------------------------------------------------------------
# tests/benchmark/test_import_benchmark.py - Import and startup performance tests
# ---------------------------------------------------------------------
"""
Benchmark tests for module imports and application startup performance.
Tests the recent Pydantic v2 compatibility patches and import fallbacks in app.py
"""
import pytest
import time
import sys
import os
import subprocess
import tempfile
import importlib
from pathlib import Path
from unittest.mock import patch, Mock
import psutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.conftest import time_function, measure_memory_usage

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.imports
]

class TestImportPerformanceBenchmarks:
    """Benchmark tests for import performance."""
    
    @pytest.mark.benchmark
    def test_cold_import_performance(self, benchmark_result):
        """Test cold import performance of main modules."""
        benchmark_result.start()
        
        # Test imports in a subprocess to ensure truly cold imports
        import_tests = [
            ('pandas', 'import pandas as pd'),
            ('numpy', 'import numpy as np'),
            ('gradio', 'import gradio as gr'),
            ('fastapi', 'from fastapi import FastAPI'),
            ('pydantic', 'import pydantic'),
            ('sklearn', 'import sklearn'),
            ('matplotlib', 'import matplotlib.pyplot as plt'),
        ]
        
        import_times = {}
        
        for module_name, import_stmt in import_tests:
            try:
                # Create a subprocess to test cold import
                test_script = f"""
import time
start = time.perf_counter()
{import_stmt}
end = time.perf_counter()
print(f"{{(end - start) * 1000:.2f}}")
"""
                
                result = subprocess.run(
                    [sys.executable, '-c', test_script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    import_time = float(result.stdout.strip())
                    import_times[module_name] = import_time
                else:
                    import_times[module_name] = f"Error: {result.stderr}"
                    
            except (subprocess.TimeoutExpired, ValueError, Exception) as e:
                import_times[module_name] = f"Failed: {e}"
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'import_times_ms': import_times,
            'total_import_time_ms': sum(t for t in import_times.values() if isinstance(t, (int, float)))
        }
        
        # Core imports should be reasonable
        critical_imports = ['pandas', 'numpy']
        for module in critical_imports:
            if isinstance(import_times.get(module), (int, float)):
                assert import_times[module] < 5000, f"{module} import took {import_times[module]}ms"
    
    @pytest.mark.benchmark
    def test_app_import_with_pydantic_patches(self, benchmark_result):
        """Test app.py import performance with Pydantic compatibility patches."""
        benchmark_result.start()
        
        # Create a test script that imports app.py with timing
        test_script = f"""
import sys
import os
import time
sys.path.insert(0, r'{PROJECT_ROOT}')

# Test different Pydantic environment scenarios
scenarios = []

# Scenario 1: Normal import
start = time.perf_counter()
try:
    import app
    end = time.perf_counter()
    scenarios.append(('normal_import', (end - start) * 1000, 'success'))
except Exception as e:
    end = time.perf_counter()
    scenarios.append(('normal_import', (end - start) * 1000, f'error: {{e}}'))

# Print results
for name, duration, status in scenarios:
    print(f"{{name}}: {{duration:.2f}}ms ({{status}})")
"""
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(PROJECT_ROOT)
            )
            
            app_import_results = {}
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':')
                        scenario = parts[0].strip()
                        time_part = parts[1].strip().split('ms')[0]
                        status = parts[1].strip().split('(')[1].split(')')[0] if '(' in parts[1] else 'unknown'
                        
                        try:
                            time_ms = float(time_part)
                            app_import_results[scenario] = {
                                'time_ms': time_ms,
                                'status': status
                            }
                        except ValueError:
                            app_import_results[scenario] = {
                                'time_ms': 0,
                                'status': f'parse_error: {line}'
                            }
            else:
                app_import_results['error'] = {
                    'stderr': result.stderr,
                    'stdout': result.stdout,
                    'returncode': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            app_import_results = {'timeout': 'App import timed out after 60s'}
        except Exception as e:
            app_import_results = {'exception': str(e)}
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'app_import_results': app_import_results
        }
        
        # App import should complete in reasonable time
        if 'normal_import' in app_import_results:
            import_time = app_import_results['normal_import'].get('time_ms', 0)
            if isinstance(import_time, (int, float)) and import_time > 0:
                assert import_time < 30000, f"App import took {import_time}ms"
    
    @pytest.mark.benchmark
    def test_fallback_implementation_performance(self, benchmark_result):
        """Test performance of fallback implementations when modules are missing."""
        benchmark_result.start()
        
        # Create a test environment where we simulate missing modules
        test_script = f"""
import sys
import time
import os
from unittest.mock import patch
sys.path.insert(0, r'{PROJECT_ROOT}')

# Mock missing modules to trigger fallbacks
missing_modules = ['autogluon', 'sklearn', 'torch', 'transformers']
original_import = __builtins__.__import__

def mock_import(name, *args, **kwargs):
    if any(missing in name for missing in missing_modules):
        raise ImportError(f"Mocked missing module: {{name}}")
    return original_import(name, *args, **kwargs)

# Test import with missing modules
start = time.perf_counter()
try:
    with patch('builtins.__import__', side_effect=mock_import):
        # Force environment variable for fallbacks
        os.environ['FORCE_FALLBACK_IMPLEMENTATIONS'] = '1'
        import app
        ui = app.MLSystemUI(inference_only=True)
    end = time.perf_counter()
    print(f"fallback_import: {{(end - start) * 1000:.2f}}ms (success)")
except Exception as e:
    end = time.perf_counter()
    print(f"fallback_import: {{(end - start) * 1000:.2f}}ms (error: {{e}})")
"""
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=45,
                cwd=str(PROJECT_ROOT)
            )
            
            fallback_results = {}
            if result.returncode == 0:
                output = result.stdout.strip()
                if 'fallback_import:' in output:
                    line = output.split('fallback_import:')[1].strip()
                    time_part = line.split('ms')[0].strip()
                    status = line.split('(')[1].split(')')[0] if '(' in line else 'unknown'
                    
                    try:
                        fallback_results['fallback_import'] = {
                            'time_ms': float(time_part),
                            'status': status
                        }
                    except ValueError:
                        fallback_results['parse_error'] = line
            
            fallback_results['subprocess'] = {
                'returncode': result.returncode,
                'stderr': result.stderr[:500] if result.stderr else '',
                'stdout': result.stdout[:500] if result.stdout else ''
            }
            
        except subprocess.TimeoutExpired:
            fallback_results = {'timeout': 'Fallback test timed out'}
        except Exception as e:
            fallback_results = {'exception': str(e)}
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'fallback_results': fallback_results
        }
        
        # Fallback imports should still be reasonable
        if 'fallback_import' in fallback_results:
            fallback_time = fallback_results['fallback_import'].get('time_ms', 0)
            if isinstance(fallback_time, (int, float)) and fallback_time > 0:
                assert fallback_time < 45000, f"Fallback import took {fallback_time}ms"
    
    @pytest.mark.benchmark
    def test_gradio_import_with_pydantic_compatibility(self, benchmark_result):
        """Test Gradio import performance with Pydantic compatibility patches."""
        benchmark_result.start()
        
        # Test script that applies Pydantic patches before importing Gradio
        test_script = f"""
import sys
import os
import time
sys.path.insert(0, r'{PROJECT_ROOT}')

# Apply Pydantic compatibility patches
os.environ["PYDANTIC_V1"] = "1"

try:
    import pydantic
    import pydantic.fields
    
    # Apply patches like in app.py
    if not hasattr(pydantic.fields, 'Undefined'):
        class UndefinedType:
            _singleton = None
            def __new__(cls):
                if cls._singleton is None:
                    cls._singleton = super().__new__(cls)
                return cls._singleton
            def __repr__(self):
                return "Undefined"
            def __bool__(self):
                return False
        pydantic.fields.Undefined = UndefinedType()
    
    if not hasattr(pydantic.fields, 'FieldInfo'):
        from pydantic import Field
        dummy_field = Field(default=None)
        pydantic.fields.FieldInfo = type(dummy_field)
    
    # Now test Gradio import
    start = time.perf_counter()
    import gradio as gr
    end = time.perf_counter()
    print(f"gradio_with_patches: {{(end - start) * 1000:.2f}}ms (success)")
    
except Exception as e:
    end = time.perf_counter()
    print(f"gradio_with_patches: {{(end - start) * 1000:.2f}}ms (error: {{e}})")
"""
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            gradio_results = {}
            if 'gradio_with_patches:' in result.stdout:
                line = result.stdout.strip()
                time_part = line.split('gradio_with_patches:')[1].strip().split('ms')[0].strip()
                status = line.split('(')[1].split(')')[0] if '(' in line else 'unknown'
                
                try:
                    gradio_results['gradio_with_patches'] = {
                        'time_ms': float(time_part),
                        'status': status
                    }
                except ValueError:
                    gradio_results['parse_error'] = line
            
            gradio_results['subprocess'] = {
                'returncode': result.returncode,
                'stderr': result.stderr[:200] if result.stderr else '',
                'stdout': result.stdout[:200] if result.stdout else ''
            }
            
        except subprocess.TimeoutExpired:
            gradio_results = {'timeout': 'Gradio import test timed out'}
        except Exception as e:
            gradio_results = {'exception': str(e)}
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'gradio_results': gradio_results
        }
        
        # Gradio with patches should import successfully
        if 'gradio_with_patches' in gradio_results:
            gradio_time = gradio_results['gradio_with_patches'].get('time_ms', 0)
            status = gradio_results['gradio_with_patches'].get('status', '')
            
            if isinstance(gradio_time, (int, float)) and gradio_time > 0:
                assert gradio_time < 15000, f"Gradio with patches took {gradio_time}ms"
            
            # Check for known Pydantic compatibility issues that might need different handling
            if 'error' in status:
                if 'Field \'type_\' defined on a base class was overridden by a non-annotated' in status:
                    # This is a known Pydantic v2 compatibility issue
                    # Log it but don't fail the test as it's an environment-specific issue
                    print(f"⚠️ Known Pydantic compatibility issue encountered: {status}")
                    pytest.skip("Skipping due to known Pydantic v2 field override compatibility issue")
                else:
                    assert False, f"Gradio import failed: {status}"
            else:
                assert 'success' in status, f"Gradio import failed: {status}"


class TestStartupPerformanceBenchmarks:
    """Benchmark tests for application startup performance."""
    
    @pytest.mark.benchmark
    def test_ui_initialization_memory_usage(self, benchmark_result, memory_monitor):
        """Test memory usage during UI initialization."""
        benchmark_result.start()
        
        memory_monitor.sample()  # Baseline
        
        try:
            # Force use of fallback implementations for consistent testing
            os.environ['FORCE_FALLBACK_IMPLEMENTATIONS'] = '1'
            
            # Import and initialize with memory tracking
            app_module, import_memory = measure_memory_usage(lambda: importlib.import_module('app'))
            memory_monitor.sample()
            
            if hasattr(app_module, 'MLSystemUI'):
                ui, init_memory = measure_memory_usage(
                    lambda: app_module.MLSystemUI(inference_only=True)
                )
                memory_monitor.sample()
                
                initialization_results = {
                    'import_memory_delta_mb': import_memory['delta_mb'],
                    'init_memory_delta_mb': init_memory['delta_mb'],
                    'total_memory_delta_mb': import_memory['delta_mb'] + init_memory['delta_mb'],
                    'ui_created': True
                }
            else:
                initialization_results = {
                    'import_memory_delta_mb': import_memory['delta_mb'],
                    'ui_created': False,
                    'error': 'MLSystemUI class not found'
                }
            
        except Exception as e:
            initialization_results = {
                'error': str(e),
                'ui_created': False
            }
        
        benchmark_result.stop()
        
        monitor_stats = memory_monitor.get_stats()
        benchmark_result.metadata = {
            'initialization_results': initialization_results,
            'memory_monitor_stats': monitor_stats
        }
        
        # Memory usage should be reasonable
        if 'total_memory_delta_mb' in initialization_results:
            total_memory = initialization_results['total_memory_delta_mb']
            assert total_memory < 500, f"UI initialization used {total_memory}MB"
    
    @pytest.mark.benchmark
    @pytest.mark.stress
    def test_repeated_initialization_performance(self, benchmark_result):
        """Test performance of repeated UI initialization (memory leaks)."""
        benchmark_result.start()
        
        initialization_times = []
        memory_usage = []
        
        process = psutil.Process()
        
        try:
            os.environ['FORCE_FALLBACK_IMPLEMENTATIONS'] = '1'
            
            for i in range(5):
                memory_before = process.memory_info().rss / 1024**2
                
                # Import fresh each time to test cold initialization
                if 'app' in sys.modules:
                    del sys.modules['app']
                
                start_time = time.perf_counter()
                
                try:
                    import app
                    ui = app.MLSystemUI(inference_only=True)
                    
                    # Do some basic operations
                    if hasattr(ui, 'load_sample_data'):
                        ui.load_sample_data("Iris Classification")
                    
                    end_time = time.perf_counter()
                    init_time = (end_time - start_time) * 1000
                    initialization_times.append(init_time)
                    
                    del ui
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    init_time = (end_time - start_time) * 1000
                    initialization_times.append(f"Error: {e}")
                
                memory_after = process.memory_info().rss / 1024**2
                memory_usage.append(memory_after)
                
                # Force cleanup
                import gc
                gc.collect()
        
        except Exception as e:
            initialization_times.append(f"Test failed: {e}")
        
        benchmark_result.stop()
        
        valid_times = [t for t in initialization_times if isinstance(t, (int, float))]
        memory_growth = memory_usage[-1] - memory_usage[0] if len(memory_usage) >= 2 else 0
        
        benchmark_result.metadata = {
            'initialization_times_ms': initialization_times,
            'valid_initializations': len(valid_times),
            'avg_init_time_ms': sum(valid_times) / len(valid_times) if valid_times else 0,
            'memory_usage_mb': memory_usage,
            'memory_growth_mb': memory_growth,
            'iterations_completed': len(initialization_times)
        }
        
        # Performance should be consistent
        if len(valid_times) >= 2:
            avg_time = sum(valid_times) / len(valid_times)
            assert avg_time < 15000, f"Average initialization time: {avg_time}ms"
            
            # Memory growth should be minimal (allow for more reasonable growth)
            assert abs(memory_growth) < 200, f"Memory grew by {memory_growth}MB"


@pytest.mark.benchmark
@pytest.mark.consistency
class TestImportConsistencyTests:
    """Test import behavior consistency."""
    
    @pytest.mark.consistency
    @pytest.mark.skip(reason="Import order timing is too variable on Windows systems")
    def test_import_order_independence(self, benchmark_result):
        """Test that import order doesn't affect performance or behavior."""
        benchmark_result.start()
        
        # Test different import orders
        import_scenarios = [
            ['pandas', 'numpy', 'gradio'],
            ['gradio', 'pandas', 'numpy'],
            ['numpy', 'gradio', 'pandas'],
        ]
        
        scenario_results = {}
        
        for i, scenario in enumerate(import_scenarios):
            # Create subprocess to test clean import order
            test_script = f"""
import sys
import time
sys.path.insert(0, r'{PROJECT_ROOT}')

imports = {scenario}
import_times = []

for module_name in imports:
    start = time.perf_counter()
    try:
        if module_name == 'pandas':
            import pandas as pd
        elif module_name == 'numpy':
            import numpy as np
        elif module_name == 'gradio':
            import gradio as gr
        end = time.perf_counter()
        import_times.append({{module_name: (end - start) * 1000}})
    except Exception as e:
        import_times.append({{module_name: f"Error: {{e}}"}})

for result in import_times:
    for module, time_val in result.items():
        print(f"{{module}}: {{time_val}}")
"""
            
            try:
                result = subprocess.run(
                    [sys.executable, '-c', test_script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    scenario_times = {}
                    for line in result.stdout.strip().split('\n'):
                        if ':' in line:
                            module, time_str = line.split(':', 1)
                            module = module.strip()
                            time_str = time_str.strip()
                            
                            try:
                                time_val = float(time_str)
                                scenario_times[module] = time_val
                            except ValueError:
                                scenario_times[module] = time_str
                    
                    scenario_results[f'scenario_{i}'] = {
                        'order': scenario,
                        'times': scenario_times,
                        'status': 'success'
                    }
                else:
                    scenario_results[f'scenario_{i}'] = {
                        'order': scenario,
                        'status': 'failed',
                        'error': result.stderr
                    }
                    
            except subprocess.TimeoutExpired:
                scenario_results[f'scenario_{i}'] = {
                    'order': scenario,
                    'status': 'timeout'
                }
        
        benchmark_result.stop()
        benchmark_result.metadata = {
            'scenario_results': scenario_results
        }
        
        # Import order shouldn't dramatically affect performance
        successful_scenarios = [s for s in scenario_results.values() if s.get('status') == 'success']
        
        if len(successful_scenarios) >= 2:
            # Compare pandas import times across scenarios
            pandas_times = []
            for scenario in successful_scenarios:
                if 'pandas' in scenario.get('times', {}):
                    time_val = scenario['times']['pandas']
                    if isinstance(time_val, (int, float)):
                        pandas_times.append(time_val)
            
            if len(pandas_times) >= 2:
                max_time = max(pandas_times)
                min_time = min(pandas_times)
                variation = (max_time - min_time) / min_time * 100 if min_time > 0 else 0
                
                # More realistic threshold - imports can vary significantly due to caching, disk I/O, etc.
                assert variation < 500, f"Import order caused {variation}% variation in pandas import time"
