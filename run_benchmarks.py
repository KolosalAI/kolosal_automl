# ---------------------------------------------------------------------
# run_benchmarks.py - Python script to run benchmark tests (cross-platform)
# ---------------------------------------------------------------------
"""
Cross-platform benchmark test runner for Kolosal AutoML with optimal device configuration.
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Optional

# Try to import device optimizer for optimal configuration
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.device_optimizer import DeviceOptimizer, OptimizationMode
    DEVICE_OPTIMIZER_AVAILABLE = True
except ImportError:
    DEVICE_OPTIMIZER_AVAILABLE = False

def setup_optimal_environment():
    """Setup optimal environment for benchmark execution."""
    if not DEVICE_OPTIMIZER_AVAILABLE:
        print("⚠️  Device optimizer not available, using default configuration")
        return {}
    
    try:
        print("🔧 Configuring optimal device settings for benchmarks...")
        
        # Create device optimizer for performance mode
        optimizer = DeviceOptimizer(
            optimization_mode=OptimizationMode.PERFORMANCE,
            workload_type="inference",  # Benchmarks are primarily inference-like
            environment="auto",
            enable_specialized_accelerators=True,
            memory_reservation_percent=5.0,  # Use more memory for benchmarks
            power_efficiency=False,  # Prioritize performance over power
            auto_tune=True,
            debug_mode=False
        )
        
        # Apply environment variables for optimal performance
        system_info = optimizer.get_system_info()
        cpu_cores = system_info.get('cpu_count_logical', os.cpu_count() or 1)
        physical_cores = system_info.get('cpu_count_physical', cpu_cores)
        
        # Set threading environment variables
        os.environ['OMP_NUM_THREADS'] = str(physical_cores)
        os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(physical_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(physical_cores)
        
        # Set process priority on Windows
        if os.name == 'nt':
            try:
                import psutil
                p = psutil.Process()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            except (ImportError, AttributeError, PermissionError):
                pass
        
        config_info = {
            'cpu_cores': cpu_cores,
            'physical_cores': physical_cores,
            'memory_gb': system_info.get('total_memory_gb', 0),
            'accelerators': system_info.get('accelerators', []),
            'optimization_mode': 'PERFORMANCE'
        }
        
        print(f"✅ Optimal configuration applied:")
        print(f"   CPU cores: {cpu_cores} (physical: {physical_cores})")
        print(f"   Memory: {config_info['memory_gb']:.1f} GB")
        print(f"   Accelerators: {config_info['accelerators']}")
        
        return config_info
        
    except Exception as e:
        print(f"⚠️  Failed to setup optimal environment: {e}")
        return {}

def create_output_dir(output_path: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def build_pytest_command(args) -> List[str]:
    """Build pytest command based on arguments."""
    cmd = ["python", "-m", "pytest"]
    
    # Basic arguments
    cmd.extend(["-ra", "--strict-markers", "--disable-warnings", "--tb=short"])
    
    # Verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    cmd.append("--durations=10")
    
    # Build marker expression
    markers = []
    
    if args.quick:
        markers.append("benchmark and not stress and not memory")
        print("⚡ Running quick benchmarks only")
    elif args.stress:
        markers.append("benchmark and stress")
        print("💪 Running stress tests")
    elif args.memory:
        markers.append("benchmark and memory")
        print("🧠 Running memory tests")
    else:
        markers.append("benchmark")
    
    # Add category filter
    if args.category != "all":
        category_map = {
            "data_loading": "data_loading",
            "ui": "ui", 
            "imports": "imports",
            "ml": "ml",
            "throughput": "throughput",
            "memory": "memory", 
            "inference": "inference"
        }
        
        if args.category.lower() in category_map:
            markers.append(category_map[args.category.lower()])
            print(f"🎯 Running category: {args.category}")
        else:
            print(f"❌ Unknown category: {args.category}")
            print("   Valid categories: all, data_loading, ui, imports, ml, throughput, memory, inference")
            sys.exit(1)
    
    # Add marker filter
    if markers:
        marker_expr = " and ".join(markers)
        cmd.extend(["-m", marker_expr])
        print(f"🔍 Marker expression: {marker_expr}")
    
    # Note: Timeout is configured in pytest.ini, not as command line argument
    # Different test types use the default 300s timeout from pytest.ini
    
    # Add size-based filters
    if args.size == "quick":
        cmd.extend(["-k", "not large and not huge and not stress"])
    elif args.size == "large":
        cmd.extend(["-k", "large or huge or stress"])
    
    # Add test path
    cmd.append("tests/benchmark/")
    
    return cmd

def print_header():
    """Print script header."""
    print("🚀 Kolosal AutoML Benchmark Runner (Optimized)")
    print("=" * 45)

def print_summary(start_time: float, exit_code: int, output_dir: Path, config_info: dict = None):
    """Print execution summary."""
    duration = time.time() - start_time
    
    print()
    print("📋 Benchmark Summary")
    print("=" * 20)
    print(f"Duration: {duration/60:.2f} minutes")
    print(f"Exit Code: {exit_code}")
    
    if config_info:
        print()
        print("🔧 Device Configuration:")
        print(f"   CPU cores used: {config_info.get('cpu_cores', 'N/A')}")
        print(f"   Memory available: {config_info.get('memory_gb', 'N/A'):.1f} GB")
        print(f"   Optimization mode: {config_info.get('optimization_mode', 'N/A')}")
    
    if exit_code == 0:
        print("Status: ✅ All tests passed")
    else:
        print("Status: ❌ Some tests failed")
    
    # Show recent results
    result_files = list(output_dir.glob("benchmark_results_*.json"))
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if result_files:
        print()
        print("📄 Recent result files:")
        for file in result_files[:3]:
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file.stat().st_mtime))
            print(f"   {file.name} ({mtime})")
    
    # Performance tips
    if exit_code != 0:
        print()
        print("💡 Troubleshooting tips:")
        print("   • Try running with --quick for faster tests")
        print("   • Use --verbose for detailed output")
        print("   • Check logs in tests/test.log")
        print("   • Run specific categories to isolate issues")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Kolosal AutoML benchmark tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py                          # Run all normal benchmarks
  python run_benchmarks.py --category data_loading  # Run data loading benchmarks
  python run_benchmarks.py --quick                  # Run quick benchmarks only
  python run_benchmarks.py --stress                 # Run stress tests
  python run_benchmarks.py --memory --verbose       # Run memory tests with verbose output

Categories:
  all          - All benchmark tests
  data_loading - Data I/O and processing benchmarks
  ui           - User interface performance benchmarks
  imports      - Import and startup benchmarks
  ml           - Machine learning operation benchmarks
  throughput   - Throughput and concurrent processing benchmarks
  memory       - Memory usage and leak detection benchmarks
  inference    - Inference engine performance benchmarks
        """
    )
    
    parser.add_argument(
        "--category", 
        default="all",
        choices=["all", "data_loading", "ui", "imports", "ml", "throughput", "memory", "inference"],
        help="Run specific benchmark category"
    )
    
    parser.add_argument(
        "--size",
        default="normal", 
        choices=["quick", "normal", "large"],
        help="Test data size"
    )
    
    parser.add_argument(
        "--output",
        default="tests/benchmark/results",
        help="Results output directory"
    )
    
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress tests (long running)"
    )
    
    parser.add_argument(
        "--memory", 
        action="store_true",
        help="Run memory tests"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true", 
        help="Run quick tests only (exclude stress and memory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Setup optimal device configuration
    config_info = setup_optimal_environment()
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"📁 Output directory: {output_dir}")
    
    # Build command
    cmd = build_pytest_command(args)
    
    print()
    print("📊 Test configuration:")
    print(f"   Category: {args.category}")
    print(f"   Size: {args.size}")
    print(f"   Output: {args.output}")
    print()
    
    print("🔄 Starting benchmark tests...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run tests
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        exit_code = 1
    
    print_summary(start_time, exit_code, output_dir, config_info)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
