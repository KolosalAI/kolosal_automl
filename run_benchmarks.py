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
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob

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

def run_benchmark_suite(test_categories: List[str], output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Run comprehensive benchmark suite with multiple categories."""
    
    results = {
        'benchmark_suite': {
            'categories_run': test_categories,
            'start_time': time.time(),
            'results': []
        }
    }
    
    for category in test_categories:
        print(f"\n🏃 Running {category} benchmarks...")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        cmd.extend(["-ra", "--strict-markers", "--disable-warnings", "--tb=short"])
        
        if verbose:
            cmd.extend(["-v", "-s"])
        else:
            cmd.append("-q")
        
        cmd.extend(["--durations=10", "-m", f"benchmark and {category}"])
        cmd.append("tests/benchmark/")
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
            duration = time.time() - start_time
            
            category_result = {
                'category': category,
                'duration_seconds': duration,
                'exit_code': result.returncode,
                'tests_passed': result.returncode == 0,
                'stdout': result.stdout if verbose else None,
                'stderr': result.stderr if result.stderr else None
            }
            
            results['benchmark_suite']['results'].append(category_result)
            
            if result.returncode == 0:
                print(f"   ✅ {category} benchmarks completed successfully in {duration:.2f}s")
            else:
                print(f"   ❌ {category} benchmarks failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                    
        except Exception as e:
            print(f"   ❌ Failed to run {category} benchmarks: {e}")
            results['benchmark_suite']['results'].append({
                'category': category,
                'duration_seconds': time.time() - start_time,
                'exit_code': -1,
                'tests_passed': False,
                'error': str(e)
            })
    
    results['benchmark_suite']['end_time'] = time.time()
    results['benchmark_suite']['total_duration'] = results['benchmark_suite']['end_time'] - results['benchmark_suite']['start_time']
    
    return results

def consolidate_results(output_dir: str) -> Optional[Path]:
    """Find the most recent benchmark results and return path."""
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None
    
    # Find the most recent results file
    pattern = results_dir / "benchmark_results_*.json"
    result_files = list(results_dir.glob("benchmark_results_*.json"))
    
    if not result_files:
        return None
    
    # Get the most recent file
    most_recent = max(result_files, key=lambda x: x.stat().st_mtime)
    return most_recent

def print_comprehensive_summary(results_file: Path):
    """Print a comprehensive summary of benchmark results."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📊 COMPREHENSIVE BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        # Session info
        session = data.get('benchmark_session', {})
        print(f"🕒 Session Duration: {session.get('session_duration_minutes', 0):.2f} minutes")
        print(f"📅 Timestamp: {session.get('timestamp', 'Unknown')}")
        
        # System info
        system = data.get('system_info', {})
        print(f"💻 System: {system.get('cpu_count_logical', 'Unknown')} cores, {system.get('memory_total_gb', 0):.1f} GB RAM")
        
        # Summary stats
        summary = session.get('summary', {})
        print(f"🧪 Total Tests: {summary.get('total_tests_run', 0)}")
        
        # Tests by category
        print(f"\n📈 Tests by Category:")
        categories = summary.get('tests_by_category', {})
        for category, count in categories.items():
            if count > 0:
                category_name = category.replace('_', ' ').title()
                print(f"   • {category_name}: {count} tests")
        
        # Performance summary
        perf = summary.get('performance_summary', {})
        if perf:
            print(f"\n⚡ Performance Summary:")
            if 'fastest_test_ms' in perf:
                print(f"   • Fastest test: {perf['fastest_test_ms']:.2f}ms")
            if 'slowest_test_ms' in perf:
                print(f"   • Slowest test: {perf['slowest_test_ms']:.2f}ms")
            if 'avg_memory_usage_mb' in perf:
                print(f"   • Avg memory usage: {perf['avg_memory_usage_mb']:.2f}MB")
        
        # Category breakdown
        categories = data.get('test_results_by_category', {})
        print(f"\n📋 Detailed Results by Category:")
        
        for category, tests in categories.items():
            if tests:
                category_name = category.replace('_', ' ').title()
                print(f"\n   {category_name} ({len(tests)} tests):")
                
                for test in tests[:3]:  # Show first 3 tests per category
                    test_name = test.get('name', 'Unknown')
                    duration = test.get('duration_ms', 0)
                    print(f"     • {test_name}: {duration:.2f}ms")
                
                if len(tests) > 3:
                    print(f"     ... and {len(tests) - 3} more tests")
        
        print(f"\n✅ Complete results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Failed to read results summary: {e}")

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
    elif args.large:
        markers.append("benchmark and large")
        print("📊 Running large-scale benchmarks")
    elif args.extreme:
        markers.append("benchmark and extreme")
        print("🚀 Running extreme load benchmarks")
    elif args.full:
        # Run all benchmark tests including throughput, latency, and inference
        markers.append("benchmark")
        print("🚀 Running FULL benchmark suite (all throughput, latency, and performance tests)")
    else:
        # Default behavior: run all benchmark tests
        markers.append("benchmark")
        print("🔥 Running ALL benchmark tests (throughput, latency, inference, concurrency, memory, etc.)")
    
    # Add category filter
    if args.category != "all":
        category_map = {
            "data_loading": "data_loading",
            "ui": "ui", 
            "imports": "imports",
            "ml": "ml",
            "throughput": "throughput",
            "latency": "latency",
            "memory": "memory", 
            "inference": "inference",
            "concurrency": "concurrency",
            "large": "large",
            "extreme": "extreme"
        }
        
        if args.category.lower() in category_map:
            markers.append(category_map[args.category.lower()])
            print(f"🎯 Running category: {args.category}")
        else:
            print(f"❌ Unknown category: {args.category}")
            print("   Valid categories: all, data_loading, ui, imports, ml, throughput, latency, memory, inference, concurrency, large, extreme")
            sys.exit(1)
    else:
        # When running "all", ensure we include all benchmark types
        print("🔥 Running ALL benchmark categories: throughput, latency, inference, concurrency, memory, and more")
    
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
  python run_benchmarks.py                          # Run all normal benchmarks (all categories)
  python run_benchmarks.py --full                   # Run complete benchmark suite (recommended)
  python run_benchmarks.py --comprehensive          # Run comprehensive multi-category suite with detailed reporting
  python run_benchmarks.py --categories throughput latency  # Run specific categories with comprehensive reporting
  python run_benchmarks.py --category throughput    # Run throughput benchmarks only
  python run_benchmarks.py --category latency       # Run latency benchmarks only
  python run_benchmarks.py --category large         # Run large-scale benchmarks
  python run_benchmarks.py --category extreme       # Run extreme load benchmarks
  python run_benchmarks.py --quick                  # Run quick benchmarks only
  python run_benchmarks.py --stress                 # Run stress tests
  python run_benchmarks.py --large                  # Run large-scale benchmarks
  python run_benchmarks.py --extreme                # Run extreme load benchmarks
  python run_benchmarks.py --memory --verbose       # Run memory tests with verbose output

Categories:
  all          - All benchmark tests (throughput, latency, inference, etc.)
  data_loading - Data I/O and processing benchmarks
  ui           - User interface performance benchmarks
  imports      - Import and startup benchmarks
  ml           - Machine learning operation benchmarks
  throughput   - Throughput and concurrent processing benchmarks
  latency      - Latency and response time benchmarks
  memory       - Memory usage and leak detection benchmarks
  inference    - Inference engine performance benchmarks
  concurrency  - Concurrency and parallel processing benchmarks
  large        - Large scale data processing and high volume benchmarks
  extreme      - Extreme load, massive concurrency, and system limit benchmarks
  stress       - Stress testing for system stability

Comprehensive Mode:
  The --comprehensive flag or specifying multiple --categories will run a comprehensive
  benchmark suite that executes multiple categories sequentially and provides detailed
  consolidated reporting with performance summaries and category breakdowns.
        """
    )
    
    parser.add_argument(
        "--categories",
        nargs="*",
        default=["all"],
        choices=["all", "throughput", "latency", "inference", "concurrency", "memory", "stress", "data_loading", "ui", "imports", "ml", "large", "extreme"],
        help="Benchmark categories to run (default: all). Can specify multiple categories."
    )
    
    parser.add_argument(
        "--category", 
        default="all",
        choices=["all", "data_loading", "ui", "imports", "ml", "throughput", "latency", "memory", "inference", "concurrency", "large", "extreme"],
        help="Run specific benchmark category (legacy - use --categories instead)"
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
        "--full",
        action="store_true",
        help="Run complete benchmark suite (all throughput, latency, inference, and performance tests)"
    )
    
    parser.add_argument(
        "--large",
        action="store_true",
        help="Run large-scale benchmarks (high volume data processing)"
    )
    
    parser.add_argument(
        "--extreme",
        action="store_true",
        help="Run extreme load benchmarks (massive concurrency and system limits)"
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive benchmark suite with detailed reporting and multi-category execution"
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
    
    # Determine if running in comprehensive mode
    if args.comprehensive or (hasattr(args, 'categories') and args.categories != ["all"] and len(args.categories) > 1):
        # Run comprehensive benchmark suite
        print("🚀 Running COMPREHENSIVE benchmark suite")
        
        # Determine categories to run
        if hasattr(args, 'categories') and args.categories != ["all"]:
            categories = args.categories
        elif "all" in getattr(args, 'categories', []):
            if args.quick:
                categories = ["throughput", "latency", "inference", "concurrency"]
            else:
                categories = ["throughput", "latency", "inference", "concurrency", "memory", "stress"]
        else:
            # Default comprehensive categories
            categories = ["throughput", "latency", "inference", "concurrency", "memory", "stress"]
        
        print(f"📊 Running benchmark categories: {', '.join(categories)}")
        
        # Run benchmark suite
        start_time = time.time()
        suite_results = run_benchmark_suite(categories, args.output, args.verbose)
        
        # Find and display results
        print(f"\n🔍 Looking for consolidated results...")
        results_file = consolidate_results(args.output)
        
        if results_file:
            print_comprehensive_summary(results_file)
        else:
            print("❌ No benchmark results file found")
        
        # Check if all categories passed
        all_passed = all(result.get('tests_passed', False) for result in suite_results['benchmark_suite']['results'])
        
        # Final summary
        total_duration = time.time() - start_time
        print(f"\n⏱️  Total execution time: {total_duration / 60:.2f} minutes")
        
        if all_passed:
            print("🎉 All benchmark categories completed successfully!")
            sys.exit(0)
        else:
            print("⚠️  Some benchmark categories had failures")
            sys.exit(1)
    
    else:
        # Run single category mode (original behavior)
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
