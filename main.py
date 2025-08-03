#!/usr/bin/env python3
"""
kolosal AutoML Training & Inference System3. 📊 BENCHMARKS & COMPARISONS
   - Standard ML baseline benchmarks
   - Kolosal AutoML advanced benchmarks
   - Head-to-head performance comparisons
   - Multiple datasets and optimization strategies
   - Comprehensive analysis and reporting
   - Perfect for performance evaluation

4. ⚖️ ML COMPARISON TOOLS
   - Direct Standard ML vs AutoML comparisons
   - Optimization strategy comparisons
   - Scalability testing across dataset sizes
   - Statistical analysis with visualizations
   - Custom comparison configurations
   - Perfect for research and validation

5. 📋 SYSTEM INFO v0.1.4
ML Main CLI

Main entry point for the kolosal AutoML system. Allows users to choose between
running the Gradio web interface or starting the API server.

Usage:
    python main.py
    python main.py --mode gui
    python main.py --mode api
    python main.py --help
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import msvcrt  # For Windows key detection
import time

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██╗  ██╗ ██████╗ ██╗      ██████╗ ███████╗ █████╗ ██╗         █████╗ ██╗    ║
║  ██║ ██╔╝██╔═══██╗██║     ██╔═══██╗██╔════╝██╔══██╗██║        ██╔══██╗██║    ║
║  █████╔╝ ██║   ██║██║     ██║   ██║███████╗███████║██║  █████╗███████║██║    ║
║  ██╔═██╗ ██║   ██║██║     ██║   ██║╚════██║██╔══██║██║  ╚════╝██╔══██║██║    ║
║  ██║  ██╗╚██████╔╝███████╗╚██████╔╝███████║██║  ██║███████╗   ██║  ██║██║    ║
║  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝  ╚═╝╚═╝    ║
║                                                                              ║
║                         AutoML Training & Inference System                   ║
║                                 Version v0.1.4                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_help():
    """Print help information."""
    help_text = """
🚀 Welcome to kolosal AutoML!

Choose how you want to run the system:

📊 OPTIONS:

1. 🌐 WEB INTERFACE (Gradio UI)
   - Interactive web-based interface
   - Visual data upload and configuration
   - Model training with real-time progress
   - Drag-and-drop file uploads
   - Performance visualizations
   - Perfect for experimentation and prototyping

2. 🔧 API SERVER (REST API)
   - RESTful API endpoints
   - Programmatic access to all features
   - JSON-based data exchange
   - Scalable for production use
   - Perfect for integration with other systems

3. � STANDARD ML BENCHMARK
   - Baseline performance benchmarks
   - Pure scikit-learn implementations
   - Performance comparison baseline
   - Multiple datasets and models
   - Perfect for performance validation

4. �📋 SYSTEM INFO
   - View detailed system information
   - Check hardware capabilities
   - Verify dependencies
   - Optimize system settings

💡 TIPS:
   - Use Web Interface for interactive machine learning
   - Use API Server for production deployments
   - Use Benchmarks for performance evaluation
   - Use Comparisons for research and validation
   - Both modes support all ML algorithms and features
   - Models trained in one mode can be used in the other

🔗 QUICK START:
   - Web Interface: http://localhost:7860 (default)
   - API Server: http://localhost:8000 (default)
   - API Documentation: http://localhost:8000/docs
   - Benchmark Results: ./benchmark_results/ directory
   - Comparison Results: ./comparison_results/ directory

📚 DOCUMENTATION:
   - Web Interface: Built-in help and examples
   - API: Interactive docs at /docs endpoint
   - Models: Saved in ./models/ directory
   - Logs: Available in respective .log files
   - Results: Comprehensive HTML reports generated
    """
    print(help_text)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import uvicorn
        import fastapi
        import gradio
        import pandas
        import numpy
        import sklearn
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install dependencies with: uv install")
        return False

def run_gradio_app(args: Optional[list] = None):
    """Run the Gradio web interface."""
    print("🌐 Starting Gradio Web Interface...")
    print("   - Interactive ML training and inference")
    print("   - Web-based UI at http://localhost:7860")
    print("   - Press Ctrl+C to stop")
    print()
    
    cmd = ["uv", "run", "python", "app.py"]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Gradio interface stopped.")
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv or run directly with python.")
        print("   Alternative: python app.py")

def run_api_server(args: Optional[list] = None):
    """Run the API server."""
    print("🔧 Starting API Server...")
    print("   - RESTful API endpoints")
    print("   - API server at http://localhost:8000")
    print("   - Interactive docs at http://localhost:8000/docs")
    print("   - Press Ctrl+C to stop")
    print()
    
    cmd = ["uv", "run", "python", "start_api.py"]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 API server stopped.")
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv or run directly with python.")
        print("   Alternative: python start_api.py")

def run_standard_ml_benchmark(args: Optional[list] = None):
    """Run the standard ML benchmark."""
    print("📊 Starting Standard ML Benchmark...")
    print("   - Pure scikit-learn baseline benchmarks")
    print("   - Performance comparison baseline")
    print("   - Results saved to ./standard_ml_results/")
    print("   - Press Ctrl+C to stop")
    print()
    
    cmd = ["uv", "run", "python", "benchmark/standard_ml_benchmark.py"]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Standard ML benchmark stopped.")
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv or run directly with python.")
        print("   Alternative: python benchmark/standard_ml_benchmark.py")

def run_kolosal_automl_benchmark(args: Optional[list] = None):
    """Run the Kolosal AutoML benchmark."""
    print("🚀 Starting Kolosal AutoML Benchmark...")
    print("   - Advanced AutoML training engine benchmarks")
    print("   - Optimization strategies comparison")
    print("   - Results saved to ./benchmark_results/")
    print("   - Press Ctrl+C to stop")
    print()
    
    cmd = ["uv", "run", "python", "benchmark/kolosal_automl_benchmark.py"]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Kolosal AutoML benchmark stopped.")
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv or run directly with python.")
        print("   Alternative: python benchmark/kolosal_automl_benchmark.py")

def run_comparison_benchmark(args: Optional[list] = None):
    """Run comparison between Standard ML and Kolosal AutoML."""
    print("⚖️ Starting Standard ML vs Kolosal AutoML Comparison...")
    print("   - Head-to-head performance comparison")
    print("   - Comprehensive analysis and reporting")
    print("   - Results saved to ./comparison_results/")
    print("   - Press Ctrl+C to stop")
    print()
    
    cmd = ["uv", "run", "python", "run_kolosal_comparison.py"]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Comparison benchmark stopped.")
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv or run directly with python.")
        print("   Alternative: python run_kolosal_comparison.py")

def run_benchmark_comparison_tool(args: Optional[list] = None):
    """Run the benchmark comparison tool."""
    print("📈 Starting Benchmark Comparison Tool...")
    print("   - Advanced comparison analysis")
    print("   - Statistical significance testing")
    print("   - Interactive visualizations")
    print("   - Press Ctrl+C to stop")
    print()
    
    cmd = ["uv", "run", "python", "benchmark/benchmark_comparison.py"]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Benchmark comparison tool stopped.")
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv or run directly with python.")
        print("   Alternative: python benchmark/benchmark_comparison.py")

def show_system_info():
    """Show system information."""
    try:
        # Import device optimizer to get system info
        from modules.device_optimizer import DeviceOptimizer
        
        print("🖥️  System Information:")
        print("=" * 60)
        
        optimizer = DeviceOptimizer()
        optimizer.analyze_system()
        
        print("\n✅ System analysis complete!")
        
    except Exception as e:
        print(f"❌ Error getting system info: {e}")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_key():
    """Get a single keypress on Windows."""
    if os.name == 'nt':
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x00' or key == b'\xe0':  # Special keys (arrows, etc.)
                    key = msvcrt.getch()
                    return ord(key)
                else:
                    return ord(key)
            time.sleep(0.01)
    else:
        # For Unix/Linux systems, you'd need termios/tty
        return ord(input()[0]) if input() else 0

def filter_options(options, query):
    """Filter menu options based on search query."""
    if not query:
        return options
    return [opt for opt in options if query.lower() in opt['title'].lower() or query.lower() in opt['description'].lower()]

def draw_cli_interface(title, options, selected_idx, search_query="", search_mode=False):
    """Draw the CLI interface with given options."""
    clear_screen()
    
    # Header
    print(f"Kolosal CLI - {title}")
    print("Use UP/DOWN arrows to navigate, ENTER to select, ESC or Ctrl+C to exit")
    print("Press '/' to search, BACKSPACE to clear search")
    print()
    
    # Search box
    if search_mode:
        print(f"Search: {search_query}_")
    else:
        if search_query:
            print(f"Search: {search_query}")
        else:
            print("Search: (Press '/' to search)")
    print()
    
    # Menu items
    filtered_options = filter_options(options, search_query)
    
    for i, option in enumerate(filtered_options):
        if i == selected_idx:
            print(f"> {option['title']}")
            print(f"  {option['description']}")
        else:
            print(f"  {option['title']}")
            print(f"  {option['description']}")
        print()
    
    # Show "... X more below" if there are more items
    if len(filtered_options) > 10:
        remaining = len(filtered_options) - 10
        if selected_idx < 10:
            print(f"... {remaining} more below")
    
    # Footer
    total_options = len(filtered_options)
    if total_options > 0:
        current_selection = filtered_options[selected_idx] if selected_idx < len(filtered_options) else None
        if current_selection:
            print(f"Selected: {current_selection['title']} ({selected_idx + 1}/{total_options})")
    
    return filtered_options

def cli_menu_handler(title, options, on_select=None):
    """Generic CLI menu handler with search and navigation."""
    selected_index = 0
    search_query = ""
    search_mode = False
    
    while True:
        try:
            filtered_options = draw_cli_interface(title, options, selected_index, search_query, search_mode)
            
            if not filtered_options:
                print("No matches found. Press any key to continue...")
                get_key()
                search_query = ""
                search_mode = False
                continue
            
            # Ensure selected_index is within bounds
            if selected_index >= len(filtered_options):
                selected_index = 0
            
            key = get_key()
            
            if search_mode:
                if key == 27:  # ESC - exit search mode
                    search_mode = False
                elif key == 8:  # Backspace
                    search_query = search_query[:-1]
                elif key == 13:  # Enter - exit search mode
                    search_mode = False
                elif 32 <= key <= 126:  # Printable characters
                    search_query += chr(key)
            else:
                if key == 27:  # ESC - return to previous menu or exit
                    return None
                elif key == 72:  # Up arrow
                    selected_index = (selected_index - 1) % len(filtered_options)
                elif key == 80:  # Down arrow
                    selected_index = (selected_index + 1) % len(filtered_options)
                elif key == 13:  # Enter
                    selected_option = filtered_options[selected_index]
                    if on_select:
                        result = on_select(selected_option)
                        if result == 'exit':
                            return 'exit'
                        elif result == 'continue':
                            continue
                    return selected_option
                elif key == 47:  # '/' - enter search mode
                    search_mode = True
                    search_query = ""
                elif key == 8:  # Backspace - clear search
                    search_query = ""
                    selected_index = 0
                elif key == 3:  # Ctrl+C
                    return 'exit'
        
        except KeyboardInterrupt:
            return 'exit'
        except EOFError:
            return 'exit'

def gui_options_menu():
    """Show GUI options submenu."""
    gui_options = [
        {
            'id': 'full',
            'title': '🎯 Full Mode (Training + Inference)',
            'description': 'Complete ML workflow with training and inference capabilities'
        },
        {
            'id': 'inference',
            'title': '🔮 Inference Only Mode',
            'description': 'Use pre-trained models for predictions only'
        },
        {
            'id': 'back',
            'title': '⬅️ Back to Main Menu',
            'description': 'Return to the main menu'
        }
    ]
    
    def handle_gui_selection(option):
        if option['id'] == 'full':
            clear_screen()
            print("🌐 Starting Full Mode Web Interface...")
            run_gradio_app()
            return 'continue'
        elif option['id'] == 'inference':
            clear_screen()
            print("🔮 Starting Inference Only Mode...")
            run_gradio_app(["--inference-only"])
            return 'continue'
        elif option['id'] == 'back':
            return 'exit'
        return 'continue'
    
    return cli_menu_handler("Web Interface Options", gui_options, handle_gui_selection)

def api_options_menu():
    """Show API options submenu."""
    api_options = [
        {
            'id': 'default',
            'title': '🚀 Start with Default Settings',
            'description': 'Launch API server on localhost:8000 with default configuration'
        },
        {
            'id': 'custom',
            'title': '⚙️ Custom Configuration',
            'description': 'Configure host, port, and other server settings'
        },
        {
            'id': 'back',
            'title': '⬅️ Back to Main Menu',
            'description': 'Return to the main menu'
        }
    ]
    
    def handle_api_selection(option):
        if option['id'] == 'default':
            clear_screen()
            print("🔧 Starting API Server with default settings...")
            run_api_server()
            return 'continue'
        elif option['id'] == 'custom':
            clear_screen()
            print("⚙️ Custom API Configuration:")
            print("(Feature coming soon - using default settings for now)")
            print()
            input("Press Enter to start with default settings...")
            run_api_server()
            return 'continue'
        elif option['id'] == 'back':
            return 'exit'
        return 'continue'
    
    return cli_menu_handler("API Server Options", api_options, handle_api_selection)

def benchmark_options_menu():
    """Show benchmark options submenu."""
    benchmark_options = [
        {
            'id': 'standard_default',
            'title': '📊 Run Standard ML Benchmark (Default)',
            'description': 'Run baseline benchmarks with default datasets and models'
        },
        {
            'id': 'standard_custom',
            'title': '⚙️ Custom Standard ML Benchmark',
            'description': 'Configure datasets, models, and optimization strategies'
        },
        {
            'id': 'kolosal_default',
            'title': '🚀 Run Kolosal AutoML Benchmark (Default)',
            'description': 'Run advanced AutoML benchmarks with default configuration'
        },
        {
            'id': 'kolosal_custom',
            'title': '🔧 Custom Kolosal AutoML Benchmark',
            'description': 'Configure advanced AutoML optimization strategies'
        },
        {
            'id': 'comparison_quick',
            'title': '⚖️ Quick Comparison (Standard vs AutoML)',
            'description': 'Fast head-to-head comparison on small datasets'
        },
        {
            'id': 'comparison_comprehensive',
            'title': '� Comprehensive Comparison Analysis',
            'description': 'Full comparison across all datasets and models'
        },
        {
            'id': 'benchmark_comparison_tool',
            'title': '📋 Benchmark Comparison Tool',
            'description': 'Advanced comparison analysis with statistical testing'
        },
        {
            'id': 'quick_demo',
            'title': '⚡ Quick Demo Benchmark',
            'description': 'Fast benchmark with small datasets for testing'
        },
        {
            'id': 'scalability_test',
            'title': '📏 Scalability Testing',
            'description': 'Test performance across different dataset sizes'
        },
        {
            'id': 'back',
            'title': '⬅️ Back to Main Menu',
            'description': 'Return to the main menu'
        }
    ]
    
    def handle_benchmark_selection(option):
        if option['id'] == 'standard_default':
            clear_screen()
            print("📊 Starting Standard ML Benchmark with default settings...")
            run_standard_ml_benchmark()
            return 'continue'
        elif option['id'] == 'standard_custom':
            clear_screen()
            print("⚙️ Custom Standard ML Benchmark Configuration:")
            print("Available options:")
            print("  --datasets: iris wine breast_cancer diabetes synthetic_small_classification")
            print("  --models: random_forest gradient_boosting logistic_regression")
            print("  --optimization: grid_search random_search")
            print()
            
            # Get user input for custom configuration
            datasets = input("Enter datasets (space-separated, or press Enter for default): ").strip()
            models = input("Enter models (space-separated, or press Enter for default): ").strip()
            optimization = input("Enter optimization strategy (grid_search/random_search, or press Enter for random_search): ").strip()
            
            custom_args = []
            if datasets:
                custom_args.extend(["--datasets"] + datasets.split())
            if models:
                custom_args.extend(["--models"] + models.split())
            if optimization:
                custom_args.extend(["--optimization", optimization])
            
            run_standard_ml_benchmark(custom_args)
            return 'continue'
        elif option['id'] == 'kolosal_default':
            clear_screen()
            print("🚀 Starting Kolosal AutoML Benchmark with default settings...")
            run_kolosal_automl_benchmark()
            return 'continue'
        elif option['id'] == 'kolosal_custom':
            clear_screen()
            print("🔧 Custom Kolosal AutoML Benchmark Configuration:")
            print("Available options:")
            print("  --datasets: iris wine breast_cancer diabetes synthetic_*")
            print("  --models: random_forest gradient_boosting logistic_regression")
            print("  --optimization: random_search bayesian_optimization asht")
            print("  --device-config: cpu_only gpu_basic gpu_optimized")
            print()
            
            # Get user input for custom configuration
            datasets = input("Enter datasets (space-separated, or press Enter for default): ").strip()
            models = input("Enter models (space-separated, or press Enter for default): ").strip()
            optimization = input("Enter optimization strategy (random_search/bayesian_optimization/asht, or press Enter for random_search): ").strip()
            device_config = input("Enter device config (cpu_only/gpu_basic/gpu_optimized, or press Enter for cpu_only): ").strip()
            
            custom_args = []
            if datasets:
                custom_args.extend(["--datasets"] + datasets.split())
            if models:
                custom_args.extend(["--models"] + models.split())
            if optimization:
                custom_args.extend(["--optimization", optimization])
            if device_config:
                custom_args.extend(["--device-config", device_config])
            
            run_kolosal_automl_benchmark(custom_args)
            return 'continue'
        elif option['id'] == 'comparison_quick':
            clear_screen()
            print("⚖️ Starting Quick Comparison (Standard vs AutoML)...")
            quick_comparison_args = ["--mode", "quick"]
            run_comparison_benchmark(quick_comparison_args)
            return 'continue'
        elif option['id'] == 'comparison_comprehensive':
            clear_screen()
            print("� Starting Comprehensive Comparison Analysis...")
            comprehensive_args = ["--mode", "comprehensive"]
            run_comparison_benchmark(comprehensive_args)
            return 'continue'
        elif option['id'] == 'benchmark_comparison_tool':
            clear_screen()
            print("📋 Starting Benchmark Comparison Tool...")
            run_benchmark_comparison_tool()
            return 'continue'
        elif option['id'] == 'quick_demo':
            clear_screen()
            print("⚡ Starting Quick Demo Benchmark...")
            quick_args = [
                "--datasets", "iris", "wine",
                "--models", "random_forest", "logistic_regression",
                "--optimization", "random_search"
            ]
            run_standard_ml_benchmark(quick_args)
            return 'continue'
        elif option['id'] == 'scalability_test':
            clear_screen()
            print("📏 Starting Scalability Testing...")
            scalability_args = ["--mode", "scalability"]
            run_comparison_benchmark(scalability_args)
            return 'continue'
        elif option['id'] == 'back':
            return 'exit'
        return 'continue'
    
    return cli_menu_handler("Benchmark & Comparison Options", benchmark_options, handle_benchmark_selection)

def comparison_options_menu():
    """Show comparison options submenu."""
    comparison_options = [
        {
            'id': 'quick_comparison',
            'title': '⚡ Quick Comparison (Standard vs AutoML)',
            'description': 'Fast head-to-head comparison on iris and wine datasets'
        },
        {
            'id': 'comprehensive_comparison',
            'title': '📈 Comprehensive Comparison',
            'description': 'Complete comparison across multiple datasets and models'
        },
        {
            'id': 'optimization_strategies',
            'title': '🔧 Optimization Strategies Comparison',
            'description': 'Compare different hyperparameter optimization approaches'
        },
        {
            'id': 'scalability_comparison',
            'title': '📏 Scalability Comparison',
            'description': 'Test and compare performance across different dataset sizes'
        },
        {
            'id': 'large_scale_comparison',
            'title': '🏢 Large Scale Comparison',
            'description': 'Performance testing on large datasets (may take longer)'
        },
        {
            'id': 'custom_comparison',
            'title': '⚙️ Custom Comparison',
            'description': 'Configure your own comparison parameters'
        },
        {
            'id': 'comparison_tool',
            'title': '📋 Advanced Comparison Tool',
            'description': 'Statistical analysis and visualization tool'
        },
        {
            'id': 'back',
            'title': '⬅️ Back to Main Menu',
            'description': 'Return to the main menu'
        }
    ]
    
    def handle_comparison_selection(option):
        if option['id'] == 'quick_comparison':
            clear_screen()
            print("⚡ Starting Quick Comparison...")
            run_comparison_benchmark(["--mode", "quick"])
            return 'continue'
        elif option['id'] == 'comprehensive_comparison':
            clear_screen()
            print("📈 Starting Comprehensive Comparison...")
            run_comparison_benchmark(["--mode", "comprehensive"])
            return 'continue'
        elif option['id'] == 'optimization_strategies':
            clear_screen()
            print("🔧 Starting Optimization Strategies Comparison...")
            run_comparison_benchmark(["--mode", "optimization_strategies"])
            return 'continue'
        elif option['id'] == 'scalability_comparison':
            clear_screen()
            print("📏 Starting Scalability Comparison...")
            run_comparison_benchmark(["--mode", "scalability"])
            return 'continue'
        elif option['id'] == 'large_scale_comparison':
            clear_screen()
            print("🏢 Starting Large Scale Comparison...")
            print("⚠️ Warning: This may take a significant amount of time and resources.")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_comparison_benchmark(["--mode", "large_scale"])
            else:
                print("Large scale comparison cancelled.")
            return 'continue'
        elif option['id'] == 'custom_comparison':
            clear_screen()
            print("⚙️ Custom Comparison Configuration:")
            print("Available datasets: iris wine breast_cancer diabetes synthetic_*")
            print("Available models: random_forest gradient_boosting logistic_regression ridge lasso")
            print("Available optimizations: random_search grid_search bayesian_optimization asht")
            print()
            
            # Get user input
            datasets = input("Enter datasets (space-separated, or press Enter for quick defaults): ").strip()
            models = input("Enter models (space-separated, or press Enter for quick defaults): ").strip()
            optimization = input("Enter optimization strategy (or press Enter for random_search): ").strip()
            
            custom_args = ["--mode", "custom"]
            if datasets:
                custom_args.extend(["--datasets"] + datasets.split())
            if models:
                custom_args.extend(["--models"] + models.split())
            if optimization:
                custom_args.extend(["--optimization", optimization])
            
            run_comparison_benchmark(custom_args)
            return 'continue'
        elif option['id'] == 'comparison_tool':
            clear_screen()
            print("📋 Starting Advanced Comparison Tool...")
            run_benchmark_comparison_tool()
            return 'continue'
        elif option['id'] == 'back':
            return 'exit'
        return 'continue'
    
    return cli_menu_handler("ML Comparison Tools", comparison_options, handle_comparison_selection)

def system_info_menu():
    """Show system information submenu."""
    system_options = [
        {
            'id': 'full_analysis',
            'title': '🖥️ Full System Analysis',
            'description': 'Complete hardware and software analysis'
        },
        {
            'id': 'hardware_only',
            'title': '⚡ Hardware Information Only',
            'description': 'Display CPU, GPU, and memory information'
        },
        {
            'id': 'dependencies',
            'title': '📦 Dependency Check',
            'description': 'Check all required dependencies and versions'
        },
        {
            'id': 'back',
            'title': '⬅️ Back to Main Menu',
            'description': 'Return to the main menu'
        }
    ]
    
    def handle_system_selection(option):
        clear_screen()
        if option['id'] == 'full_analysis':
            print("🖥️ Running Full System Analysis...")
            show_system_info()
        elif option['id'] == 'hardware_only':
            print("⚡ Hardware Information:")
            try:
                from modules.device_optimizer import DeviceOptimizer
                optimizer = DeviceOptimizer()
                optimizer.analyze_hardware()
            except Exception as e:
                print(f"❌ Error getting hardware info: {e}")
        elif option['id'] == 'dependencies':
            print("📦 Checking Dependencies...")
            check_dependencies()
            print("\n✅ Dependency check complete!")
        elif option['id'] == 'back':
            return 'exit'
        
        if option['id'] != 'back':
            print("\nPress any key to continue...")
            get_key()
        return 'continue'
    
    return cli_menu_handler("System Information", system_options, handle_system_selection)

def interactive_mode():
    """Run in interactive mode with CLI interface."""
    
    # Main menu options
    main_menu_options = [
        {
            'id': 'gui',
            'title': '🌐 Launch Web Interface (Gradio UI)',
            'description': 'Interactive web-based interface for ML training and inference'
        },
        {
            'id': 'api', 
            'title': '🔧 Start API Server (REST API)',
            'description': 'RESTful API endpoints for programmatic access'
        },
        {
            'id': 'benchmark',
            'title': '📊 Benchmarks & Comparisons',
            'description': 'Run ML benchmarks and comparison analysis (Standard ML vs AutoML)'
        },
        {
            'id': 'comparison',
            'title': '⚖️ ML Comparison Tools',
            'description': 'Direct comparison between different ML approaches and frameworks'
        },
        {
            'id': 'system_info',
            'title': '📋 Show System Information', 
            'description': 'View detailed system information and hardware capabilities'
        },
        {
            'id': 'help',
            'title': '❓ Show Help',
            'description': 'Display detailed help and usage information'
        },
        {
            'id': 'exit',
            'title': '🚪 Exit',
            'description': 'Exit the kolosal AutoML system'
        }
    ]
    
    def handle_main_selection(option):
        if option['id'] == 'gui':
            result = gui_options_menu()
            return 'continue'
        elif option['id'] == 'api':
            result = api_options_menu()
            return 'continue'
        elif option['id'] == 'benchmark':
            result = benchmark_options_menu()
            return 'continue'
        elif option['id'] == 'comparison':
            result = comparison_options_menu()
            return 'continue'
        elif option['id'] == 'system_info':
            result = system_info_menu()
            return 'continue'
        elif option['id'] == 'help':
            clear_screen()
            print_help()
            print("\nPress any key to continue...")
            get_key()
            return 'continue'
        elif option['id'] == 'exit':
            clear_screen()
            print("👋 Thank you for using kolosal AutoML!")
            return 'exit'
        return 'continue'
    
    # Main menu loop
    while True:
        result = cli_menu_handler("Select Mode", main_menu_options, handle_main_selection)
        if result == 'exit' or result is None:
            break

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="kolosal AutoML - Machine Learning Training & Inference System v0.1.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                # Interactive mode (choose what to run)
  python main.py --mode gui                     # Launch Gradio web interface
  python main.py --mode api                     # Start API server
  python main.py --mode benchmark               # Run benchmark options menu
  python main.py --mode comparison              # Run comparison options menu
  python main.py --mode gui --help              # Show Gradio-specific help
  python main.py --mode api --help              # Show API-specific help
  python main.py --system-info                  # Show system information
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["gui", "api", "interactive", "benchmark", "comparison"],
        default="interactive",
        help="Mode to run: 'gui' for Gradio interface, 'api' for REST API server, 'benchmark' for ML benchmarks, 'comparison' for ML comparisons, 'interactive' to choose"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="kolosal AutoML v0.1.4"
    )
    
    parser.add_argument(
        "--system-info",
        action="store_true",
        help="Show system information and exit"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the banner display"
    )
    
    # Parse known args to allow passing through to subcommands
    args, unknown_args = parser.parse_known_args()
    
    # Print banner unless disabled
    if not args.no_banner:
        print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Handle system info request
    if args.system_info:
        show_system_info()
        return 0
    
    # Route to appropriate mode
    if args.mode == "gui":
        run_gradio_app(unknown_args)
    elif args.mode == "api":
        run_api_server(unknown_args)
    elif args.mode == "benchmark":
        result = benchmark_options_menu()
    elif args.mode == "comparison":
        result = comparison_options_menu()
    elif args.mode == "interactive":
        interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
