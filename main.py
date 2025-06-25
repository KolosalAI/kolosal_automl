#!/usr/bin/env python3
"""
kolosal AutoML Training & Inference System Version v0.1.2
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
║                                 Version v0.1.3                               ║
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

3. 📋 SYSTEM INFO
   - View detailed system information
   - Check hardware capabilities
   - Verify dependencies
   - Optimize system settings

💡 TIPS:
   - Use Web Interface for interactive machine learning
   - Use API Server for production deployments
   - Both modes support all ML algorithms and features
   - Models trained in one mode can be used in the other

🔗 QUICK START:
   - Web Interface: http://localhost:7860 (default)
   - API Server: http://localhost:8000 (default)
   - API Documentation: http://localhost:8000/docs

📚 DOCUMENTATION:
   - Web Interface: Built-in help and examples
   - API: Interactive docs at /docs endpoint
   - Models: Saved in ./models/ directory
   - Logs: Available in respective .log files
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

def interactive_mode():
    """Run in interactive mode - let user choose."""
    print_help()
    
    while True:
        print("\n" + "="*60)
        print("🚀 What would you like to do?")
        print()
        print("1. 🌐 Launch Web Interface (Gradio UI)")
        print("2. 🔧 Start API Server (REST API)")  
        print("3. 📋 Show System Information")
        print("4. ❓ Show Help")
        print("5. 🚪 Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                # Ask for additional options
                print("\n🌐 Web Interface Options:")
                print("1. Full mode (Training + Inference)")
                print("2. Inference only mode")
                
                mode_choice = input("Choose mode (1-2, default=1): ").strip() or "1"
                
                if mode_choice == "2":
                    run_gradio_app(["--inference-only"])
                else:
                    run_gradio_app()
                
            elif choice == "2":
                # Ask for additional options
                print("\n🔧 API Server Options:")
                print("Press Enter for default settings or Ctrl+C to cancel")
                
                try:
                    input("Press Enter to start API server...")
                    run_api_server()
                except KeyboardInterrupt:
                    print("\n⏪ Returning to main menu...")
                    continue
                
            elif choice == "3":
                show_system_info()
                
            elif choice == "4":
                print_help()
                
            elif choice == "5":
                print("\n👋 Thank you for using kolosal AutoML!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="kolosal AutoML - Machine Learning Training & Inference System v0.1.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode (choose what to run)
  python main.py --mode gui         # Launch Gradio web interface
  python main.py --mode api         # Start API server
  python main.py --mode gui --help  # Show Gradio-specific help
  python main.py --mode api --help  # Show API-specific help
  python main.py --system-info      # Show system information
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["gui", "api", "interactive"],
        default="interactive",
        help="Mode to run: 'gui' for Gradio interface, 'api' for REST API server, 'interactive' to choose"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="kolosal AutoML v0.1.3"
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
    elif args.mode == "interactive":
        interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
