#!/usr/bin/env python3
"""
kolosal AutoML Training & Inference System Version v0.1.4
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
    elif args.mode == "interactive":
        interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
