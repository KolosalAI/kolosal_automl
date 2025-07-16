# kolosal AutoML CLI Usage

The main CLI entry point for kolosal AutoML system that allows you to choose between different modes of operation.

## Quick Start

```bash
# Interactive mode (recommended for first-time users)
python main.py

# Launch Gradio web interface directly
python main.py --mode gui

# Start API server directly  
python main.py --mode api

# Show version
python main.py --version

# Show system information
python main.py --system-info

# Show help
python main.py --help
```

## Available Modes

### 1. 🌐 Web Interface (Gradio UI)
- Interactive web-based interface at http://localhost:7860
- Visual data upload and configuration
- Model training with real-time progress
- Perfect for experimentation and learning

### 2. 🔧 API Server (REST API)
- RESTful API endpoints at http://localhost:8000  
- Interactive documentation at http://localhost:8000/docs
- Programmatic access to all features
- Perfect for production deployments

### 3. 📋 System Information
- View detailed hardware and software information
- Check system capabilities and optimization recommendations
- Verify dependencies

## Command Line Options

```
--mode {gui,api,interactive}    Mode to run (default: interactive)
--version                       Show version and exit
--system-info                   Show system information and exit  
--no-banner                     Skip the banner display
--help                          Show help message and exit
```

## Examples

```bash
# Interactive mode - choose what to run
python main.py

# Launch web interface in inference-only mode
python main.py --mode gui --inference-only

# Start API server with custom host/port
python main.py --mode api --host 0.0.0.0 --port 8080

# Quick system check
python main.py --system-info --no-banner
```

## Files

- `main.py` - Main CLI entry point
- `app.py` - Gradio web interface application  
- `start_api.py` - API server startup script
- `modules/api/app.py` - Core API application

## Version

Current version: **v0.1.4**
