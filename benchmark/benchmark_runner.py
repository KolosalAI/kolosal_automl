#!/usr/bin/env python3
"""
Genta AutoML Benchmark Runner

This script runs comprehensive benchmarks for the Genta AutoML training engine
using a configuration file that specifies datasets, models, and optimization strategies.

Features:
- Configurable benchmark execution with JSON configuration
- Subset execution for targeted benchmarking
- Comprehensive error handling and reporting
- Performance analytics and detailed reporting
- Memory usage tracking and optimization
- Graceful shutdown and cleanup
"""

import argparse
import json
import os
import sys
import time
import logging
import gc
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from training_engine_benchmark import BenchmarkRunner

# Set up enhanced logging with both file and console output
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup enhanced logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"benchmark_run_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_filepath = logs_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("BenchmarkRunner")
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    return logger

logger = setup_logging()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load benchmark configuration from a JSON file with enhanced validation.
    
    Args:
        config_path: Path to the benchmark configuration JSON file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        SystemExit: If configuration loading fails
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Validate required configuration fields
        required_fields = ['benchmarks']
        for field in required_fields:
            if field not in config:
                logger.error(f"Required field '{field}' missing from configuration")
                sys.exit(1)
                
        # Set default values for optional fields
        config.setdefault('output_dir', './benchmark_results')
        config.setdefault('enable_experiment_tracking', True)
        config.setdefault('random_seed', 42)
        
        logger.info(f"Configuration loaded successfully from: {config_path}")
        logger.info(f"Found {len(config.get('benchmarks', []))} benchmark groups")
        
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

def monitor_system_resources() -> Dict[str, Any]:
    """Monitor and log current system resources."""
    try:
        # Get system information
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        resources = {
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
            "cpu_percent": cpu_percent,
            "cpu_count": psutil.cpu_count(),
            "disk_total_gb": round(disk_usage.total / (1024**3), 2),
            "disk_free_gb": round(disk_usage.free / (1024**3), 2),
            "disk_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
        }
        
        logger.info(f"System Resources - Memory: {resources['memory_percent']:.1f}% "
                   f"({resources['memory_available_gb']:.1f}GB free), "
                   f"CPU: {resources['cpu_percent']:.1f}%, "
                   f"Disk: {resources['disk_percent']:.1f}%")
        
        return resources
        
    except Exception as e:
        logger.warning(f"Failed to collect system resources: {e}")
        return {"error": str(e)}


def cleanup_resources():
    """Perform cleanup operations to free up resources."""
    try:
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Additional cleanup if needed
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
            
    except ImportError:
        # PyTorch not available, skip CUDA cleanup
        pass
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def validate_benchmarks(benchmarks: List[Dict[str, Any]]) -> bool:
    """
    Validate benchmark configurations.
    
    Args:
        benchmarks: List of benchmark configurations
        
    Returns:
        True if all configurations are valid, False otherwise
    """
    required_config_fields = ['dataset', 'model', 'optimization_strategy']
    
    for i, benchmark in enumerate(benchmarks):
        configurations = benchmark.get('configurations', [])
        for j, config in enumerate(configurations):
            # Check required fields
            missing_fields = [field for field in required_config_fields if field not in config]
            if missing_fields:
                logger.error(f"Benchmark {i}, config {j} missing required fields: {missing_fields}")
                return False
                
            # Validate optimization strategy
            valid_strategies = ['grid_search', 'random_search', 'bayesian_optimization', 'asht']
            if config['optimization_strategy'] not in valid_strategies:
                logger.error(f"Invalid optimization strategy: {config['optimization_strategy']}. "
                           f"Valid options: {valid_strategies}")
                return False
    
    return True

def run_benchmarks(config_path: str, subset: Optional[List[str]] = None, 
                  output_dir: Optional[str] = None, verbose: bool = True,
                  resource_monitoring: bool = True) -> Dict[str, Any]:
    """
    Run benchmarks based on the configuration file with enhanced features.
    
    Args:
        config_path: Path to the benchmark configuration JSON
        subset: Optional list of benchmark names to run (subset of those in config)
        output_dir: Optional custom output directory
        verbose: Enable verbose logging
        resource_monitoring: Enable system resource monitoring
    
    Returns:
        Dictionary containing benchmark results and metadata
    """
    start_time = time.time()
    
    # Monitor initial system resources
    if resource_monitoring:
        initial_resources = monitor_system_resources()
    else:
        initial_resources = {}
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Use specified output directory or from config
        if output_dir:
            result_dir = output_dir
        else:
            result_dir = config.get("output_dir", "./benchmark_results")
        
        # Ensure the output directory exists
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create benchmark runner with enhanced configuration
        runner_output_dir = result_dir / f"run_{timestamp}"
        runner = BenchmarkRunner(output_dir=str(runner_output_dir))
        
        # Filter benchmarks if subset is specified
        if subset:
            # Convert to set for faster lookup
            subset_set = set(subset)
            benchmarks = [b for b in config.get("benchmarks", []) if b.get("name") in subset_set]
            if not benchmarks:
                logger.error(f"No benchmark found with names: {subset}")
                missing_names = subset_set - {b.get("name") for b in config.get("benchmarks", [])}
                available_names = [b.get("name") for b in config.get("benchmarks", [])]
                logger.error(f"Missing benchmark names: {missing_names}")
                logger.info(f"Available benchmark names: {available_names}")
                return {
                    "success": False,
                    "error": f"No benchmarks found with names: {subset}",
                    "available_benchmarks": available_names
                }
        else:
            benchmarks = config.get("benchmarks", [])
        
        # Validate benchmark configurations
        if not validate_benchmarks(benchmarks):
            logger.error("Benchmark validation failed")
            return {
                "success": False,
                "error": "Invalid benchmark configurations"
            }
        
        # Print the benchmarks that will be run
        logger.info(f"Running {len(benchmarks)} benchmark groups:")
        total_configs = 0
        for b in benchmarks:
            config_count = len(b.get('configurations', []))
            total_configs += config_count
            logger.info(f" - {b.get('name')}: {b.get('description', 'No description')} "
                       f"({config_count} configurations)")
        
        # Collect all configurations from selected benchmarks
        all_configurations = []
        for benchmark in benchmarks:
            all_configurations.extend(benchmark.get("configurations", []))
        
        # Skip if no configurations found
        if not all_configurations:
            logger.error("No benchmark configurations found in the configuration file.")
            return {
                "success": False,
                "error": "No benchmark configurations found"
            }
        
        # Log system information before starting
        logger.info(f"Starting benchmarks with {total_configs} total configurations")
        logger.info(f"Output directory: {runner_output_dir}")
        if resource_monitoring:
            logger.info(f"System resources at start: Memory {initial_resources.get('memory_percent', 0):.1f}%, "
                       f"CPU {initial_resources.get('cpu_percent', 0):.1f}%")
        
        # Run all benchmarks with progress tracking
        logger.info(f"Starting to run {len(all_configurations)} benchmark configurations...")
        
        try:
            results = runner.run_multiple_benchmarks(all_configurations)
            
            # Save results
            results_file = runner.save_results()
            
            # Generate report
            report_file = runner.generate_report()
            
        except KeyboardInterrupt:
            logger.warning("Benchmark execution interrupted by user")
            return {
                "success": False,
                "error": "Interrupted by user",
                "partial_results": getattr(runner, 'results', [])
            }
        except Exception as e:
            logger.error(f"Error during benchmark execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": getattr(runner, 'results', [])
            }
        
        # Calculate total run time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Monitor final system resources
        if resource_monitoring:
            final_resources = monitor_system_resources()
        else:
            final_resources = {}
        
        # Performance cleanup
        cleanup_resources()
        
        # Log completion summary
        logger.info(f"All benchmarks completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report generated: {report_file}")
        
        # Count successful vs failed benchmarks
        successful_count = sum(1 for r in results if r.get('success', False))
        failed_count = len(results) - successful_count
        
        if failed_count > 0:
            logger.warning(f"Benchmark summary: {successful_count} successful, {failed_count} failed")
        else:
            logger.info(f"All {successful_count} benchmarks completed successfully")
        
        return {
            "success": True,
            "results_file": results_file,
            "report_file": report_file,
            "total_benchmarks": len(all_configurations),
            "successful_benchmarks": successful_count,
            "failed_benchmarks": failed_count,
            "total_time_seconds": total_time,
            "initial_resources": initial_resources,
            "final_resources": final_resources,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Critical error in benchmark execution: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_time_seconds": time.time() - start_time
        }
    
    finally:
        # Final cleanup
        cleanup_resources()

def show_available_benchmarks(config_path: str) -> None:
    """
    Display available benchmark configurations from the config file.
    
    Args:
        config_path: Path to the benchmark configuration file
    """
    try:
        config = load_config(config_path)
        benchmarks = config.get("benchmarks", [])
        
        print("\n" + "=" * 60)
        print("AVAILABLE BENCHMARK CONFIGURATIONS")
        print("=" * 60)
        
        total_configs = 0
        for i, benchmark in enumerate(benchmarks, 1):
            name = benchmark.get('name', f'Unnamed_{i}')
            description = benchmark.get('description', 'No description available')
            configurations = benchmark.get('configurations', [])
            config_count = len(configurations)
            total_configs += config_count
            
            print(f"\n{i}. {name}")
            print(f"   Description: {description}")
            print(f"   Configurations: {config_count}")
            
            # Show first few configurations as examples
            if configurations:
                print("   Sample configurations:")
                for j, config in enumerate(configurations[:3]):  # Show first 3
                    dataset = config.get('dataset', 'N/A')
                    model = config.get('model', 'N/A')
                    strategy = config.get('optimization_strategy', 'N/A')
                    print(f"     - {dataset} + {model} + {strategy}")
                
                if len(configurations) > 3:
                    print(f"     ... and {len(configurations) - 3} more")
        
        print(f"\nTotal benchmark groups: {len(benchmarks)}")
        print(f"Total configurations: {total_configs}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error reading configuration: {e}")


def validate_config_file(config_path: str) -> bool:
    """
    Validate the configuration file format and content.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        config = load_config(config_path)
        benchmarks = config.get("benchmarks", [])
        
        if not benchmarks:
            logger.error("Configuration file contains no benchmarks")
            return False
            
        if not validate_benchmarks(benchmarks):
            return False
            
        logger.info("Configuration file validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def main():
    """Enhanced main entry point with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run Genta AutoML benchmarks with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.json                          # Run all benchmarks
  %(prog)s config.json --subset small_datasets  # Run specific benchmark group
  %(prog)s config.json --output-dir ./results   # Custom output directory
  %(prog)s config.json --verbose --no-monitoring # Verbose mode without resource monitoring
  %(prog)s config.json --list                   # Show available benchmarks
  %(prog)s config.json --validate               # Validate configuration only
        """
    )
    
    parser.add_argument(
        "config", 
        help="Path to benchmark configuration file (JSON format)"
    )
    parser.add_argument(
        "--subset", 
        nargs="+", 
        help="Run only specific benchmark groups by name"
    )
    parser.add_argument(
        "--output-dir", 
        help="Custom output directory for results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging output"
    )
    parser.add_argument(
        "--no-monitoring", 
        action="store_true", 
        help="Disable system resource monitoring"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available benchmark configurations and exit"
    )
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate configuration file and exit"
    )
    
    args = parser.parse_args()
    
    # Update logger level if specified
    if args.log_level != "INFO":
        logger.setLevel(getattr(logging, args.log_level))
        
    # Validate configuration file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Handle list option
    if args.list:
        show_available_benchmarks(args.config)
        sys.exit(0)
    
    # Handle validate option
    if args.validate:
        if validate_config_file(args.config):
            print("✓ Configuration file is valid")
            sys.exit(0)
        else:
            print("✗ Configuration file validation failed")
            sys.exit(1)
    
    # Log execution parameters
    logger.info("=" * 60)
    logger.info("Genta AutoML Benchmark Runner Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration file: {args.config}")
    if args.subset:
        logger.info(f"Running subset: {args.subset}")
    if args.output_dir:
        logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info(f"Resource monitoring: {not args.no_monitoring}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 60)
    
    try:
        # Run benchmarks with enhanced parameters
        result = run_benchmarks(
            config_path=args.config,
            subset=args.subset,
            output_dir=args.output_dir,
            verbose=args.verbose,
            resource_monitoring=not args.no_monitoring
        )
        
        # Print final summary
        if result.get("success", False):
            logger.info("=" * 60)
            logger.info("BENCHMARK EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total benchmarks: {result.get('total_benchmarks', 0)}")
            logger.info(f"Successful: {result.get('successful_benchmarks', 0)}")
            logger.info(f"Failed: {result.get('failed_benchmarks', 0)}")
            logger.info(f"Total time: {result.get('total_time_seconds', 0):.2f} seconds")
            logger.info(f"Results: {result.get('results_file', 'N/A')}")
            logger.info(f"Report: {result.get('report_file', 'N/A')}")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.error("=" * 60)
            logger.error("BENCHMARK EXECUTION FAILED")
            logger.error("=" * 60)
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            if result.get('partial_results'):
                logger.info(f"Partial results available: {len(result['partial_results'])} benchmarks completed")
            logger.error("=" * 60)
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\nBenchmark execution interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()