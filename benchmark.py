#!/usr/bin/env python3
"""
Genta AutoML Benchmark Runner

This script runs comprehensive benchmarks for the Genta AutoML training engine
using a configuration file that specifies datasets, models, and optimization strategies.
"""

import argparse
import json
import os
import sys
import time
import logging
import gc
from datetime import datetime

# Try to import the benchmark runner
try:
    from benchmark.benchmark_train_engine import BenchmarkRunner
except ImportError:
    print("ERROR: Cannot import BenchmarkRunner. Make sure training_engine_benchmark.py is in the current directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BenchmarkRunner")

def load_config(config_path):
    """Load benchmark configuration from a JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

def run_benchmarks(config_path, subset=None, output_dir=None):
    """
    Run benchmarks based on the configuration file
    
    Args:
        config_path: Path to the benchmark configuration JSON
        subset: Optional list of benchmark names to run (subset of those in config)
        output_dir: Optional custom output directory
    """
    # Load configuration
    config = load_config(config_path)
    
    # Use specified output directory or from config
    if output_dir:
        result_dir = output_dir
    else:
        result_dir = config.get("output_dir", "./benchmark_results")
    
    # Ensure the output directory exists
    os.makedirs(result_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=os.path.join(result_dir, f"run_{timestamp}"))
    
    # Collect all benchmark configurations
    all_configurations = []
    
    # Filter benchmarks if subset is specified
    if subset:
        # Convert to set for faster lookup
        subset_set = set(subset)
        benchmarks = [b for b in config.get("benchmarks", []) if b.get("name") in subset_set]
        if not benchmarks:
            logger.error(f"No benchmark found with names: {subset}")
            return
    else:
        benchmarks = config.get("benchmarks", [])
    
    # Print the benchmarks that will be run
    logger.info(f"Running {len(benchmarks)} benchmark groups:")
    for b in benchmarks:
        logger.info(f" - {b.get('name')}: {b.get('description')} ({len(b.get('configurations', []))} configurations)")
    
    # Collect all configurations from selected benchmarks
    for benchmark in benchmarks:
        all_configurations.extend(benchmark.get("configurations", []))
    
    # Skip if no configurations found
    if not all_configurations:
        logger.error("No benchmark configurations found in the configuration file.")
        return
    
    # Run all benchmarks
    logger.info(f"Starting to run {len(all_configurations)} benchmark configurations...")
    start_time = time.time()
    
    results = runner.run_multiple_benchmarks(all_configurations)
    
    # Save results
    results_file = runner.save_results()
    
    # Generate report
    report_file = runner.generate_report()
    
    # Calculate total run time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"All benchmarks completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Report generated: {report_file}")
    
    return {
        "results_file": results_file,
        "report_file": report_file,
        "total_benchmarks": len(all_configurations),
        "total_time_seconds": total_time
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Genta AutoML benchmarks")
    parser.add_argument("config", help="Path to benchmark configuration file (JSON)")
    parser.add_argument("--subset", nargs="+", help="Run only specific benchmark groups by name")
    parser.add_argument("--output-dir", help="Custom output directory")
    args = parser.parse_args()
    
    # Run benchmarks
    run_benchmarks(args.config, args.subset, args.output_dir)

if __name__ == "__main__":
    main()