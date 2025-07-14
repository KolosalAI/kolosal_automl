#!/usr/bin/env python3
"""
Genta AutoML vs Standard ML Comparison Runner

This script runs predefined comparison configurations to benchmark
Genta AutoML against standard scikit-learn approaches.

Usage:
    python run_comparison.py                           # Run quick comparison
    python run_comparison.py --config comprehensive_small  # Run specific config
    python run_comparison.py --custom --datasets iris wine --models random_forest
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from benchmark_comparison import ComparisonBenchmarkRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_comparison_config(config_file: str = "benchmark_comparison_config.json"):
    """Load comparison configuration."""
    config_path = Path(config_file)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_compatible_configurations(config, datasets, models, device_configs=None, optimizer_configs=None):
    """Get only compatible dataset-model combinations with device and optimizer configurations."""
    configurations = []
    dataset_info = {}
    
    # Build dataset lookup
    for category in config['datasets'].values():
        for dataset in category:
            dataset_info[dataset['name']] = dataset
    
    # Default configurations if not provided
    if device_configs is None:
        device_configs = [dc['name'] for dc in config.get('device_configurations', [{'name': 'cpu_only'}])]
    if optimizer_configs is None:
        optimizer_configs = [oc['name'] for oc in config.get('optimizer_configurations', [{'name': 'random_search_small'}])]
    
    for dataset_name in datasets:
        if dataset_name not in dataset_info:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue
            
        dataset = dataset_info[dataset_name]
        compatible_models = dataset.get('compatible_models', [])
        
        for model_name in models:
            if model_name in compatible_models:
                for device_config in device_configs:
                    for optimizer_config in optimizer_configs:
                        configurations.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'device_config': device_config,
                            'optimizer_config': optimizer_config
                        })
            else:
                logger.info(f"Skipping incompatible combination: {dataset_name} + {model_name}")
    
    return configurations

def run_predefined_comparison(config_name: str, config: dict):
    """Run a predefined comparison configuration."""
    comparison_configs = config.get('comparison_configurations', [])
    
    # Find the specified configuration
    target_config = None
    for comp_config in comparison_configs:
        if comp_config['name'] == config_name:
            target_config = comp_config
            break
    
    if not target_config:
        available_configs = [c['name'] for c in comparison_configs]
        logger.error(f"Configuration '{config_name}' not found. Available: {available_configs}")
        return False
    
    logger.info(f"Running comparison configuration: {target_config['name']}")
    logger.info(f"Description: {target_config['description']}")
    
    # Get number of trials and plotting setting
    num_trials = config.get('comparison_settings', {}).get('num_trials', 1)
    enable_plotting = config.get('comparison_settings', {}).get('enable_trial_plotting', True)
    
    # Get device and optimizer configurations
    device_configs = config.get('device_configurations', [])
    optimizer_configs = config.get('optimizer_configurations', [])
    
    device_names = [dc['name'] for dc in device_configs] if device_configs else ['cpu_only']
    optimizer_names = [oc['name'] for oc in optimizer_configs] if optimizer_configs else ['random_search_small']
    
    # Get compatible configurations
    configurations = get_compatible_configurations(
        config, 
        target_config['datasets'], 
        target_config['models'],
        device_names[:2],  # Limit to first 2 device configs to avoid too many combinations
        optimizer_names[:2]  # Limit to first 2 optimizer configs
    )
    
    if not configurations:
        logger.error("No compatible configurations found")
        return False
    
    logger.info(f"Running {len(configurations)} configurations with {num_trials} trials each...")
    logger.info(f"Total experiments: {len(configurations) * num_trials * 2}")  # *2 for standard and genta
    
    # Run comparison
    runner = ComparisonBenchmarkRunner(f"./comparison_results_{config_name}")
    
    logger.info(f"Starting experiments...")
    start_time = time.time()
    
    runner.run_multiple_comparisons(configurations, num_trials)
    
    total_time = time.time() - start_time
    
    # Save results and generate report
    results_file = runner.save_results()
    
    # Generate trial plots if enabled
    plot_file = ""
    if enable_plotting:
        plot_file = runner.plot_trial_results()
    
    report_file = runner.generate_comparison_report()
    
    # Summary
    successful_results = [r for r in runner.results if r.success]
    standard_successful = len([r for r in successful_results if r.approach == "standard"])
    genta_successful = len([r for r in successful_results if r.approach == "genta"])
    
    # Calculate improvements
    standard_times = [r.training_time for r in successful_results if r.approach == "standard"]
    genta_times = [r.training_time for r in successful_results if r.approach == "genta"]
    standard_scores = [r.test_score for r in successful_results if r.approach == "standard"]
    genta_scores = [r.test_score for r in successful_results if r.approach == "genta"]
    standard_memory = [r.memory_peak_mb for r in successful_results if r.approach == "standard"]
    genta_memory = [r.memory_peak_mb for r in successful_results if r.approach == "genta"]
    
    speed_improvement = 0
    accuracy_improvement = 0
    memory_improvement = 0
    
    if standard_times and genta_times:
        speed_improvement = (sum(standard_times)/len(standard_times) / (sum(genta_times)/len(genta_times)) - 1) * 100
    if standard_scores and genta_scores:
        accuracy_improvement = (sum(genta_scores)/len(genta_scores) / (sum(standard_scores)/len(standard_scores)) - 1) * 100
    if standard_memory and genta_memory:
        memory_improvement = (sum(standard_memory)/len(standard_memory) / (sum(genta_memory)/len(genta_memory)) - 1) * 100
    
    # Get dataset size range
    dataset_sizes = [r.dataset_size[0] for r in successful_results]
    min_size = min(dataset_sizes) if dataset_sizes else 0
    max_size = max(dataset_sizes) if dataset_sizes else 0
    
    # Get experiment statistics
    unique_experiments = len(set(r.experiment_id for r in successful_results))
    total_trials = len(successful_results) // 2 if len(successful_results) > 1 else len(successful_results)  # Divide by 2 since each trial has 2 approaches
    
    logger.info("=" * 80)
    logger.info(f"COMPARISON '{config_name}' COMPLETED")
    logger.info("=" * 80)
    logger.info(f"📊 EXECUTION SUMMARY:")
    logger.info(f"   Total time: {total_time:.2f} seconds")
    logger.info(f"   Total experiments: {unique_experiments}")
    logger.info(f"   Total trials per experiment: {num_trials}")
    logger.info(f"   Total comparisons: {len(configurations)}")
    logger.info(f"   Standard ML successful: {standard_successful}")
    logger.info(f"   Genta AutoML successful: {genta_successful}")
    logger.info(f"   Dataset size range: {min_size:,} - {max_size:,} samples")
    logger.info("")
    logger.info(f"🚀 PERFORMANCE ANALYSIS:")
    logger.info(f"   Speed improvement: {speed_improvement:+.1f}% ({'Genta faster' if speed_improvement > 0 else 'Standard faster'})")
    logger.info(f"   Accuracy improvement: {accuracy_improvement:+.1f}% ({'Genta better' if accuracy_improvement > 0 else 'Standard better'})")
    logger.info(f"   Memory efficiency: {memory_improvement:+.1f}% ({'Genta efficient' if memory_improvement > 0 else 'Standard efficient'})")
    logger.info("")
    logger.info(f"📁 OUTPUT FILES:")
    logger.info(f"   Results: {results_file}")
    if plot_file:
        logger.info(f"   Trial Plots: {plot_file}")
    if report_file:
        logger.info(f"   Report: {report_file}")
    logger.info("=" * 80)
    
    return True

def run_custom_comparison(datasets, models, optimization_strategy, output_dir, num_trials=1):
    """Run a custom comparison configuration."""
    config = load_comparison_config()
    
    # Get device and optimizer configurations
    device_configs = config.get('device_configurations', [])
    optimizer_configs = config.get('optimizer_configurations', [])
    
    device_names = [dc['name'] for dc in device_configs] if device_configs else ['cpu_only']
    optimizer_names = [oc['name'] for oc in optimizer_configs] if optimizer_configs else ['random_search_small']
    
    # Get compatible configurations
    configurations = get_compatible_configurations(config, datasets, models, device_names[:1], optimizer_names[:1])
    
    if not configurations:
        logger.error("No compatible configurations found")
        return False
    
    # Run comparison
    runner = ComparisonBenchmarkRunner(output_dir)
    
    logger.info(f"Running {len(configurations)} custom comparisons with {num_trials} trials each...")
    start_time = time.time()
    
    runner.run_multiple_comparisons(configurations, num_trials)
    
    total_time = time.time() - start_time
    
    # Save results and generate report
    results_file = runner.save_results()
    
    # Generate trial plots
    plot_file = runner.plot_trial_results()
    
    report_file = runner.generate_comparison_report()
    
    # Summary
    successful_results = [r for r in runner.results if r.success]
    standard_successful = len([r for r in successful_results if r.approach == "standard"])
    genta_successful = len([r for r in successful_results if r.approach == "genta"])
    
    # Calculate improvements
    standard_times = [r.training_time for r in successful_results if r.approach == "standard"]
    genta_times = [r.training_time for r in successful_results if r.approach == "genta"]
    standard_scores = [r.test_score for r in successful_results if r.approach == "standard"]
    genta_scores = [r.test_score for r in successful_results if r.approach == "genta"]
    standard_memory = [r.memory_peak_mb for r in successful_results if r.approach == "standard"]
    genta_memory = [r.memory_peak_mb for r in successful_results if r.approach == "genta"]
    
    speed_improvement = 0
    accuracy_improvement = 0
    memory_improvement = 0
    
    if standard_times and genta_times:
        speed_improvement = (sum(standard_times)/len(standard_times) / (sum(genta_times)/len(genta_times)) - 1) * 100
    if standard_scores and genta_scores:
        accuracy_improvement = (sum(genta_scores)/len(genta_scores) / (sum(standard_scores)/len(standard_scores)) - 1) * 100
    if standard_memory and genta_memory:
        memory_improvement = (sum(standard_memory)/len(standard_memory) / (sum(genta_memory)/len(genta_memory)) - 1) * 100
    
    # Get dataset size range
    dataset_sizes = [r.dataset_size[0] for r in successful_results]
    min_size = min(dataset_sizes) if dataset_sizes else 0
    max_size = max(dataset_sizes) if dataset_sizes else 0
    
    # Get experiment statistics
    unique_experiments = len(set(r.experiment_id for r in successful_results))
    
    logger.info("=" * 80)
    logger.info("CUSTOM COMPARISON COMPLETED")
    logger.info("=" * 80)
    logger.info(f"📊 EXECUTION SUMMARY:")
    logger.info(f"   Total time: {total_time:.2f} seconds")
    logger.info(f"   Total experiments: {unique_experiments}")
    logger.info(f"   Total trials per experiment: {num_trials}")
    logger.info(f"   Total comparisons: {len(configurations)}")
    logger.info(f"   Standard ML successful: {standard_successful}")
    logger.info(f"   Genta AutoML successful: {genta_successful}")
    logger.info(f"   Dataset size range: {min_size:,} - {max_size:,} samples")
    logger.info("")
    logger.info(f"🚀 PERFORMANCE ANALYSIS:")
    logger.info(f"   Speed improvement: {speed_improvement:+.1f}% ({'Genta faster' if speed_improvement > 0 else 'Standard faster'})")
    logger.info(f"   Accuracy improvement: {accuracy_improvement:+.1f}% ({'Genta better' if accuracy_improvement > 0 else 'Standard better'})")
    logger.info(f"   Memory efficiency: {memory_improvement:+.1f}% ({'Genta efficient' if memory_improvement > 0 else 'Standard efficient'})")
    logger.info("")
    logger.info(f"📁 OUTPUT FILES:")
    logger.info(f"   Results: {results_file}")
    if plot_file:
        logger.info(f"   Trial Plots: {plot_file}")
    if report_file:
        logger.info(f"   Report: {report_file}")
    logger.info("=" * 80)
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run Genta AutoML vs Standard ML comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Predefined Configurations:
  quick_comparison         - Quick test on small datasets
  comprehensive_small      - Comprehensive test on small/medium datasets  
  performance_test         - Performance test on larger datasets
  scalability_test         - Test scalability across dataset sizes
  large_scale_test         - Large scale performance test for big datasets
  massive_scale_test       - Massive scale test up to 10M samples
  regression_scalability_test - Regression scalability test across dataset sizes
  full_scalability_comparison - Complete scalability comparison from small to massive datasets
  algorithm_comparison     - Compare algorithms on same datasets

Examples:
  python run_comparison.py
  python run_comparison.py --config comprehensive_small
  python run_comparison.py --config massive_scale_test --trials 3
  python run_comparison.py --custom --datasets iris wine --models random_forest logistic_regression --trials 5
        """
    )
    
    parser.add_argument(
        "--config", 
        default="quick_comparison",
        help="Predefined configuration to run (default: quick_comparison)"
    )
    parser.add_argument(
        "--list-configs", 
        action="store_true",
        help="List available predefined configurations"
    )
    parser.add_argument(
        "--custom", 
        action="store_true",
        help="Run custom comparison (requires --datasets and --models)"
    )
    parser.add_argument(
        "--datasets", 
        nargs="+",
        help="Datasets for custom comparison"
    )
    parser.add_argument(
        "--models", 
        nargs="+",
        help="Models for custom comparison"
    )
    parser.add_argument(
        "--optimization", 
        choices=["grid_search", "random_search"],
        default="random_search",
        help="Optimization strategy (default: random_search)"
    )
    parser.add_argument(
        "--trials", 
        type=int,
        default=1,
        help="Number of trials to run for each experiment (default: 1)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./comparison_results_custom",
        help="Output directory for custom comparison"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_comparison_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # List configurations
    if args.list_configs:
        print("\nAvailable Comparison Configurations:")
        print("=" * 50)
        for comp_config in config.get('comparison_configurations', []):
            print(f"\n{comp_config['name']}")
            print(f"  Description: {comp_config['description']}")
            print(f"  Datasets: {', '.join(comp_config['datasets'])}")
            print(f"  Models: {', '.join(comp_config['models'])}")
        print()
        return
    
    # Run custom comparison
    if args.custom:
        if not args.datasets or not args.models:
            logger.error("Custom comparison requires --datasets and --models arguments")
            sys.exit(1)
        
        success = run_custom_comparison(
            args.datasets, 
            args.models, 
            args.optimization,
            args.output_dir,
            args.trials
        )
        sys.exit(0 if success else 1)
    
    # Run predefined comparison
    success = run_predefined_comparison(args.config, config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
