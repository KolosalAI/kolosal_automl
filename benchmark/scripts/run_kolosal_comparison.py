#!/usr/bin/env python3
"""
Simple script to run Kolosal AutoML vs Standard ML comparison
"""

import sys
import time
import os

# Handle Windows console encoding issues
if sys.platform.startswith('win'):
    # Try to set UTF-8 encoding for Windows console
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        # Fallback: just continue with default encoding
        pass

# Add the benchmark directory to the path to import benchmark_comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmark_comparison import ComparisonBenchmarkRunner, logger

def quick_comparison():
    """Run a quick comparison on small datasets."""
    logger.info("Starting Kolosal AutoML vs Standard ML Quick Comparison")
    
    # Create benchmark runner
    runner = ComparisonBenchmarkRunner("./comparison_results_quick_comparison")
    
    # Define configurations for quick test
    configurations = [
        {"dataset": "iris", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "iris", "model": "logistic_regression", "optimization_strategy": "random_search"},
        {"dataset": "wine", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "breast_cancer", "model": "gradient_boosting", "optimization_strategy": "random_search"},
        {"dataset": "diabetes", "model": "ridge", "optimization_strategy": "random_search"},
        {"dataset": "synthetic_small_classification", "model": "random_forest", "optimization_strategy": "random_search"},
        {"dataset": "synthetic_small_regression", "model": "ridge", "optimization_strategy": "random_search"},
    ]
    
    logger.info(f"Running {len(configurations)} quick comparisons...")
    
    # Run comparisons
    start_time = time.time()
    try:
        runner.run_multiple_comparisons(configurations, num_trials=1)
        total_time = time.time() - start_time
        
        # Save results and generate report
        results_file = runner.save_results()
        report_file = runner.generate_comparison_report()
        
        # Summary
        successful_results = [r for r in runner.results if r.success]
        standard_successful = len([r for r in successful_results if r.approach == "standard"])
        kolosal_successful = len([r for r in successful_results if r.approach == "kolosal"])
        
        logger.info("=" * 60)
        logger.info("QUICK COMPARISON COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Total comparisons: {len(configurations)}")
        logger.info(f"Standard ML successful: {standard_successful}")
        logger.info(f"Kolosal AutoML successful: {kolosal_successful}")
        logger.info(f"Results saved to: {results_file}")
        if report_file:
            logger.info(f"Report generated: {report_file}")
        
        # Display quick stats
        if successful_results:
            std_results = [r for r in successful_results if r.approach == "standard"]
            kolosal_results = [r for r in successful_results if r.approach == "kolosal"]
            
            if std_results and kolosal_results:
                avg_std_time = sum(r.training_time for r in std_results) / len(std_results)
                avg_kolosal_time = sum(r.training_time for r in kolosal_results) / len(kolosal_results)
                
                avg_std_score = sum(r.test_score for r in std_results) / len(std_results)
                avg_kolosal_score = sum(r.test_score for r in kolosal_results) / len(kolosal_results)
                
                speed_improvement = ((avg_std_time / avg_kolosal_time) - 1) * 100 if avg_kolosal_time > 0 else 0
                accuracy_improvement = ((avg_kolosal_score / avg_std_score) - 1) * 100 if avg_std_score > 0 else 0
                
                logger.info("\n*** QUICK RESULTS SUMMARY ***")
                logger.info(f"   Speed: Kolosal is {speed_improvement:+.1f}% {'faster' if speed_improvement > 0 else 'slower'}")
                logger.info(f"   Accuracy: Kolosal is {accuracy_improvement:+.1f}% {'better' if accuracy_improvement > 0 else 'worse'}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Quick comparison failed: {e}")
        return False
    
    return True

def comprehensive_comparison():
    """Run a comprehensive comparison including larger datasets."""
    logger.info("Starting Kolosal AutoML vs Standard ML Comprehensive Comparison")
    
    # Create benchmark runner
    runner = ComparisonBenchmarkRunner("./comparison_results_comprehensive")
    
    # Define configurations for comprehensive test
    configurations = []
    
    # Small datasets
    small_datasets = ["iris", "wine", "breast_cancer", "digits", "diabetes"]
    # Medium synthetic datasets
    medium_datasets = ["synthetic_medium_classification", "synthetic_medium_regression"]
    
    models_classification = ["random_forest", "gradient_boosting", "logistic_regression"]
    models_regression = ["random_forest", "gradient_boosting", "ridge"]
    
    # Add small dataset combinations
    for dataset in small_datasets:
        if dataset in ["diabetes"]:  # regression
            models = models_regression
        else:  # classification
            models = models_classification
            
        for model in models:
            # Skip incompatible combinations
            if (dataset == "diabetes" and model == "logistic_regression") or \
               (dataset != "diabetes" and model == "ridge"):
                continue
                
            configurations.append({
                "dataset": dataset,
                "model": model,
                "optimization_strategy": "random_search"
            })
    
    # Add medium synthetic datasets
    for dataset in medium_datasets:
        models = models_regression if "regression" in dataset else models_classification
        for model in models:
            if ("regression" in dataset and model == "logistic_regression") or \
               ("classification" in dataset and model == "ridge"):
                continue
                
            configurations.append({
                "dataset": dataset,
                "model": model,
                "optimization_strategy": "random_search"
            })
    
    logger.info(f"Running {len(configurations)} comprehensive comparisons...")
    
    # Run comparisons
    start_time = time.time()
    try:
        runner.run_multiple_comparisons(configurations, num_trials=2)
        total_time = time.time() - start_time
        
        # Save results and generate report
        results_file = runner.save_results()
        report_file = runner.generate_comparison_report()
        
        # Summary
        successful_results = [r for r in runner.results if r.success]
        standard_successful = len([r for r in successful_results if r.approach == "standard"])
        kolosal_successful = len([r for r in successful_results if r.approach == "kolosal"])
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE COMPARISON COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Total comparisons: {len(configurations)}")
        logger.info(f"Standard ML successful: {standard_successful}")
        logger.info(f"Kolosal AutoML successful: {kolosal_successful}")
        logger.info(f"Results saved to: {results_file}")
        if report_file:
            logger.info(f"Report generated: {report_file}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Comprehensive comparison failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kolosal AutoML Comparison")
    parser.add_argument("--mode", choices=["quick", "comprehensive"], default="quick",
                       help="Comparison mode: quick (small datasets) or comprehensive (includes larger datasets)")
    
    args = parser.parse_args()
    
    logger.info("*** Kolosal AutoML Comparison Benchmark ***")
    logger.info("This script compares Kolosal AutoML against standard scikit-learn approaches")
    logger.info("=" * 80)
    
    if args.mode == "quick":
        success = quick_comparison()
    else:
        success = comprehensive_comparison()
    
    if success:
        logger.info("*** Comparison completed successfully! ***")
        logger.info("Check the output directory for detailed results and HTML report.")
    else:
        logger.error("*** Comparison failed! ***")
        sys.exit(1)
