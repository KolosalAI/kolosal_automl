#!/usr/bin/env python3
"""
Simple test script to debug Kolosal AutoML issues
"""

import sys
import os

# Handle Windows console encoding issues
if sys.platform.startswith('win'):
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

from benchmark.benchmark_comparison import ComparisonBenchmarkRunner, logger, KolosalMLBenchmark, DatasetManager

def test_single_kolosal():
    """Test a single Kolosal ML run to debug issues."""
    logger.info("Testing single Kolosal AutoML run...")
    
    try:
        # Test the Kolosal benchmark directly
        kolosal_benchmark = KolosalMLBenchmark()
        
        # Run a simple test
        result = kolosal_benchmark.run_benchmark(
            dataset_name="iris",
            model_name="random_forest",
            optimization_strategy="random_search",
            experiment_id="TEST_001",
            trial_number=1,
            device_config="cpu_only",
            optimizer_config="random_search_small"
        )
        
        logger.info(f"Kolosal benchmark result: success={result.success}")
        if not result.success:
            logger.error(f"Error: {result.error_message}")
        else:
            logger.info(f"Training time: {result.training_time:.2f}s")
            logger.info(f"Test score: {result.test_score:.4f}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_loading():
    """Test dataset loading"""
    logger.info("Testing dataset loading...")
    try:
        X, y, task_type = DatasetManager.load_dataset("iris")
        logger.info(f"Loaded iris: shape={X.shape}, task_type={task_type}")
        
        X, y, task_type = DatasetManager.load_dataset("diabetes")
        logger.info(f"Loaded diabetes: shape={X.shape}, task_type={task_type}")
        
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("*** Kolosal AutoML Debug Test ***")
    
    # Test dataset loading first
    test_dataset_loading()
    
    # Test single run
    test_single_kolosal()
    
    logger.info("*** Test completed ***")
