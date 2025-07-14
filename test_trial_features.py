#!/usr/bin/env python3
"""
Test script to demonstrate the new trial and experiment features.

This script shows how to use the enhanced comparison runner with:
- Experiment numbering
- Multiple trials per experiment
- Device/optimizer configurations
- Trial plotting
"""

import sys
import json
from pathlib import Path
from run_comparison import load_comparison_config, get_compatible_configurations, run_custom_comparison

def test_configuration_loading():
    """Test loading the updated configuration with new features."""
    print("Testing configuration loading...")
    
    config = load_comparison_config()
    
    # Check for new sections
    assert 'device_configurations' in config, "Device configurations not found"
    assert 'optimizer_configurations' in config, "Optimizer configurations not found"
    assert 'num_trials' in config['comparison_settings'], "num_trials setting not found"
    
    print(f"✓ Found {len(config['device_configurations'])} device configurations")
    print(f"✓ Found {len(config['optimizer_configurations'])} optimizer configurations")
    print(f"✓ Default trials: {config['comparison_settings']['num_trials']}")
    
    return config

def test_configuration_generation():
    """Test the new configuration generation with device/optimizer combinations."""
    print("\nTesting configuration generation...")
    
    config = load_comparison_config()
    
    datasets = ['iris', 'wine']
    models = ['random_forest']
    
    # Test with device and optimizer configurations
    device_configs = [dc['name'] for dc in config['device_configurations']][:2]
    optimizer_configs = [oc['name'] for oc in config['optimizer_configurations']][:2]
    
    configurations = get_compatible_configurations(
        config, datasets, models, device_configs, optimizer_configs
    )
    
    print(f"✓ Generated {len(configurations)} configurations")
    print("Configuration examples:")
    for i, cfg in enumerate(configurations[:3]):
        print(f"  {i+1}. Dataset: {cfg['dataset']}, Model: {cfg['model']}")
        print(f"     Device: {cfg['device_config']}, Optimizer: {cfg['optimizer_config']}")
    
    return configurations

def test_trial_simulation():
    """Simulate trial results to test plotting functionality."""
    print("\nTesting trial result simulation...")
    
    # This would normally run actual benchmarks, but for testing we'll simulate
    from benchmark_comparison import ComparisonBenchmarkRunner, BenchmarkResult
    import random
    import time
    
    runner = ComparisonBenchmarkRunner("./test_results")
    
    # Create simulated results for testing
    for exp_num in range(1, 3):  # 2 experiments
        experiment_id = f"EXP_{exp_num:03d}_iris_random_forest_cpu_only_random_search_small"
        
        for trial in range(1, 4):  # 3 trials each
            # Simulate standard result
            std_result = BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial,
                approach="standard",
                dataset_name="iris",
                model_name="random_forest",
                dataset_size=(150, 4),
                task_type="classification",
                device_config="cpu_only",
                optimizer_config="random_search_small",
                training_time=random.uniform(1.0, 3.0),
                prediction_time=random.uniform(0.01, 0.05),
                memory_peak_mb=random.uniform(50, 100),
                memory_final_mb=random.uniform(30, 60),
                train_score=random.uniform(0.95, 0.99),
                test_score=random.uniform(0.90, 0.96),
                cv_score_mean=random.uniform(0.92, 0.97),
                cv_score_std=random.uniform(0.01, 0.03),
                best_params={'n_estimators': 100, 'max_depth': 10},
                feature_count=4,
                model_size_mb=random.uniform(0.5, 2.0),
                preprocessing_time=random.uniform(0.01, 0.1),
                success=True,
                error_message=""
            )
            
            # Simulate genta result (slightly better performance)
            genta_result = BenchmarkResult(
                experiment_id=experiment_id,
                trial_number=trial,
                approach="genta",
                dataset_name="iris",
                model_name="random_forest",
                dataset_size=(150, 4),
                task_type="classification",
                device_config="cpu_only",
                optimizer_config="random_search_small",
                training_time=std_result.training_time * random.uniform(0.7, 0.9),  # Faster
                prediction_time=std_result.prediction_time * random.uniform(0.8, 1.0),
                memory_peak_mb=std_result.memory_peak_mb * random.uniform(0.8, 1.1),
                memory_final_mb=std_result.memory_final_mb * random.uniform(0.8, 1.0),
                train_score=std_result.train_score * random.uniform(1.0, 1.02),  # Slightly better
                test_score=std_result.test_score * random.uniform(1.0, 1.02),
                cv_score_mean=std_result.cv_score_mean * random.uniform(1.0, 1.02),
                cv_score_std=std_result.cv_score_std,
                best_params={'n_estimators': 100, 'max_depth': 15},
                feature_count=4,
                model_size_mb=std_result.model_size_mb * random.uniform(0.9, 1.1),
                preprocessing_time=0.0,  # Included in training time
                success=True,
                error_message=""
            )
            
            runner.results.extend([std_result, genta_result])
    
    print(f"✓ Created {len(runner.results)} simulated results")
    
    # Test saving results
    results_file = runner.save_results()
    print(f"✓ Results saved to: {results_file}")
    
    # Test trial plotting
    plot_file = runner.plot_trial_results()
    print(f"✓ Trial plots saved to: {plot_file}")
    
    # Test statistics generation
    stats = runner.generate_trial_statistics()
    print(f"✓ Generated statistics for {stats['total_experiments']} experiments")
    
    return runner

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Enhanced Trial and Experiment Features")
    print("=" * 60)
    
    try:
        # Test 1: Configuration loading
        config = test_configuration_loading()
        
        # Test 2: Configuration generation
        configurations = test_configuration_generation()
        
        # Test 3: Trial simulation
        runner = test_trial_simulation()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        print("\nNew Features Summary:")
        print("1. ✓ Experiment numbering with unique IDs")
        print("2. ✓ Multiple trials per experiment")
        print("3. ✓ Device configuration support")
        print("4. ✓ Optimizer configuration support")
        print("5. ✓ Trial result plotting")
        print("6. ✓ Comprehensive trial statistics")
        print("7. ✓ Enhanced result saving with metadata")
        
        print(f"\nTest results saved in: {runner.output_dir}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
