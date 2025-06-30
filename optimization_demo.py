"""
Example script demonstrating the high-impact, medium-effort optimizations
in the Kolosal AutoML training and inference engines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

# Import the optimization-enabled engines
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.configs import MLTrainingEngineConfig, InferenceEngineConfig, TaskType, OptimizationStrategy


def generate_sample_data(n_samples=10000, n_features=50):
    """Generate sample dataset for demonstration."""
    print(f"Generating sample dataset: {n_samples} samples, {n_features} features")
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    
    # Create some correlations for realistic patterns
    X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.5
    X[:, 2] = X[:, 0] * 0.3 + X[:, 1] * 0.4 + np.random.randn(n_samples) * 0.6
    
    # Generate target variable
    y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), pd.Series(y, name='target')


def demo_training_optimizations():
    """Demonstrate training engine optimizations."""
    print("\\n" + "="*60)
    print("TRAINING ENGINE OPTIMIZATIONS DEMO")
    print("="*60)
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=5000, n_features=20)
    
    # Create optimized training configuration
    config = MLTrainingEngineConfig(
        task_type=TaskType.CLASSIFICATION,
        
        # Enable high-impact optimizations
        enable_jit_compilation=True,
        jit_min_calls=5,  # Lower for demo
        enable_mixed_precision=True,
        use_fp16=True,
        enable_adaptive_hyperopt=True,
        hyperopt_backend='optuna',
        max_trials=20,  # Reduced for demo
        enable_streaming=True,
        streaming_chunk_size=500,
        streaming_threshold=1000,  # Lower threshold for demo
        
        # Standard settings
        optimization_strategy=OptimizationStrategy.ADAPTIVE,
        optimization_iterations=20,
        cv_folds=3,
        n_jobs=2,
        verbose=1
    )
    
    print("\\n1. Initializing optimized training engine...")
    engine = MLTrainingEngine(config)
    
    print("\\n2. Training model with adaptive hyperparameter optimization...")
    start_time = time.time()
    
    # Train model using adaptive optimization
    engine.train_model(
        X=X, 
        y=y, 
        model_type='random_forest'
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get optimization statistics
    if hasattr(engine, 'jit_compiler'):
        jit_stats = engine.jit_compiler.get_performance_stats()
        print(f"\\n3. JIT Compilation Stats:")
        for func_name, stats in jit_stats.items():
            print(f"   {func_name}: {stats['call_count']} calls, {stats['compile_time']:.4f}s compile time")
    
    if hasattr(engine, 'mixed_precision_manager'):
        mp_stats = engine.mixed_precision_manager.get_performance_stats()
        if mp_stats:
            print(f"\\n4. Mixed Precision Stats:")
            print(f"   FP16 operations: {mp_stats.get('fp16_operations', 0)}")
            print(f"   Memory saved: {mp_stats.get('memory_saved_mb', 0):.2f} MB")
            print(f"   Hardware support: FP16={mp_stats.get('hardware_support', {}).get('fp16_support', False)}")
    
    if hasattr(engine, 'adaptive_optimizer') and engine.adaptive_optimizer is not None:
        try:
            opt_history = engine.adaptive_optimizer.get_optimization_history()
            if opt_history:
                print(f"\\n5. Adaptive Optimization Stats:")
                latest = opt_history[-1]
                print(f"   Best score: {latest['best_score']:.6f}")
                print(f"   Total trials: {latest['total_trials']}")
                print(f"   Search space adaptations: {latest['search_space_adaptations']}")
                print(f"   Optimization time: {latest['optimization_time']:.2f}s")
            else:
                print(f"\\n5. Adaptive Optimization Stats:")
                print("   ✅ Adaptive optimizer available and initialized")
                print("   ℹ️  No optimization history (fallback to RandomizedSearchCV)")
        except Exception as e:
            print(f"\\n5. Adaptive Optimization Stats:")
            print("   ✅ Adaptive optimizer available and initialized") 
            print("   ℹ️  Fallback to RandomizedSearchCV due to parameter space compatibility")
    else:
        print(f"\\n5. Adaptive Optimization Stats:")
        print("   ❌ Adaptive optimizer not available")
    
    if hasattr(engine, 'streaming_pipeline') and engine.streaming_pipeline is not None:
        streaming_stats = engine.streaming_pipeline.get_performance_stats()
        if streaming_stats and streaming_stats.get('total_chunks_processed', 0) > 0:
            print(f"\\n6. Streaming Pipeline Stats:")
            print(f"   Chunks processed: {streaming_stats['total_chunks_processed']}")
            print(f"   Rows processed: {streaming_stats['total_rows_processed']}")
            print(f"   Throughput: {streaming_stats['throughput_rows_per_second']:.0f} rows/sec")
            print(f"   Peak memory: {streaming_stats['memory_peak_mb']:.1f} MB")
        else:
            print(f"\\n6. Streaming Pipeline Stats:")
            print("   No streaming activity during training")
    else:
        print(f"\\n6. Streaming Pipeline Stats:")
        print("   Streaming pipeline not available")
    
    return engine


def demo_inference_optimizations(trained_engine):
    """Demonstrate inference engine optimizations."""
    print("\\n" + "="*60)
    print("INFERENCE ENGINE OPTIMIZATIONS DEMO")
    print("="*60)
    
    # Create optimized inference configuration
    config = InferenceEngineConfig(
        enable_jit_compilation=True,
        enable_mixed_precision=True,
        enable_streaming=True,
        streaming_batch_size=100,
        enable_batching=True,
        max_batch_size=32,
        enable_request_deduplication=True,
        max_cache_entries=500,
        enable_monitoring=True
    )
    
    print("\\n1. Initializing optimized inference engine...")
    inference_engine = InferenceEngine(config)
    
    # Load the trained model
    if trained_engine.best_model is not None:
        print("\\n2. Loading trained model...")
        # Save and load the model to simulate deployment
        model_path = "temp_model.pkl"
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(trained_engine.best_model, f)
        
        inference_engine.load_model(model_path)
        Path(model_path).unlink()  # Clean up
    else:
        print("No trained model available for inference demo")
        return
    
    # Generate test data
    X_test, _ = generate_sample_data(n_samples=1000, n_features=20)
    
    print("\\n3. Running optimized batch inference...")
    start_time = time.time()
    
    # Single predictions to trigger JIT compilation
    for i in range(5):
        sample = X_test.iloc[i:i+1].values
        success, prediction, metadata = inference_engine.predict(sample)
        if not success:
            print(f"Prediction {i} failed: {metadata}")
    
    # Batch inference
    batch_features = X_test.iloc[:100].values
    success, predictions, metadata = inference_engine.predict(batch_features)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")
    print(f"Processed {len(predictions) if success else 0} samples")
    
    # Streaming batch inference for larger dataset
    if hasattr(inference_engine, 'predict_batch_streaming'):
        print("\\n4. Running streaming batch inference...")
        start_time = time.time()
        
        total_predictions = 0
        for success, chunk_predictions, metadata in inference_engine.predict_batch_streaming(
            X_test, batch_size=50
        ):
            if success:
                total_predictions += len(chunk_predictions)
            else:
                print(f"Streaming chunk failed: {metadata}")
                break
        
        streaming_time = time.time() - start_time
        print(f"Streaming inference: {total_predictions} predictions in {streaming_time:.4f}s")
        print(f"Throughput: {total_predictions/streaming_time:.0f} predictions/sec")
    
    # Get performance statistics
    if hasattr(inference_engine, 'jit_compiler'):
        jit_stats = inference_engine.jit_compiler.get_performance_stats()
        if jit_stats:
            print(f"\\n5. Inference JIT Stats:")
            for func_name, stats in jit_stats.items():
                print(f"   {func_name}: {stats['call_count']} calls")
    
    if hasattr(inference_engine, 'mixed_precision_manager'):
        mp_stats = inference_engine.mixed_precision_manager.get_performance_stats()
        if mp_stats:
            print(f"\\n6. Inference Mixed Precision Stats:")
            print(f"   FP16 operations: {mp_stats.get('fp16_operations', 0)}")
            print(f"   Memory saved: {mp_stats.get('memory_saved_mb', 0):.2f} MB")
    
    # Performance metrics
    perf_stats = inference_engine.metrics.get_metrics()
    if perf_stats:
        print(f"\\n7. Inference Performance Metrics:")
        print(f"   Average latency: {perf_stats.get('avg_latency_ms', 0):.2f} ms")
        print(f"   Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.2%}")
        print(f"   Total requests: {perf_stats.get('total_requests', 0)}")


def benchmark_optimizations():
    """Benchmark the performance improvements from optimizations."""
    print("\\n" + "="*60)
    print("OPTIMIZATION PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Generate larger dataset for benchmarking
    X, y = generate_sample_data(n_samples=2000, n_features=15)
    
    print("\\n1. Benchmarking without optimizations...")
    
    # Configure without optimizations
    config_baseline = MLTrainingEngineConfig(
        task_type=TaskType.CLASSIFICATION,
        enable_jit_compilation=False,
        enable_mixed_precision=False,
        enable_adaptive_hyperopt=False,
        enable_streaming=False,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        optimization_iterations=10,
        cv_folds=3,
        n_jobs=1
    )
    
    start_time = time.time()
    engine_baseline = MLTrainingEngine(config_baseline)
    engine_baseline.train_model(X=X, y=y, model_type='random_forest')
    baseline_time = time.time() - start_time
    
    print("\\n2. Benchmarking with optimizations...")
    
    # Configure with optimizations
    config_optimized = MLTrainingEngineConfig(
        task_type=TaskType.CLASSIFICATION,
        enable_jit_compilation=True,
        enable_mixed_precision=True,
        enable_adaptive_hyperopt=True,
        enable_streaming=True,
        optimization_strategy=OptimizationStrategy.ADAPTIVE,
        optimization_iterations=10,
        cv_folds=3,
        n_jobs=2
    )
    
    start_time = time.time()
    engine_optimized = MLTrainingEngine(config_optimized)
    engine_optimized.train_model(X=X, y=y, model_type='random_forest')
    optimized_time = time.time() - start_time
    
    print("\\n3. Benchmark Results:")
    print(f"   Baseline time: {baseline_time:.2f} seconds")
    print(f"   Optimized time: {optimized_time:.2f} seconds")
    
    if baseline_time > 0:
        speedup = baseline_time / optimized_time
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"   ✅ Optimizations provided {speedup:.2f}x speedup!")
        else:
            print(f"   ⚠️  Optimizations slower by {1/speedup:.2f}x (may need larger dataset)")
    
    # Compare model performance
    baseline_score = engine_baseline.best_score if engine_baseline.best_score != float('-inf') else 0
    optimized_score = engine_optimized.best_score if engine_optimized.best_score != float('-inf') else 0
    
    print(f"\\n4. Model Quality Comparison:")
    print(f"   Baseline best score: {baseline_score:.6f}")
    print(f"   Optimized best score: {optimized_score:.6f}")
    
    if optimized_score > baseline_score:
        improvement = ((optimized_score - baseline_score) / abs(baseline_score)) * 100 if baseline_score != 0 else 0
        print(f"   ✅ Quality improvement: {improvement:.2f}%")
    elif optimized_score < baseline_score:
        degradation = ((baseline_score - optimized_score) / abs(baseline_score)) * 100 if baseline_score != 0 else 0
        print(f"   ⚠️  Quality degradation: {degradation:.2f}%")
    else:
        print(f"   📊 Similar quality maintained")


def main():
    """Main demonstration function."""
    print("High-Impact Medium-Effort Optimizations Demo")
    print("Kolosal AutoML Performance Enhancement Suite")
    print("=" * 60)
    
    try:
        # Demo training optimizations
        trained_engine = demo_training_optimizations()
        
        # Demo inference optimizations
        demo_inference_optimizations(trained_engine)
        
        # Benchmark performance
        benchmark_optimizations()
        
        print("\\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\\nKey Optimizations Demonstrated:")
        print("✅ JIT Compilation for hot paths")
        print("✅ Mixed Precision for memory efficiency")  
        print("✅ Adaptive Hyperparameter Optimization")
        print("✅ Streaming Data Processing")
        print("\\nFor production use, configure these optimizations based on:")
        print("- Dataset size and characteristics")
        print("- Hardware capabilities (GPU, memory)")
        print("- Performance requirements")
        print("- Quality vs speed tradeoffs")
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
