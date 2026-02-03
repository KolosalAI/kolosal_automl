#!/usr/bin/env python3
# ---------------------------------------------------------------------
# scripts/benchmark_optimize.py - Benchmark optimization utility
# ---------------------------------------------------------------------
"""
Utility script to optimize system settings for running benchmarks.
This script configures the system for maximum performance during benchmark runs.
"""

import os
import sys
import logging
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from modules.device_optimizer import DeviceOptimizer, OptimizationMode
    DEVICE_OPTIMIZER_AVAILABLE = True
except ImportError:
    DEVICE_OPTIMIZER_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_optimizer")


class BenchmarkOptimizer:
    """Optimizes system configuration for benchmark execution."""
    
    def __init__(self):
        self.original_env = {}
        self.applied_optimizations = []
        
    def save_original_environment(self):
        """Save original environment variables."""
        env_vars_to_save = [
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'
        ]
        
        for var in env_vars_to_save:
            self.original_env[var] = os.environ.get(var)
    
    def apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations."""
        try:
            # Get CPU information
            cpu_count = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False) or cpu_count
            
            # Set threading environment variables for optimal performance
            thread_configs = {
                'OMP_NUM_THREADS': str(physical_cores),
                'MKL_NUM_THREADS': str(physical_cores),
                'OPENBLAS_NUM_THREADS': str(physical_cores),
                'NUMEXPR_NUM_THREADS': str(physical_cores),
                'VECLIB_MAXIMUM_THREADS': str(physical_cores),
                'OMP_DYNAMIC': 'FALSE',
                'MKL_DYNAMIC': 'FALSE'
            }
            
            for var, value in thread_configs.items():
                os.environ[var] = value
                logger.info(f"Set {var}={value}")
            
            self.applied_optimizations.append(f"CPU threading optimized for {physical_cores} cores")
            
            # Set CPU affinity on supported platforms
            if hasattr(os, 'sched_setaffinity'):
                try:
                    available_cpus = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, available_cpus)
                    self.applied_optimizations.append("CPU affinity set to use all available cores")
                except Exception as e:
                    logger.warning(f"Failed to set CPU affinity: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply CPU optimizations: {e}")
            return False
    
    def apply_memory_optimizations(self):
        """Apply memory-specific optimizations."""
        try:
            # Get memory information
            memory_info = psutil.virtual_memory()
            total_gb = memory_info.total / (1024**3)
            available_gb = memory_info.available / (1024**3)
            
            # Set memory-related environment variables
            memory_configs = {
                'MALLOC_TRIM_THRESHOLD_': '0',  # Disable malloc trimming for performance
                'MALLOC_MMAP_THRESHOLD_': str(1024 * 1024),  # Use mmap for large allocations
            }
            
            for var, value in memory_configs.items():
                os.environ[var] = value
                logger.info(f"Set {var}={value}")
            
            self.applied_optimizations.append(f"Memory optimized ({available_gb:.1f}GB available)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply memory optimizations: {e}")
            return False
    
    def apply_process_priority(self):
        """Set high process priority for benchmarks."""
        try:
            current_process = psutil.Process()
            
            if os.name == 'nt':  # Windows
                try:
                    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                    self.applied_optimizations.append("Process priority set to HIGH")
                except PermissionError:
                    logger.warning("Insufficient permissions to set high priority on Windows")
                    
            else:  # Unix-like systems
                try:
                    current_process.nice(-10)  # Negative values = higher priority
                    self.applied_optimizations.append("Process nice value set to -10")
                except PermissionError:
                    logger.warning("Insufficient permissions to set high priority on Unix")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set process priority: {e}")
            return False
    
    def apply_device_optimizer_config(self):
        """Apply device optimizer configuration if available."""
        if not DEVICE_OPTIMIZER_AVAILABLE:
            logger.warning("Device optimizer not available")
            return False
        
        try:
            # Create device optimizer for performance mode
            optimizer = DeviceOptimizer(
                optimization_mode=OptimizationMode.PERFORMANCE,
                workload_type="inference",
                environment="auto",
                enable_specialized_accelerators=True,
                memory_reservation_percent=5.0,
                power_efficiency=False,
                auto_tune=True,
                debug_mode=False
            )
            
            # Get system information
            system_info = optimizer.get_system_info()
            
            # Log optimization details
            accelerators = system_info.get('accelerators', [])
            if accelerators:
                self.applied_optimizations.append(f"Hardware accelerators detected: {accelerators}")
            
            cpu_features = system_info.get('cpu_features', {})
            enabled_features = [k for k, v in cpu_features.items() if v]
            if enabled_features:
                self.applied_optimizations.append(f"CPU features enabled: {enabled_features}")
            
            self.applied_optimizations.append("Device optimizer configuration applied")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply device optimizer config: {e}")
            return False
    
    def optimize_for_benchmarks(self) -> Dict[str, Any]:
        """Apply all optimizations for benchmark execution."""
        logger.info("Starting benchmark optimization...")
        
        # Save original environment
        self.save_original_environment()
        
        # Apply optimizations
        results = {
            'cpu_optimizations': self.apply_cpu_optimizations(),
            'memory_optimizations': self.apply_memory_optimizations(),
            'process_priority': self.apply_process_priority(),
            'device_optimizer': self.apply_device_optimizer_config()
        }
        
        # Summary
        successful_optimizations = sum(results.values())
        total_optimizations = len(results)
        
        logger.info(f"Optimization complete: {successful_optimizations}/{total_optimizations} successful")
        
        if self.applied_optimizations:
            logger.info("Applied optimizations:")
            for opt in self.applied_optimizations:
                logger.info(f"  - {opt}")
        
        return {
            'success': successful_optimizations > 0,
            'results': results,
            'optimizations': self.applied_optimizations,
            'system_info': self.get_system_summary()
        }
    
    def restore_environment(self):
        """Restore original environment variables."""
        logger.info("Restoring original environment...")
        
        for var, value in self.original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value
        
        logger.info("Environment restored")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of system resources."""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            memory_info = psutil.virtual_memory()
            
            return {
                'cpu_logical_cores': cpu_count,
                'cpu_physical_cores': physical_cores,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_available_gb': memory_info.available / (1024**3),
                'memory_percent_used': memory_info.percent,
                'platform': sys.platform,
                'python_version': sys.version
            }
            
        except Exception as e:
            logger.error(f"Failed to get system summary: {e}")
            return {}


def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize system settings for benchmark execution"
    )
    parser.add_argument(
        '--restore', 
        action='store_true',
        help='Restore original environment (use after benchmarks)'
    )
    parser.add_argument(
        '--info-only',
        action='store_true', 
        help='Show system information without applying optimizations'
    )
    
    args = parser.parse_args()
    
    optimizer = BenchmarkOptimizer()
    
    if args.info_only:
        print("System Information:")
        print("=" * 50)
        system_info = optimizer.get_system_summary()
        for key, value in system_info.items():
            print(f"{key}: {value}")
        return
    
    if args.restore:
        optimizer.restore_environment()
        print("Environment restored to original state")
        return
    
    # Apply optimizations
    result = optimizer.optimize_for_benchmarks()
    
    if result['success']:
        print("✅ Benchmark optimization completed successfully!")
        print("\nApplied optimizations:")
        for opt in result['optimizations']:
            print(f"  • {opt}")
        
        print("\n🚀 System is now optimized for benchmark execution")
        print("   Run your benchmarks now for best performance")
        print("   Use --restore flag to restore original settings after benchmarks")
    else:
        print("❌ Benchmark optimization failed")
        print("   Check logs for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
