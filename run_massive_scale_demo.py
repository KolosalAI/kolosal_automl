#!/usr/bin/env python3
"""
Massive Scale Comparison Demo

This script demonstrates the enhanced comparison capabilities
with datasets ranging from small to massive scale (up to 10M samples).

Usage:
    python run_massive_scale_demo.py
"""

import subprocess
import sys
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_comparison(config_name, description):
    """Run a comparison configuration and time it."""
    logger.info(f"🚀 Starting {description}")
    logger.info(f"Configuration: {config_name}")
    
    start_time = time.time()
    
    try:
        # Run the comparison
        result = subprocess.run([
            sys.executable, "run_comparison.py", 
            "--config", config_name
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully in {elapsed_time:.1f} seconds")
            logger.info(f"Output preview:\n{result.stdout[-500:]}")  # Last 500 chars
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            logger.error(f"Error output:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} timed out after 1 hour")
    except Exception as e:
        logger.error(f"💥 {description} failed with exception: {e}")
    
    logger.info("-" * 60)

def main():
    """Run a series of comparisons to demonstrate scalability."""
    
    logger.info("=" * 80)
    logger.info("🔬 GENTA AUTOML MASSIVE SCALE COMPARISON DEMO")
    logger.info("=" * 80)
    logger.info("This demo will run comparisons across different dataset sizes")
    logger.info("to demonstrate the scalability and performance characteristics")
    logger.info("of Genta AutoML vs Standard ML approaches.")
    logger.info("")
    
    # Define the comparison sequence
    comparisons = [
        ("quick_comparison", "Quick Test on Small Datasets"),
        ("scalability_test", "Original Scalability Test"),
        ("large_scale_test", "Large Scale Test (up to 100K samples)"),
        ("massive_scale_test", "Massive Scale Test (up to 10M samples)"),
        ("full_scalability_comparison", "Complete Scalability Analysis")
    ]
    
    total_start = time.time()
    
    for config_name, description in comparisons:
        run_comparison(config_name, description)
        
        # Brief pause between comparisons
        time.sleep(2)
    
    total_elapsed = time.time() - total_start
    
    logger.info("=" * 80)
    logger.info("🏁 MASSIVE SCALE DEMO COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total demonstration time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    logger.info("")
    logger.info("📊 Check the generated reports in the comparison_results_* directories")
    logger.info("📈 Look for:")
    logger.info("   - comparison_charts_*.png (Performance visualizations)")
    logger.info("   - scalability_analysis_*.png (Scalability analysis)")
    logger.info("   - comparison_report_*.html (Detailed HTML reports)")
    logger.info("   - comparison_results_*.json (Raw results data)")
    logger.info("")
    logger.info("🎯 Key metrics to analyze:")
    logger.info("   - Training time scaling with dataset size")
    logger.info("   - Memory usage efficiency")
    logger.info("   - Model accuracy consistency")
    logger.info("   - Performance improvements at different scales")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
