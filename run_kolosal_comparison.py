#!/usr/bin/env python3
"""
Kolosal AutoML vs Standard ML Comparison Runner

This script runs comprehensive comparisons between Kolosal AutoML and standard ML approaches.
It integrates both benchmark scripts and provides unified reporting.

Usage:
    python run_kolosal_comparison.py --mode quick
    python run_kolosal_comparison.py --mode comprehensive --datasets iris wine --models random_forest
    python run_kolosal_comparison.py --mode custom --config custom_config.json
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KolosalComparisonRunner")

class ComparisonRunner:
    """Main runner for Kolosal AutoML vs Standard ML comparison."""
    
    def __init__(self, output_dir: str = "./comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define benchmark script paths
        self.project_root = Path(__file__).parent
        self.standard_ml_script = self.project_root / "benchmark" / "standard_ml_benchmark.py"
        self.kolosal_automl_script = self.project_root / "benchmark" / "kolosal_automl_benchmark.py"
        
        logger.info(f"Comparison runner initialized. Results will be saved to: {self.output_dir}")
    
    def get_predefined_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined comparison configurations."""
        return {
            "quick": {
                "datasets": ["iris", "wine"],
                "models": ["random_forest", "logistic_regression"],
                "optimization": "random_search",
                "description": "Quick comparison on small datasets"
            },
            "comprehensive": {
                "datasets": [
                    "iris", "wine", "breast_cancer", "diabetes",
                    "synthetic_small_classification", "synthetic_small_regression"
                ],
                "models": ["random_forest", "gradient_boosting", "logistic_regression"],
                "optimization": "random_search",
                "description": "Comprehensive comparison across multiple datasets and models"
            },
            "optimization_strategies": {
                "datasets": ["breast_cancer", "diabetes"],
                "models": ["random_forest"],
                "optimization": ["random_search", "grid_search", "bayesian_optimization", "asht"],
                "description": "Compare different optimization strategies"
            },
            "scalability": {
                "datasets": [
                    "synthetic_small_classification", "synthetic_medium_classification", 
                    "synthetic_small_regression", "synthetic_medium_regression"
                ],
                "models": ["random_forest", "gradient_boosting"],
                "optimization": "random_search",
                "description": "Scalability testing across different dataset sizes"
            },
            "large_scale": {
                "datasets": [
                    "synthetic_medium_classification", "synthetic_large_classification",
                    "synthetic_medium_regression", "synthetic_large_regression"
                ],
                "models": ["random_forest"],
                "optimization": "random_search",
                "description": "Large scale performance testing"
            }
        }
    
    def run_standard_ml_benchmark(self, datasets: List[str], models: List[str], 
                                 optimization: str) -> Optional[str]:
        """Run standard ML benchmark."""
        logger.info("Running Standard ML benchmark...")
        
        standard_output_dir = self.output_dir / "standard_ml"
        standard_output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, str(self.standard_ml_script),
            "--output-dir", str(standard_output_dir),
            "--datasets"] + datasets + [
            "--models"] + models + [
            "--optimization", optimization
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("Standard ML benchmark completed successfully")
                return str(standard_output_dir)
            else:
                logger.error(f"Standard ML benchmark failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("Standard ML benchmark timed out")
            return None
        except Exception as e:
            logger.error(f"Error running Standard ML benchmark: {e}")
            return None
    
    def run_kolosal_automl_benchmark(self, datasets: List[str], models: List[str], 
                                    optimization: str) -> Optional[str]:
        """Run Kolosal AutoML benchmark."""
        logger.info("Running Kolosal AutoML benchmark...")
        
        kolosal_output_dir = self.output_dir / "kolosal_automl"
        kolosal_output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, str(self.kolosal_automl_script),
            "--output-dir", str(kolosal_output_dir),
            "--datasets"] + datasets + [
            "--models"] + models + [
            "--optimization", optimization
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info("Kolosal AutoML benchmark completed successfully")
                return str(kolosal_output_dir)
            else:
                logger.error(f"Kolosal AutoML benchmark failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("Kolosal AutoML benchmark timed out")
            return None
        except Exception as e:
            logger.error(f"Error running Kolosal AutoML benchmark: {e}")
            return None
    
    def load_benchmark_results(self, results_dir: str) -> Optional[Dict[str, Any]]:
        """Load benchmark results from directory."""
        results_path = Path(results_dir)
        
        # Find the most recent results file
        result_files = list(results_path.glob("*_results_*.json"))
        if not result_files:
            logger.warning(f"No result files found in {results_dir}")
            return None
        
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results from {latest_file}: {e}")
            return None
    
    def generate_comparison_report(self, standard_results: Dict[str, Any], 
                                 kolosal_results: Dict[str, Any]) -> str:
        """Generate comprehensive comparison report."""
        report_file = self.output_dir / f"comparison_report_{self.timestamp}.html"
        
        # Extract successful results
        std_successful = [r for r in standard_results.get('results', []) if r.get('success', False)]
        kolosal_successful = [r for r in kolosal_results.get('results', []) if r.get('success', False)]
        
        if not std_successful or not kolosal_successful:
            logger.warning("Insufficient successful results for comparison")
            return ""
        
        # Calculate summary statistics
        def calc_avg(results, metric):
            values = [r.get(metric, 0) for r in results]
            return sum(values) / len(values) if values else 0
        
        std_avg_time = calc_avg(std_successful, 'training_time')
        kolosal_avg_time = calc_avg(kolosal_successful, 'training_time')
        
        std_avg_score = calc_avg(std_successful, 'test_score')
        kolosal_avg_score = calc_avg(kolosal_successful, 'test_score')
        
        std_avg_memory = calc_avg(std_successful, 'memory_peak_mb')
        kolosal_avg_memory = calc_avg(kolosal_successful, 'memory_peak_mb')
        
        # Calculate improvements
        speed_improvement = ((std_avg_time / kolosal_avg_time) - 1) * 100 if kolosal_avg_time > 0 else 0
        accuracy_improvement = ((kolosal_avg_score / std_avg_score) - 1) * 100 if std_avg_score > 0 else 0
        memory_improvement = ((std_avg_memory / kolosal_avg_memory) - 1) * 100 if kolosal_avg_memory > 0 else 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kolosal AutoML vs Standard ML Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .improvement {{ color: #27ae60; font-weight: bold; }}
                .degradation {{ color: #e74c3c; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 20px; background-color: white; border-radius: 10px; text-align: center; min-width: 200px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }}
                .comparison-highlight {{ background: linear-gradient(45deg, #f39c12, #e67e22); color: white; padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Kolosal AutoML vs Standard ML Comparison Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Fair Comparison Analysis</p>
            </div>
            
            <div class="summary">
                <h2>📊 Executive Summary</h2>
                <p>This report compares the performance of <strong>Kolosal AutoML</strong> (with optimizations disabled for fair comparison) against <strong>standard scikit-learn approaches</strong>.</p>
                
                <div class="comparison-highlight">
                    <h3>🏆 Key Performance Highlights</h3>
                    <p>Based on {len(std_successful)} standard ML runs vs {len(kolosal_successful)} Kolosal AutoML runs</p>
                </div>
                
                <div style="text-align: center; margin: 20px 0;">
                    <div class="metric-box">
                        <div class="metric-label">Training Speed</div>
                        <div class="metric-value {'improvement' if speed_improvement > 0 else 'degradation'}">
                            {speed_improvement:+.1f}%
                        </div>
                        <p>{'Faster' if speed_improvement > 0 else 'Slower'} than standard ML</p>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value {'improvement' if accuracy_improvement > 0 else 'degradation'}">
                            {accuracy_improvement:+.1f}%
                        </div>
                        <p>{'Better' if accuracy_improvement > 0 else 'Lower'} accuracy</p>
                    </div>
                    
                    <div class="metric-box">
                        <div class="metric-label">Memory Efficiency</div>
                        <div class="metric-value {'improvement' if memory_improvement > 0 else 'degradation'}">
                            {memory_improvement:+.1f}%
                        </div>
                        <p>{'Less' if memory_improvement > 0 else 'More'} memory usage</p>
                    </div>
                </div>
            </div>
            
            <h2>📋 Detailed Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Standard ML</th>
                    <th>Kolosal AutoML</th>
                    <th>Improvement</th>
                </tr>
                <tr>
                    <td>Average Training Time</td>
                    <td>{std_avg_time:.2f} seconds</td>
                    <td>{kolosal_avg_time:.2f} seconds</td>
                    <td class="{'improvement' if speed_improvement > 0 else 'degradation'}">{speed_improvement:+.1f}%</td>
                </tr>
                <tr>
                    <td>Average Test Score</td>
                    <td>{std_avg_score:.4f}</td>
                    <td>{kolosal_avg_score:.4f}</td>
                    <td class="{'improvement' if accuracy_improvement > 0 else 'degradation'}">{accuracy_improvement:+.1f}%</td>
                </tr>
                <tr>
                    <td>Average Memory Usage</td>
                    <td>{std_avg_memory:.1f} MB</td>
                    <td>{kolosal_avg_memory:.1f} MB</td>
                    <td class="{'improvement' if memory_improvement > 0 else 'degradation'}">{memory_improvement:+.1f}%</td>
                </tr>
            </table>
            
            <h2>📝 Individual Results</h2>
            <h3>Standard ML Results</h3>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Training Time (s)</th>
                    <th>Test Score</th>
                    <th>Memory (MB)</th>
                </tr>
        """
        
        # Add standard ML results
        for result in std_successful:
            html_content += f"""
                <tr>
                    <td>{result.get('dataset_name', 'N/A')}</td>
                    <td>{result.get('model_name', 'N/A')}</td>
                    <td>{result.get('training_time', 0):.2f}</td>
                    <td>{result.get('test_score', 0):.4f}</td>
                    <td>{result.get('memory_peak_mb', 0):.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h3>Kolosal AutoML Results</h3>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Optimization Strategy</th>
                    <th>Training Time (s)</th>
                    <th>Test Score</th>
                    <th>Memory (MB)</th>
                </tr>
        """
        
        # Add Kolosal AutoML results
        for result in kolosal_successful:
            html_content += f"""
                <tr>
                    <td>{result.get('dataset_name', 'N/A')}</td>
                    <td>{result.get('model_name', 'N/A')}</td>
                    <td>{result.get('optimization_strategy', 'N/A')}</td>
                    <td>{result.get('training_time', 0):.2f}</td>
                    <td>{result.get('test_score', 0):.4f}</td>
                    <td>{result.get('memory_peak_mb', 0):.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="summary">
                <h2>🔍 Analysis Summary</h2>
                <ul>
        """
        
        if speed_improvement > 0:
            html_content += f"<li>Kolosal AutoML is <strong>{speed_improvement:.1f}% faster</strong> than standard ML approaches on average.</li>"
        else:
            html_content += f"<li>Standard ML is <strong>{abs(speed_improvement):.1f}% faster</strong> than Kolosal AutoML on average.</li>"
            
        if accuracy_improvement > 0:
            html_content += f"<li>Kolosal AutoML achieves <strong>{accuracy_improvement:.1f}% better accuracy</strong> than standard ML approaches on average.</li>"
        else:
            html_content += f"<li>Standard ML achieves <strong>{abs(accuracy_improvement):.1f}% better accuracy</strong> than Kolosal AutoML on average.</li>"
            
        if memory_improvement > 0:
            html_content += f"<li>Kolosal AutoML uses <strong>{memory_improvement:.1f}% less memory</strong> than standard ML approaches on average.</li>"
        else:
            html_content += f"<li>Standard ML uses <strong>{abs(memory_improvement):.1f}% less memory</strong> than Kolosal AutoML on average.</li>"
        
        html_content += """
                </ul>
                <p><em>Note: This comparison was conducted with Kolosal AutoML optimizations disabled to ensure fair comparison against standard scikit-learn approaches.</em></p>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comparison report generated: {report_file}")
        return str(report_file)
    
    def run_comparison(self, config_name: str = None, datasets: List[str] = None, 
                      models: List[str] = None, optimization: str = "random_search") -> Dict[str, Any]:
        """Run full comparison between Standard ML and Kolosal AutoML."""
        start_time = time.time()
        
        # Get configuration
        if config_name:
            configs = self.get_predefined_configs()
            if config_name not in configs:
                raise ValueError(f"Unknown configuration: {config_name}")
            config = configs[config_name]
            datasets = config["datasets"]
            models = config["models"]
            optimization = config.get("optimization", "random_search")
            logger.info(f"Using predefined configuration: {config_name} - {config['description']}")
        else:
            if not datasets or not models:
                raise ValueError("Must provide datasets and models if not using predefined config")
        
        logger.info(f"Running comparison with datasets: {datasets}, models: {models}, optimization: {optimization}")
        
        # Run benchmarks
        standard_dir = self.run_standard_ml_benchmark(datasets, models, optimization)
        kolosal_dir = self.run_kolosal_automl_benchmark(datasets, models, optimization)
        
        # Check if both benchmarks completed successfully
        if not standard_dir:
            logger.error("Standard ML benchmark failed")
            return {"success": False, "error": "Standard ML benchmark failed"}
        
        if not kolosal_dir:
            logger.error("Kolosal AutoML benchmark failed")
            return {"success": False, "error": "Kolosal AutoML benchmark failed"}
        
        # Load results
        standard_results = self.load_benchmark_results(standard_dir)
        kolosal_results = self.load_benchmark_results(kolosal_dir)
        
        if not standard_results or not kolosal_results:
            logger.error("Failed to load benchmark results")
            return {"success": False, "error": "Failed to load benchmark results"}
        
        # Generate comparison report
        report_file = self.generate_comparison_report(standard_results, kolosal_results)
        
        total_time = time.time() - start_time
        
        # Save comparison summary
        summary = {
            "timestamp": self.timestamp,
            "configuration": config_name or "custom",
            "datasets": datasets,
            "models": models,
            "optimization": optimization,
            "total_time": total_time,
            "standard_ml_results": standard_results,
            "kolosal_automl_results": kolosal_results,
            "report_file": report_file,
            "success": True
        }
        
        summary_file = self.output_dir / f"comparison_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("COMPARISON COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Standard ML results: {standard_dir}")
        logger.info(f"Kolosal AutoML results: {kolosal_dir}")
        logger.info(f"Comparison report: {report_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info("=" * 60)
        
        return summary

def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description="Run Kolosal AutoML vs Standard ML Comparison")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "optimization_strategies", "scalability", "large_scale", "custom"], 
                       default="quick", help="Comparison mode")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", help="Custom datasets (for custom mode)")
    parser.add_argument("--models", nargs="+", help="Custom models (for custom mode)")
    parser.add_argument("--optimization", default="random_search", 
                       choices=["grid_search", "random_search", "bayesian_optimization", "asht"],
                       help="Optimization strategy (for custom mode)")
    parser.add_argument("--config", help="Path to custom configuration JSON file")
    
    args = parser.parse_args()
    
    # Create comparison runner
    runner = ComparisonRunner(args.output_dir)
    
    try:
        if args.mode == "custom":
            if args.config:
                # Load custom configuration
                with open(args.config, 'r') as f:
                    config = json.load(f)
                result = runner.run_comparison(
                    datasets=config.get("datasets"),
                    models=config.get("models"),
                    optimization=config.get("optimization", "random_search")
                )
            else:
                # Use command line arguments
                result = runner.run_comparison(
                    datasets=args.datasets,
                    models=args.models,
                    optimization=args.optimization
                )
        else:
            # Use predefined configuration
            result = runner.run_comparison(config_name=args.mode)
        
        if result["success"]:
            print(f"\n✅ Comparison completed successfully!")
            print(f"📊 Report available at: {result['report_file']}")
        else:
            print(f"\n❌ Comparison failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Comparison failed with error: {e}")
        print(f"\n❌ Comparison failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()