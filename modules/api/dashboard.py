"""
HTML Dashboard for kolosal AutoML Monitoring

Provides a simple web-based dashboard for monitoring system metrics,
performance analytics, and alerts.

Author: AI Assistant
Date: 2025-07-20
"""

from fastapi import Request
from fastapi.responses import HTMLResponse
from typing import Dict, Any
import json


def generate_dashboard_html(metrics_data: Dict[str, Any]) -> str:
    """Generate HTML dashboard content"""
    
    # Extract key metrics for display
    current_metrics = metrics_data.get("current_metrics", {})
    performance_analysis = metrics_data.get("performance_analysis", {})
    resource_analysis = metrics_data.get("resource_analysis", {})
    active_alerts = metrics_data.get("active_alerts", [])
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>kolosal AutoML - Monitoring Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .dashboard-container {{
                max-width: 1400px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
            }}
            
            .header h1 {{
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            
            .header p {{
                color: #7f8c8d;
                font-size: 1.1em;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .metric-card {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
                border: 1px solid #e8ecf0;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
            }}
            
            .metric-card h3 {{
                color: #2c3e50;
                font-size: 1.2em;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                font-weight: 600;
            }}
            
            .metric-card .icon {{
                width: 24px;
                height: 24px;
                margin-right: 10px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                color: white;
                font-weight: bold;
            }}
            
            .cpu-icon {{ background: #e74c3c; }}
            .memory-icon {{ background: #3498db; }}
            .requests-icon {{ background: #2ecc71; }}
            .errors-icon {{ background: #f39c12; }}
            .alerts-icon {{ background: #9b59b6; }}
            
            .metric-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }}
            
            .metric-unit {{
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            
            .metric-trend {{
                margin-top: 10px;
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 500;
            }}
            
            .trend-good {{ background: #d5f4e6; color: #27ae60; }}
            .trend-warning {{ background: #fef9e7; color: #f39c12; }}
            .trend-danger {{ background: #fdeaea; color: #e74c3c; }}
            
            .alerts-section {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            }}
            
            .alerts-section h2 {{
                color: #2c3e50;
                font-size: 1.5em;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
            }}
            
            .alert-item {{
                display: flex;
                align-items: center;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                border-left: 4px solid;
            }}
            
            .alert-info {{ border-left-color: #3498db; background: #ebf3fd; }}
            .alert-warning {{ border-left-color: #f39c12; background: #fef9e7; }}
            .alert-error {{ border-left-color: #e74c3c; background: #fdeaea; }}
            .alert-critical {{ border-left-color: #8e44ad; background: #f4ecf7; }}
            
            .alert-level {{
                font-weight: bold;
                text-transform: uppercase;
                font-size: 0.8em;
                padding: 4px 8px;
                border-radius: 4px;
                margin-right: 15px;
            }}
            
            .performance-section {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            }}
            
            .performance-section h2 {{
                color: #2c3e50;
                font-size: 1.5em;
                margin-bottom: 20px;
            }}
            
            .performance-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            
            .performance-stat {{
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
            }}
            
            .performance-stat .label {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-bottom: 5px;
            }}
            
            .performance-stat .value {{
                color: #2c3e50;
                font-size: 1.3em;
                font-weight: bold;
            }}
            
            .last-updated {{
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
            }}
            
            .refresh-btn {{
                position: fixed;
                bottom: 30px;
                right: 30px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 50px;
                padding: 15px 25px;
                font-size: 1em;
                cursor: pointer;
                box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
                transition: all 0.3s ease;
            }}
            
            .refresh-btn:hover {{
                background: #2980b9;
                transform: translateY(-2px);
                box-shadow: 0 7px 20px rgba(52, 152, 219, 0.4);
            }}
            
            @media (max-width: 768px) {{
                .dashboard-container {{
                    padding: 20px;
                }}
                
                .metrics-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .header h1 {{
                    font-size: 2em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header">
                <h1>🚀 kolosal AutoML</h1>
                <p>Real-time Monitoring Dashboard</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3><span class="icon cpu-icon">🔥</span>CPU Usage</h3>
                    <div class="metric-value">{current_metrics.get('cpu_usage', 0):.1f}</div>
                    <div class="metric-unit">%</div>
                    <div class="metric-trend {'trend-good' if current_metrics.get('cpu_usage', 0) < 70 else 'trend-warning' if current_metrics.get('cpu_usage', 0) < 90 else 'trend-danger'}">
                        {'Normal' if current_metrics.get('cpu_usage', 0) < 70 else 'High' if current_metrics.get('cpu_usage', 0) < 90 else 'Critical'}
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3><span class="icon memory-icon">💾</span>Memory Usage</h3>
                    <div class="metric-value">{current_metrics.get('memory_usage', 0):.1f}</div>
                    <div class="metric-unit">%</div>
                    <div class="metric-trend {'trend-good' if current_metrics.get('memory_usage', 0) < 80 else 'trend-warning' if current_metrics.get('memory_usage', 0) < 95 else 'trend-danger'}">
                        {'Normal' if current_metrics.get('memory_usage', 0) < 80 else 'High' if current_metrics.get('memory_usage', 0) < 95 else 'Critical'}
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3><span class="icon requests-icon">📊</span>Request Rate</h3>
                    <div class="metric-value">{current_metrics.get('request_rate', 0):.1f}</div>
                    <div class="metric-unit">req/sec</div>
                    <div class="metric-trend trend-good">Active</div>
                </div>
                
                <div class="metric-card">
                    <h3><span class="icon errors-icon">⚠️</span>Error Rate</h3>
                    <div class="metric-value">{current_metrics.get('error_rate', 0):.2f}</div>
                    <div class="metric-unit">%</div>
                    <div class="metric-trend {'trend-good' if current_metrics.get('error_rate', 0) < 1 else 'trend-warning' if current_metrics.get('error_rate', 0) < 5 else 'trend-danger'}">
                        {'Good' if current_metrics.get('error_rate', 0) < 1 else 'Elevated' if current_metrics.get('error_rate', 0) < 5 else 'High'}
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3><span class="icon alerts-icon">🔔</span>Active Alerts</h3>
                    <div class="metric-value">{len(active_alerts)}</div>
                    <div class="metric-unit">alerts</div>
                    <div class="metric-trend {'trend-good' if len(active_alerts) == 0 else 'trend-warning' if len(active_alerts) < 3 else 'trend-danger'}">
                        {'All Clear' if len(active_alerts) == 0 else 'Some Issues' if len(active_alerts) < 3 else 'Attention Needed'}
                    </div>
                </div>
            </div>
            
            {'<div class="alerts-section"><h2>🚨 Active Alerts</h2>' if active_alerts else ''}
            {''.join([f'''
                <div class="alert-item alert-{alert.get("level", "info")}">
                    <span class="alert-level">{alert.get("level", "info")}</span>
                    <div>
                        <strong>{alert.get("name", "Unknown Alert")}</strong><br>
                        {alert.get("message", "No message available")}
                    </div>
                </div>
            ''' for alert in active_alerts])}
            {'</div>' if active_alerts else ''}
            
            <div class="performance-section">
                <h2>📈 Performance Analytics</h2>
                <div class="performance-stats">
                    <div class="performance-stat">
                        <div class="label">Avg Response Time</div>
                        <div class="value">{performance_analysis.get('processing_time_stats', {}).get('avg', 0):.3f}s</div>
                    </div>
                    <div class="performance-stat">
                        <div class="label">P95 Response Time</div>
                        <div class="value">{performance_analysis.get('processing_time_stats', {}).get('p95', 0):.3f}s</div>
                    </div>
                    <div class="performance-stat">
                        <div class="label">Total Requests</div>
                        <div class="value">{performance_analysis.get('processing_time_stats', {}).get('count', 0)}</div>
                    </div>
                    <div class="performance-stat">
                        <div class="label">Min Response Time</div>
                        <div class="value">{performance_analysis.get('processing_time_stats', {}).get('min', 0):.3f}s</div>
                    </div>
                    <div class="performance-stat">
                        <div class="label">Max Response Time</div>
                        <div class="value">{performance_analysis.get('processing_time_stats', {}).get('max', 0):.3f}s</div>
                    </div>
                    <div class="performance-stat">
                        <div class="label">Request Rate</div>
                        <div class="value">{performance_analysis.get('request_rate_per_second', 0):.2f}/s</div>
                    </div>
                </div>
            </div>
            
            <div class="last-updated">
                Last updated: {metrics_data.get('timestamp', 'Unknown')}
            </div>
        </div>
        
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function() {{
                location.reload();
            }}, 30000);
            
            // Add some interactivity
            document.querySelectorAll('.metric-card').forEach(card => {{
                card.addEventListener('click', function() {{
                    this.style.transform = 'scale(0.98)';
                    setTimeout(() => {{
                        this.style.transform = '';
                    }}, 150);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content
