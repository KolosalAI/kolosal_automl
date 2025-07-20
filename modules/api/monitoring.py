"""
Advanced Monitoring and Analytics Module for kolosal AutoML

Provides comprehensive monitoring including:
- Real-time performance metrics
- Resource utilization tracking
- Error analytics and reporting
- Custom dashboards
- Alerting system

Author: AI Assistant
Date: 2025-07-20
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import asyncio
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import numpy as np
except ImportError:
    np = None


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """Alert definition"""
    name: str
    condition: str
    threshold: float
    level: AlertLevel
    message: str
    cooldown_seconds: int = 300
    last_triggered: Optional[float] = None


class MetricsCollector:
    """Collects and manages metrics data"""
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_types: Dict[str, MetricType] = {}
        self.labels: Dict[str, Dict[str, str]] = {}
        self.lock = threading.RLock()
        
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric"""
        with self.lock:
            self.metric_types[name] = MetricType.COUNTER
            self.labels[name] = labels or {}
            self.metrics[name].append(MetricPoint(time.time(), value, labels))
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric"""
        with self.lock:
            self.metric_types[name] = MetricType.GAUGE
            self.labels[name] = labels or {}
            self.metrics[name].append(MetricPoint(time.time(), value, labels))
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric"""
        with self.lock:
            self.metric_types[name] = MetricType.HISTOGRAM
            self.labels[name] = labels or {}
            self.metrics[name].append(MetricPoint(time.time(), value, labels))
    
    def get_metric_data(self, name: str, since: Optional[float] = None) -> List[MetricPoint]:
        """Get metric data points"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            points = list(self.metrics[name])
            if since is not None:
                points = [p for p in points if p.timestamp >= since]
            
            return points
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            return self.metrics[name][-1].value
    
    def calculate_rate(self, name: str, window_seconds: float = 60.0) -> float:
        """Calculate rate for counter metrics"""
        with self.lock:
            if name not in self.metrics:
                return 0.0
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            points = [p for p in self.metrics[name] if p.timestamp >= cutoff_time]
            if len(points) < 2:
                return 0.0
            
            total_value = sum(p.value for p in points)
            return total_value / window_seconds
    
    def calculate_percentiles(self, name: str, percentiles: List[float], 
                            window_seconds: float = 300.0) -> Dict[float, float]:
        """Calculate percentiles for histogram metrics"""
        if not np:
            return {}
        
        with self.lock:
            if name not in self.metrics:
                return {}
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            values = [p.value for p in self.metrics[name] if p.timestamp >= cutoff_time]
            if not values:
                return {}
            
            return {p: np.percentile(values, p) for p in percentiles}
    
    def get_summary_stats(self, name: str, window_seconds: float = 300.0) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            values = [p.value for p in self.metrics[name] if p.timestamp >= cutoff_time]
            if not values:
                return {}
            
            stats = {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values)
            }
            
            if np:
                stats.update({
                    "std": float(np.std(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99))
                })
            
            return stats


class SystemMonitor:
    """Monitors system resources"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        self.interval = 10.0  # seconds
        
    def start_monitoring(self, interval: float = 10.0):
        """Start system monitoring"""
        self.interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        if not psutil:
            return
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record_gauge("system_cpu_usage_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.record_gauge("system_memory_usage_percent", memory.percent)
        self.metrics.record_gauge("system_memory_available_bytes", memory.available)
        self.metrics.record_gauge("system_memory_total_bytes", memory.total)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.record_gauge("system_disk_usage_percent", 
                                 (disk.used / disk.total) * 100)
        self.metrics.record_gauge("system_disk_free_bytes", disk.free)
        
        # Network metrics
        try:
            network = psutil.net_io_counters()
            self.metrics.record_counter("system_network_bytes_sent", network.bytes_sent)
            self.metrics.record_counter("system_network_bytes_recv", network.bytes_recv)
        except:
            pass  # Network metrics might not be available in some environments


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        self.active_alerts: Dict[str, Alert] = {}
        
    def add_alert(self, alert: Alert):
        """Add an alert rule"""
        self.alerts.append(alert)
    
    def add_alert_handler(self, handler: Callable[[Alert, float], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self):
        """Check all alert conditions"""
        current_time = time.time()
        
        for alert in self.alerts:
            try:
                should_trigger = self._evaluate_alert_condition(alert)
                
                if should_trigger:
                    # Check cooldown
                    if (alert.last_triggered is None or 
                        current_time - alert.last_triggered > alert.cooldown_seconds):
                        
                        alert.last_triggered = current_time
                        self.active_alerts[alert.name] = alert
                        
                        # Trigger alert handlers
                        for handler in self.alert_handlers:
                            try:
                                handler(alert, current_time)
                            except Exception as e:
                                logging.error(f"Alert handler error: {e}")
                else:
                    # Clear active alert if condition no longer met
                    if alert.name in self.active_alerts:
                        del self.active_alerts[alert.name]
                        
            except Exception as e:
                logging.error(f"Error evaluating alert {alert.name}: {e}")
    
    def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate an alert condition"""
        # Simple condition evaluation - can be extended for complex expressions
        if ">" in alert.condition:
            metric_name, threshold_str = alert.condition.split(">")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())
            
            current_value = self.metrics.get_latest_value(metric_name)
            return current_value is not None and current_value > threshold
        
        elif "<" in alert.condition:
            metric_name, threshold_str = alert.condition.split("<")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())
            
            current_value = self.metrics.get_latest_value(metric_name)
            return current_value is not None and current_value < threshold
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())


class PerformanceAnalyzer:
    """Analyzes performance trends and patterns"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        
    def analyze_throughput_trends(self, window_hours: int = 24) -> Dict[str, Any]:
        """Analyze throughput trends"""
        window_seconds = window_hours * 3600
        current_time = time.time()
        
        # Get request rate data
        request_rate = self.metrics.calculate_rate("api_requests_total", window_seconds)
        
        # Get error rate
        error_count = self.metrics.get_latest_value("api_errors_total") or 0
        total_requests = self.metrics.get_latest_value("api_requests_total") or 1
        error_rate = (error_count / total_requests) * 100
        
        # Get processing time stats
        processing_stats = self.metrics.get_summary_stats("request_processing_time", window_seconds)
        
        return {
            "request_rate_per_second": request_rate,
            "error_rate_percent": error_rate,
            "processing_time_stats": processing_stats,
            "window_hours": window_hours,
            "analysis_timestamp": current_time
        }
    
    def analyze_resource_utilization(self, window_hours: int = 6) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        window_seconds = window_hours * 3600
        
        cpu_stats = self.metrics.get_summary_stats("system_cpu_usage_percent", window_seconds)
        memory_stats = self.metrics.get_summary_stats("system_memory_usage_percent", window_seconds)
        
        # Detect resource stress patterns
        cpu_stress = cpu_stats.get("avg", 0) > 80 if cpu_stats else False
        memory_stress = memory_stats.get("avg", 0) > 85 if memory_stats else False
        
        return {
            "cpu_utilization": cpu_stats,
            "memory_utilization": memory_stats,
            "cpu_stress_detected": cpu_stress,
            "memory_stress_detected": memory_stress,
            "recommendations": self._generate_optimization_recommendations(cpu_stats, memory_stats)
        }
    
    def _generate_optimization_recommendations(self, cpu_stats: Dict, memory_stats: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if cpu_stats and cpu_stats.get("avg", 0) > 80:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        if memory_stats and memory_stats.get("avg", 0) > 85:
            recommendations.append("Consider increasing memory allocation or optimizing memory usage")
        
        if cpu_stats and cpu_stats.get("p95", 0) > 95:
            recommendations.append("CPU usage spikes detected - investigate batch size optimization")
        
        return recommendations


class MonitoringDashboard:
    """Creates monitoring dashboards and reports"""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 performance_analyzer: PerformanceAnalyzer,
                 alert_manager: AlertManager):
        self.metrics = metrics_collector
        self.analyzer = performance_analyzer
        self.alerts = alert_manager
        
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data"""
        current_time = time.time()
        
        # Basic metrics
        dashboard_data = {
            "timestamp": current_time,
            "current_metrics": {
                "cpu_usage": self.metrics.get_latest_value("system_cpu_usage_percent"),
                "memory_usage": self.metrics.get_latest_value("system_memory_usage_percent"),
                "request_rate": self.metrics.calculate_rate("api_requests_total", 60),
                "error_rate": self._calculate_error_rate(),
                "active_connections": self.metrics.get_latest_value("active_connections") or 0
            },
            
            # Performance analysis
            "performance_analysis": self.analyzer.analyze_throughput_trends(24),
            "resource_analysis": self.analyzer.analyze_resource_utilization(6),
            
            # Alerts
            "active_alerts": [asdict(alert) for alert in self.alerts.get_active_alerts()],
            "alert_count_by_level": self._count_alerts_by_level(),
            
            # Historical data (last 24 hours)
            "historical_data": self._get_historical_dashboard_data()
        }
        
        return dashboard_data
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        errors = self.metrics.get_latest_value("api_errors_total") or 0
        requests = self.metrics.get_latest_value("api_requests_total") or 1
        return (errors / requests) * 100
    
    def _count_alerts_by_level(self) -> Dict[str, int]:
        """Count active alerts by level"""
        count_by_level = {level.value: 0 for level in AlertLevel}
        
        for alert in self.alerts.get_active_alerts():
            count_by_level[alert.level.value] += 1
        
        return count_by_level
    
    def _get_historical_dashboard_data(self) -> Dict[str, List]:
        """Get historical data for dashboard charts"""
        window_seconds = 24 * 3600  # 24 hours
        
        # Sample data points every 5 minutes
        sample_interval = 300  # 5 minutes
        current_time = time.time()
        start_time = current_time - window_seconds
        
        timestamps = []
        cpu_data = []
        memory_data = []
        request_rates = []
        
        sample_time = start_time
        while sample_time <= current_time:
            timestamps.append(sample_time)
            
            # Get closest metric values to sample time
            cpu_data.append(self._get_metric_value_at_time("system_cpu_usage_percent", sample_time))
            memory_data.append(self._get_metric_value_at_time("system_memory_usage_percent", sample_time))
            request_rates.append(self._get_request_rate_at_time(sample_time))
            
            sample_time += sample_interval
        
        return {
            "timestamps": timestamps,
            "cpu_usage": cpu_data,
            "memory_usage": memory_data,
            "request_rates": request_rates
        }
    
    def _get_metric_value_at_time(self, metric_name: str, target_time: float) -> Optional[float]:
        """Get metric value closest to target time"""
        points = self.metrics.get_metric_data(metric_name)
        if not points:
            return None
        
        # Find closest point to target time
        closest_point = min(points, key=lambda p: abs(p.timestamp - target_time))
        return closest_point.value
    
    def _get_request_rate_at_time(self, target_time: float) -> float:
        """Get request rate at specific time"""
        # Calculate rate in 5-minute window around target time
        return self.metrics.calculate_rate("api_requests_total", 300)


class MonitoringManager:
    """Main monitoring manager that coordinates all monitoring components"""
    
    def __init__(self, enable_system_monitoring: bool = True):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)
        self.dashboard = MonitoringDashboard(
            self.metrics_collector,
            self.performance_analyzer,
            self.alert_manager
        )
        
        self.enable_system_monitoring = enable_system_monitoring
        self.alert_check_interval = 30.0  # seconds
        self.alert_thread = None
        self.running = False
        
        # Set up default alerts
        self._setup_default_alerts()
        
        # Set up default alert handlers
        self._setup_default_alert_handlers()
    
    def start(self):
        """Start monitoring"""
        self.running = True
        
        if self.enable_system_monitoring:
            self.system_monitor.start_monitoring()
        
        # Start alert checking thread
        self.alert_thread = threading.Thread(target=self._alert_check_loop, daemon=True)
        self.alert_thread.start()
        
        logging.info("Monitoring system started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        
        if self.enable_system_monitoring:
            self.system_monitor.stop_monitoring()
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        
        logging.info("Monitoring system stopped")
    
    def _alert_check_loop(self):
        """Alert checking loop"""
        while self.running:
            try:
                self.alert_manager.check_alerts()
                time.sleep(self.alert_check_interval)
            except Exception as e:
                logging.error(f"Error in alert checking: {e}")
                time.sleep(self.alert_check_interval)
    
    def _setup_default_alerts(self):
        """Set up default alert rules"""
        default_alerts = [
            Alert(
                name="high_cpu_usage",
                condition="system_cpu_usage_percent > 90",
                threshold=90,
                level=AlertLevel.WARNING,
                message="High CPU usage detected",
                cooldown_seconds=300
            ),
            Alert(
                name="high_memory_usage",
                condition="system_memory_usage_percent > 90",
                threshold=90,
                level=AlertLevel.WARNING,
                message="High memory usage detected",
                cooldown_seconds=300
            ),
            Alert(
                name="high_error_rate",
                condition="api_error_rate_percent > 5",
                threshold=5,
                level=AlertLevel.ERROR,
                message="High error rate detected",
                cooldown_seconds=180
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert(alert)
    
    def _setup_default_alert_handlers(self):
        """Set up default alert handlers"""
        def log_alert_handler(alert: Alert, timestamp: float):
            """Log alerts to file"""
            alert_log = logging.getLogger("kolosal_alerts")
            alert_log.error(
                f"ALERT: {alert.name} - {alert.message} "
                f"(Level: {alert.level.value}, Time: {datetime.fromtimestamp(timestamp)})"
            )
        
        self.alert_manager.add_alert_handler(log_alert_handler)
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, 
                          processing_time: float):
        """Record API request metrics"""
        labels = {"endpoint": endpoint, "method": method, "status": str(status_code)}
        
        self.metrics_collector.record_counter("api_requests_total", 1.0, labels)
        self.metrics_collector.record_histogram("request_processing_time", processing_time, labels)
        
        if status_code >= 400:
            self.metrics_collector.record_counter("api_errors_total", 1.0, labels)
    
    def record_batch_metrics(self, batch_size: int, processing_time: float, 
                           queue_size: int, success: bool):
        """Record batch processing metrics"""
        self.metrics_collector.record_histogram("batch_size", batch_size)
        self.metrics_collector.record_histogram("batch_processing_time", processing_time)
        self.metrics_collector.record_gauge("batch_queue_size", queue_size)
        
        if success:
            self.metrics_collector.record_counter("batch_success_total", 1.0)
        else:
            self.metrics_collector.record_counter("batch_error_total", 1.0)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return self.dashboard.generate_dashboard_data()
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        # This would export metrics in Prometheus format
        # Implementation depends on specific Prometheus client library
        pass


# Default monitoring instance
default_monitoring = MonitoringManager()
