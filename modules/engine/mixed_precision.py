"""
Mixed Precision Training and Inference Support
Provides mixed precision capabilities for faster training and inference with reduced memory usage.
"""

import logging
import time
import functools
from typing import Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import threading
from contextlib import contextmanager

# PyTorch AMP support
try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# TensorFlow mixed precision support
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Scikit-learn with reduced precision support
try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class MixedPrecisionStats:
    """Statistics for mixed precision operations."""
    total_operations: int
    fp16_operations: int
    fp32_operations: int
    memory_saved_bytes: int
    speedup_factor: float
    accuracy_loss: float


class MixedPrecisionConfig:
    """Configuration for mixed precision operations."""
    
    def __init__(self,
                 enable_fp16: bool = True,
                 enable_automatic_scaling: bool = True,
                 loss_scale_factor: float = 65536.0,
                 min_loss_scale: float = 1.0,
                 max_loss_scale: float = 65536.0,
                 scale_growth_factor: float = 2.0,
                 scale_backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        Initialize mixed precision configuration.
        
        Args:
            enable_fp16: Whether to use FP16 precision
            enable_automatic_scaling: Whether to use automatic loss scaling
            loss_scale_factor: Initial loss scaling factor
            min_loss_scale: Minimum loss scale value
            max_loss_scale: Maximum loss scale value
            scale_growth_factor: Factor to grow loss scale
            scale_backoff_factor: Factor to reduce loss scale on overflow
            growth_interval: Interval to attempt scale growth
        """
        self.enable_fp16 = enable_fp16
        self.enable_automatic_scaling = enable_automatic_scaling
        self.loss_scale_factor = loss_scale_factor
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self.scale_growth_factor = scale_growth_factor
        self.scale_backoff_factor = scale_backoff_factor
        self.growth_interval = growth_interval


class MixedPrecisionManager:
    """
    Manager for mixed precision operations across different ML frameworks.
    
    Features:
    - Automatic precision selection based on hardware capabilities
    - Framework-agnostic mixed precision support
    - Performance monitoring and adaptive precision
    - Memory usage optimization
    - Fallback to full precision if needed
    """
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        """
        Initialize the mixed precision manager.
        
        Args:
            config: Mixed precision configuration
        """
        self.config = config or MixedPrecisionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Hardware capability detection
        self.supports_fp16 = self._detect_fp16_support()
        self.supports_tensor_cores = self._detect_tensor_core_support()
        
        # Framework-specific components
        self.torch_scaler = None
        self.tf_policy = None
        
        # Performance tracking
        self.stats = MixedPrecisionStats(
            total_operations=0,
            fp16_operations=0,
            fp32_operations=0,
            memory_saved_bytes=0,
            speedup_factor=1.0,
            accuracy_loss=0.0
        )
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize framework-specific components
        self._initialize_frameworks()
        
        self.logger.info(f"Mixed precision manager initialized. FP16 support: {self.supports_fp16}")
    
    def _detect_fp16_support(self) -> bool:
        """Detect if hardware supports FP16 operations efficiently."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Check for GPU with FP16 support
            try:
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                # Modern GPUs (compute capability >= 7.0) have good FP16 support
                return properties.major >= 7
            except:
                pass
        
        # Check CPU FP16 support (limited)
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx2' in cpu_info.get('flags', [])
        except:
            pass
        
        return False
    
    def _detect_tensor_core_support(self) -> bool:
        """Detect if hardware has Tensor Core support."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                # Tensor Cores available on V100 (7.0), T4 (7.5), A100 (8.0+)
                return properties.major >= 7
            except:
                pass
        return False
    
    def _initialize_frameworks(self):
        """Initialize framework-specific mixed precision components."""
        # PyTorch AMP
        if TORCH_AVAILABLE and self.supports_fp16:
            try:
                self.torch_scaler = GradScaler(
                    init_scale=self.config.loss_scale_factor,
                    growth_factor=self.config.scale_growth_factor,
                    backoff_factor=self.config.scale_backoff_factor,
                    growth_interval=self.config.growth_interval
                )
                self.logger.info("PyTorch AMP scaler initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PyTorch AMP: {str(e)}")
        
        # TensorFlow mixed precision
        if TF_AVAILABLE and self.supports_fp16:
            try:
                if hasattr(tf.keras.mixed_precision, 'Policy'):
                    self.tf_policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(self.tf_policy)
                    self.logger.info("TensorFlow mixed precision policy set")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorFlow mixed precision: {str(e)}")
    
    @contextmanager
    def autocast_context(self, framework: str = 'auto'):
        """
        Context manager for automatic mixed precision casting.
        
        Args:
            framework: Target framework ('torch', 'tf', or 'auto')
        """
        if not self.config.enable_fp16 or not self.supports_fp16:
            yield
            return
        
        if framework == 'auto':
            framework = self._detect_active_framework()
        
        try:
            if framework == 'torch' and TORCH_AVAILABLE and self.torch_scaler:
                with autocast():
                    yield
            elif framework == 'tf' and TF_AVAILABLE and self.tf_policy:
                # TensorFlow autocast is enabled globally via policy
                yield
            else:
                yield
        except Exception as e:
            self.logger.warning(f"Mixed precision autocast failed: {str(e)}")
            yield
    
    def _detect_active_framework(self) -> str:
        """Detect which ML framework is currently active."""
        # Simple heuristic based on available frameworks
        if TORCH_AVAILABLE:
            return 'torch'
        elif TF_AVAILABLE:
            return 'tf'
        else:
            return 'numpy'
    
    def optimize_numpy_precision(self, data: np.ndarray, 
                                 target_precision: str = 'auto') -> np.ndarray:
        """
        Optimize NumPy array precision for memory and performance.
        
        Args:
            data: Input NumPy array
            target_precision: Target precision ('fp16', 'fp32', 'auto')
            
        Returns:
            Optimized array with appropriate precision
        """
        if not self.config.enable_fp16:
            return data
        
        original_dtype = data.dtype
        original_size = data.nbytes
        
        try:
            if target_precision == 'auto':
                # Automatic precision selection based on data characteristics
                target_precision = self._select_optimal_precision(data)
            
            if target_precision == 'fp16' and self.supports_fp16:
                # Convert to FP16 if data range allows
                if np.all(np.abs(data) < 65504):  # FP16 max value
                    optimized_data = data.astype(np.float16)
                    
                    # Track memory savings
                    memory_saved = original_size - optimized_data.nbytes
                    with self.lock:
                        self.stats.memory_saved_bytes += memory_saved
                        self.stats.fp16_operations += 1
                    
                    return optimized_data
            
            # Fallback to original precision
            with self.lock:
                self.stats.fp32_operations += 1
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Precision optimization failed: {str(e)}")
            return data
    
    def _select_optimal_precision(self, data: np.ndarray) -> str:
        """Select optimal precision based on data characteristics."""
        # Check data range
        data_max = np.max(np.abs(data))
        
        # FP16 range check
        if data_max < 65504 and np.all(np.isfinite(data)):
            # Check if precision loss is acceptable
            if self._check_precision_loss_acceptable(data):
                return 'fp16'
        
        return 'fp32'
    
    def _check_precision_loss_acceptable(self, data: np.ndarray, 
                                       threshold: float = 1e-3) -> bool:
        """Check if FP16 conversion causes acceptable precision loss."""
        try:
            # Sample a subset for efficiency
            sample_size = min(1000, len(data.flat))
            sample_indices = np.random.choice(len(data.flat), sample_size, replace=False)
            sample_data = data.flat[sample_indices]
            
            # Convert to FP16 and back
            fp16_data = sample_data.astype(np.float16).astype(np.float32)
            
            # Calculate relative error
            relative_error = np.mean(np.abs(sample_data - fp16_data) / (np.abs(sample_data) + 1e-8))
            
            return relative_error < threshold
            
        except Exception:
            return False
    
    def scale_loss(self, loss: Union[float, np.ndarray], framework: str = 'auto') -> Union[float, np.ndarray]:
        """
        Scale loss for mixed precision training.
        
        Args:
            loss: Loss value to scale
            framework: Target framework
            
        Returns:
            Scaled loss value
        """
        if not self.config.enable_automatic_scaling:
            return loss
        
        if framework == 'auto':
            framework = self._detect_active_framework()
        
        try:
            if framework == 'torch' and TORCH_AVAILABLE and self.torch_scaler:
                if isinstance(loss, torch.Tensor):
                    return self.torch_scaler.scale(loss)
            
            # Manual scaling for other frameworks
            if isinstance(loss, np.ndarray):
                return loss * self.config.loss_scale_factor
            else:
                return loss * self.config.loss_scale_factor
                
        except Exception as e:
            self.logger.warning(f"Loss scaling failed: {str(e)}")
            return loss
    
    def unscale_gradients(self, gradients, framework: str = 'auto'):
        """
        Unscale gradients after backward pass.
        
        Args:
            gradients: Gradients to unscale
            framework: Target framework
        """
        if not self.config.enable_automatic_scaling:
            return gradients
        
        if framework == 'auto':
            framework = self._detect_active_framework()
        
        try:
            if framework == 'torch' and TORCH_AVAILABLE and self.torch_scaler:
                # PyTorch AMP handles this automatically
                return gradients
            
            # Manual unscaling for other frameworks
            if isinstance(gradients, np.ndarray):
                return gradients / self.config.loss_scale_factor
            else:
                return [g / self.config.loss_scale_factor for g in gradients]
                
        except Exception as e:
            self.logger.warning(f"Gradient unscaling failed: {str(e)}")
            return gradients
    
    def step_optimizer(self, optimizer, framework: str = 'auto'):
        """
        Step optimizer with mixed precision support.
        
        Args:
            optimizer: Optimizer to step
            framework: Target framework
        """
        if framework == 'auto':
            framework = self._detect_active_framework()
        
        try:
            if framework == 'torch' and TORCH_AVAILABLE and self.torch_scaler:
                self.torch_scaler.step(optimizer)
                self.torch_scaler.update()
            else:
                # Standard optimizer step
                if hasattr(optimizer, 'step'):
                    optimizer.step()
                    
        except Exception as e:
            self.logger.warning(f"Optimizer step failed: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get mixed precision performance statistics."""
        with self.lock:
            total_ops = self.stats.total_operations
            if total_ops == 0:
                return {}
            
            return {
                'total_operations': total_ops,
                'fp16_operations': self.stats.fp16_operations,
                'fp32_operations': self.stats.fp32_operations,
                'fp16_usage_ratio': self.stats.fp16_operations / total_ops,
                'memory_saved_mb': self.stats.memory_saved_bytes / (1024 * 1024),
                'estimated_speedup': self.stats.speedup_factor,
                'accuracy_loss': self.stats.accuracy_loss,
                'hardware_support': {
                    'fp16_support': self.supports_fp16,
                    'tensor_cores': self.supports_tensor_cores
                }
            }
    
    def benchmark_precision_modes(self, operation_func: Callable, 
                                 data: np.ndarray, 
                                 iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark different precision modes for a given operation.
        
        Args:
            operation_func: Function to benchmark
            data: Test data
            iterations: Number of iterations for benchmarking
            
        Returns:
            Benchmark results for different precision modes
        """
        results = {}
        
        # Benchmark FP32
        fp32_data = data.astype(np.float32)
        start_time = time.perf_counter()
        for _ in range(iterations):
            operation_func(fp32_data)
        fp32_time = time.perf_counter() - start_time
        results['fp32_time'] = fp32_time
        
        # Benchmark FP16 if supported
        if self.supports_fp16:
            fp16_data = data.astype(np.float16)
            start_time = time.perf_counter()
            for _ in range(iterations):
                operation_func(fp16_data)
            fp16_time = time.perf_counter() - start_time
            results['fp16_time'] = fp16_time
            results['speedup'] = fp32_time / fp16_time if fp16_time > 0 else 1.0
        else:
            results['fp16_time'] = None
            results['speedup'] = 1.0
        
        return results
    
    def enable_mixed_precision(self):
        """Enable mixed precision mode."""
        self.config.enable_fp16 = True and self.supports_fp16
        self.logger.info("Mixed precision enabled")
    
    def disable_mixed_precision(self):
        """Disable mixed precision mode."""
        self.config.enable_fp16 = False
        self.logger.info("Mixed precision disabled")


# Decorators for automatic mixed precision
def mixed_precision(config: Optional[MixedPrecisionConfig] = None,
                   framework: str = 'auto'):
    """
    Decorator to automatically apply mixed precision to functions.
    
    Args:
        config: Mixed precision configuration
        framework: Target framework
    """
    manager = MixedPrecisionManager(config)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with manager.autocast_context(framework):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global mixed precision manager
_global_mixed_precision_manager = None

def get_global_mixed_precision_manager() -> MixedPrecisionManager:
    """Get or create the global mixed precision manager."""
    global _global_mixed_precision_manager
    if _global_mixed_precision_manager is None:
        _global_mixed_precision_manager = MixedPrecisionManager()
    return _global_mixed_precision_manager
