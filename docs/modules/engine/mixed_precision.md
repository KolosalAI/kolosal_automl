# Mixed Precision (`modules/engine/mixed_precision.py`)

## Overview

The Mixed Precision module provides mixed precision training and inference capabilities for faster computation and reduced memory usage. It supports multiple frameworks including PyTorch, TensorFlow, and scikit-learn with intelligent precision management.

## Features

- **Multi-Framework Support**: PyTorch AMP, TensorFlow mixed precision, scikit-learn optimization
- **Automatic Loss Scaling**: Dynamic loss scaling to prevent gradient underflow
- **Memory Optimization**: Significant memory reduction with minimal accuracy loss
- **Performance Monitoring**: Real-time tracking of speedup and accuracy metrics
- **Adaptive Precision**: Dynamic precision adjustment based on numerical stability
- **Gradient Clipping**: Advanced gradient management for stable training

## Core Classes

### MixedPrecisionManager

Main mixed precision manager with framework-agnostic interface:

```python
class MixedPrecisionManager:
    def __init__(
        self,
        framework: str = "auto",
        precision_policy: str = "mixed_float16",
        enable_loss_scaling: bool = True,
        initial_loss_scale: float = 2**15,
        enable_monitoring: bool = True
    )
```

**Parameters:**
- `framework`: Target framework ("pytorch", "tensorflow", "sklearn", "auto")
- `precision_policy`: Precision policy ("mixed_float16", "mixed_bfloat16", "float32")
- `enable_loss_scaling`: Enable automatic loss scaling
- `initial_loss_scale`: Initial scale factor for loss scaling
- `enable_monitoring`: Enable performance and accuracy monitoring

### MixedPrecisionConfig

Configuration class for mixed precision settings:

```python
@dataclass
class MixedPrecisionConfig:
    precision_policy: str = "mixed_float16"
    loss_scale: float = 2**15
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_growth_interval: int = 2000
    enable_autocast: bool = True
    autocast_device_type: str = "cuda"
    gradient_clipping_threshold: float = 1.0
    numerical_stability_check: bool = True
```

## Usage Examples

### PyTorch Mixed Precision Training

```python
from modules.engine.mixed_precision import MixedPrecisionManager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Initialize mixed precision manager
mp_manager = MixedPrecisionManager(
    framework="pytorch",
    precision_policy="mixed_float16",
    enable_loss_scaling=True,
    enable_monitoring=True
)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet(784, 512, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Prepare mixed precision training
model, optimizer, scaler = mp_manager.setup_pytorch_training(
    model=model,
    optimizer=optimizer
)

# Create sample data
X = torch.randn(10000, 784)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training loop with mixed precision
def train_with_mixed_precision(model, dataloader, optimizer, scaler, criterion, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass
            with mp_manager.autocast_context():
                output = model(data)
                loss = criterion(output, target)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping (optional)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        print(f'Epoch {epoch}: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
        
        # Log mixed precision statistics
        mp_stats = mp_manager.get_statistics()
        print(f'Memory saved: {mp_stats.memory_saved_bytes / 1024**2:.1f} MB, '
              f'Speedup: {mp_stats.speedup_factor:.2f}x')

# Run training
train_with_mixed_precision(model, dataloader, optimizer, scaler, criterion)

# Get final statistics
final_stats = mp_manager.get_comprehensive_stats()
print("Final Mixed Precision Statistics:")
print(f"  Total operations: {final_stats['total_operations']}")
print(f"  FP16 operations: {final_stats['fp16_operations']}")
print(f"  Memory reduction: {final_stats['memory_reduction']:.1%}")
print(f"  Average speedup: {final_stats['average_speedup']:.2f}x")
print(f"  Accuracy preservation: {final_stats['accuracy_preservation']:.3f}")
```

### TensorFlow Mixed Precision

```python
from modules.engine.mixed_precision import MixedPrecisionManager
import tensorflow as tf
import numpy as np

# Initialize TensorFlow mixed precision
tf_mp_manager = MixedPrecisionManager(
    framework="tensorflow",
    precision_policy="mixed_float16",
    enable_monitoring=True
)

# Set up TensorFlow mixed precision policy
tf_mp_manager.setup_tensorflow_policy()

# Create a model with mixed precision
def create_mixed_precision_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # Keep output in float32
    ])
    return model

# Create and compile model
model = create_mixed_precision_model()

# Use mixed precision optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf_mp_manager.wrap_optimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create sample data
X_train = np.random.randn(10000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, (10000,))
X_val = np.random.randn(2000, 784).astype(np.float32)
y_val = np.random.randint(0, 10, (2000,))

# Custom training loop with mixed precision monitoring
class MixedPrecisionCallback(tf.keras.callbacks.Callback):
    def __init__(self, mp_manager):
        self.mp_manager = mp_manager
        
    def on_epoch_end(self, epoch, logs=None):
        stats = self.mp_manager.get_statistics()
        print(f"Epoch {epoch}: Memory saved: {stats.memory_saved_bytes / 1024**2:.1f} MB")

# Train with mixed precision
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[MixedPrecisionCallback(tf_mp_manager)],
    verbose=1
)

# Evaluate mixed precision benefits
tf_stats = tf_mp_manager.get_comprehensive_stats()
print("TensorFlow Mixed Precision Results:")
print(f"  Training speedup: {tf_stats['training_speedup']:.2f}x")
print(f"  Memory usage reduction: {tf_stats['memory_reduction']:.1%}")
print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

### Scikit-learn Precision Optimization

```python
from modules.engine.mixed_precision import SklearnPrecisionOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize sklearn precision optimizer
sklearn_optimizer = SklearnPrecisionOptimizer(
    enable_float32_reduction=True,
    enable_int_downcasting=True,
    memory_threshold_mb=1024
)

# Generate sample data
X, y = make_classification(
    n_samples=50000,
    n_features=100,
    n_informative=80,
    n_redundant=10,
    n_classes=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Original data types: X={X_train.dtype}, y={y_train.dtype}")
print(f"Original memory usage: {X_train.nbytes / 1024**2:.1f} MB")

# Optimize data precision
X_train_opt, y_train_opt = sklearn_optimizer.optimize_data_precision(
    X_train, y_train
)
X_test_opt, y_test_opt = sklearn_optimizer.optimize_data_precision(
    X_test, y_test
)

print(f"Optimized data types: X={X_train_opt.dtype}, y={y_train_opt.dtype}")
print(f"Optimized memory usage: {X_train_opt.nbytes / 1024**2:.1f} MB")

memory_reduction = 1 - (X_train_opt.nbytes / X_train.nbytes)
print(f"Memory reduction: {memory_reduction:.1%}")

# Train models with both precisions for comparison
print("\nTraining with original precision...")
model_original = RandomForestClassifier(n_estimators=100, random_state=42)
import time
start_time = time.time()
model_original.fit(X_train, y_train)
original_training_time = time.time() - start_time
original_accuracy = model_original.score(X_test, y_test)

print("\nTraining with optimized precision...")
model_optimized = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
model_optimized.fit(X_train_opt, y_train_opt)
optimized_training_time = time.time() - start_time
optimized_accuracy = model_optimized.score(X_test_opt, y_test_opt)

# Performance comparison
speedup = original_training_time / optimized_training_time
accuracy_diff = abs(original_accuracy - optimized_accuracy)

print(f"\nPerformance Comparison:")
print(f"  Original accuracy: {original_accuracy:.4f}")
print(f"  Optimized accuracy: {optimized_accuracy:.4f}")
print(f"  Accuracy difference: {accuracy_diff:.4f}")
print(f"  Training speedup: {speedup:.2f}x")
print(f"  Memory reduction: {memory_reduction:.1%}")
```

## Advanced Mixed Precision Features

### Adaptive Precision Management

```python
from modules.engine.mixed_precision import AdaptivePrecisionManager
import torch
import torch.nn as nn

# Create adaptive precision manager
adaptive_manager = AdaptivePrecisionManager(
    stability_threshold=1e-4,
    adaptation_interval=100,
    enable_gradient_monitoring=True
)

class AdaptiveNet(nn.Module):
    def __init__(self):
        super(AdaptiveNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = AdaptiveNet().cuda()
optimizer = torch.optim.Adam(model.parameters())

# Training with adaptive precision
def adaptive_precision_training(model, data_loader, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            
            # Adaptive precision context
            with adaptive_manager.adaptive_context(
                model=model, 
                batch_idx=batch_idx
            ) as precision_context:
                
                optimizer.zero_grad()
                
                # Forward pass with adaptive precision
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                # Check numerical stability
                if precision_context.is_stable(loss):
                    # Backward pass with current precision
                    loss.backward()
                    optimizer.step()
                else:
                    # Fall back to higher precision
                    print(f"Numerical instability detected at batch {batch_idx}, using FP32")
                    precision_context.fallback_to_fp32()
                    
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=False):
                        output = model(data)
                        loss = nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Log adaptation decisions
            if batch_idx % 100 == 0:
                adaptation_stats = adaptive_manager.get_adaptation_stats()
                print(f"Batch {batch_idx}: FP16 ratio = {adaptation_stats['fp16_ratio']:.2%}")

# Run adaptive training
# adaptive_precision_training(model, dataloader)
```

### Custom Mixed Precision Policies

```python
from modules.engine.mixed_precision import CustomPrecisionPolicy

# Define custom precision policy
class ConservativeMixedPrecision(CustomPrecisionPolicy):
    """Conservative mixed precision policy with enhanced stability"""
    
    def __init__(self):
        super().__init__()
        self.layer_precision_map = {}
        
    def should_use_fp16(self, layer_type, layer_name):
        """Determine if layer should use FP16"""
        # Use FP32 for critical layers
        critical_layers = ['attention', 'embedding', 'output']
        if any(critical in layer_name.lower() for critical in critical_layers):
            return False
        
        # Use FP16 for compute-heavy layers
        compute_heavy = ['conv', 'linear', 'dense']
        if any(compute in layer_type.lower() for compute in compute_heavy):
            return True
        
        return False
    
    def get_loss_scale_factor(self, gradient_norm):
        """Dynamic loss scaling based on gradient norm"""
        if gradient_norm > 10.0:
            return 2**12  # Lower scale for large gradients
        elif gradient_norm < 0.1:
            return 2**18  # Higher scale for small gradients
        else:
            return 2**15  # Default scale

# Apply custom policy
custom_manager = MixedPrecisionManager(
    framework="pytorch",
    custom_policy=ConservativeMixedPrecision()
)

# Use custom policy in training
model, optimizer, scaler = custom_manager.setup_pytorch_training(
    model=model,
    optimizer=optimizer,
    use_custom_policy=True
)
```

### Mixed Precision Inference Optimization

```python
from modules.engine.mixed_precision import InferencePrecisionOptimizer
import torch

# Create inference precision optimizer
inference_optimizer = InferencePrecisionOptimizer(
    target_precision="int8",
    calibration_samples=1000,
    accuracy_threshold=0.02  # 2% accuracy drop tolerance
)

# Load pre-trained model
model = torch.load("pretrained_model.pth")

# Optimize model for inference
optimized_model = inference_optimizer.optimize_for_inference(
    model=model,
    calibration_data=calibration_loader,
    target_device="cuda"
)

# Benchmark inference performance
def benchmark_inference(model, test_data, name):
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for data, _ in test_data:
            data = data.cuda()
            output = model(data)
    
    inference_time = time.time() - start_time
    return inference_time

# Compare original vs optimized
original_time = benchmark_inference(model, test_loader, "Original")
optimized_time = benchmark_inference(optimized_model, test_loader, "Optimized")

speedup = original_time / optimized_time
print(f"Inference speedup: {speedup:.2f}x")

# Check memory usage
original_memory = torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
benchmark_inference(optimized_model, test_loader, "Optimized")
optimized_memory = torch.cuda.max_memory_allocated()

memory_reduction = 1 - (optimized_memory / original_memory)
print(f"Memory reduction: {memory_reduction:.1%}")
```

## Performance Monitoring and Analysis

### Comprehensive Statistics Tracking

```python
from modules.engine.mixed_precision import MixedPrecisionAnalyzer
import matplotlib.pyplot as plt

# Create analyzer
analyzer = MixedPrecisionAnalyzer(mp_manager)

# Run analysis during training
analyzer.start_analysis()

# Your training code here...
# train_model()

analyzer.stop_analysis()

# Generate comprehensive report
report = analyzer.generate_report()

print("Mixed Precision Analysis Report:")
print(f"  Operation breakdown:")
print(f"    FP16 operations: {report['fp16_ops_percentage']:.1%}")
print(f"    FP32 operations: {report['fp32_ops_percentage']:.1%}")
print(f"    Mixed operations: {report['mixed_ops_percentage']:.1%}")

print(f"  Performance metrics:")
print(f"    Average speedup: {report['average_speedup']:.2f}x")
print(f"    Peak speedup: {report['peak_speedup']:.2f}x")
print(f"    Memory saved: {report['memory_saved_gb']:.2f} GB")

print(f"  Numerical stability:")
print(f"    Overflow events: {report['overflow_events']}")
print(f"    Underflow events: {report['underflow_events']}")
print(f"    Scale adjustments: {report['scale_adjustments']}")

# Generate visualization plots
analyzer.plot_precision_timeline("precision_timeline.png")
analyzer.plot_speedup_distribution("speedup_distribution.png")
analyzer.plot_memory_usage("memory_usage.png")
analyzer.plot_numerical_stability("numerical_stability.png")

print("Analysis plots saved to disk")
```

## Best Practices

### 1. Gradual Mixed Precision Adoption

```python
# Start with conservative settings
conservative_config = MixedPrecisionConfig(
    precision_policy="mixed_float16",
    loss_scale=2**12,  # Lower initial scale
    gradient_clipping_threshold=0.5,  # Aggressive clipping
    numerical_stability_check=True
)

# Gradually increase optimization
def gradual_mixed_precision_training(model, epochs=20):
    """Gradually increase mixed precision optimization"""
    
    for epoch in range(epochs):
        # Adjust precision aggressiveness based on training stability
        if epoch < 5:
            # Conservative mode for initial epochs
            loss_scale = 2**12
            autocast_enabled = False
        elif epoch < 10:
            # Moderate mixed precision
            loss_scale = 2**14
            autocast_enabled = True
        else:
            # Full mixed precision optimization
            loss_scale = 2**16
            autocast_enabled = True
        
        # Update configuration
        mp_manager.update_config(
            loss_scale=loss_scale,
            autocast_enabled=autocast_enabled
        )
        
        # Train epoch
        train_epoch(model, epoch)
```

### 2. Framework-Specific Optimizations

```python
# PyTorch-specific optimizations
def optimize_pytorch_mixed_precision(model):
    """PyTorch-specific mixed precision optimizations"""
    
    # Enable optimized attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Use channels_last memory format for CNNs
    if isinstance(model, torch.nn.Conv2d):
        model = model.to(memory_format=torch.channels_last)
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    return model

# TensorFlow-specific optimizations
def optimize_tensorflow_mixed_precision():
    """TensorFlow-specific mixed precision optimizations"""
    
    # Enable mixed precision globally
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
```

### 3. Error Handling and Fallbacks

```python
def robust_mixed_precision_training(model, data_loader):
    """Robust training with mixed precision fallbacks"""
    
    try:
        # Attempt mixed precision training
        with mp_manager.autocast_context():
            train_with_mixed_precision(model, data_loader)
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM with mixed precision, falling back to FP32")
            # Clear cache and retry with FP32
            torch.cuda.empty_cache()
            train_with_fp32(model, data_loader)
        else:
            raise e
    
    except OverflowError:
        print("Numerical overflow detected, adjusting loss scaling")
        # Reduce loss scale and retry
        mp_manager.adjust_loss_scale(factor=0.5)
        train_with_mixed_precision(model, data_loader)
```

## Related Documentation

- [JIT Compiler Documentation](jit_compiler.md)
- [Performance Metrics Documentation](performance_metrics.md)
- [Training Engine Documentation](train_engine.md)
- [Quantizer Documentation](quantizer.md)
- [Memory Pool Documentation](memory_pool.md)
