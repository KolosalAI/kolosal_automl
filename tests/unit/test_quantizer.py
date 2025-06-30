import unittest
import numpy as np
from typing import Optional, Dict
import tempfile
import os
import json

# Import the classes from your codebase
from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
from modules.engine.quantizer import Quantizer


class TestQuantizer(unittest.TestCase):
    """Test suite for the Quantizer class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a default quantization config
        self.default_config = QuantizationConfig()
        
        # Create a default quantizer
        self.quantizer = Quantizer(self.default_config)
        
        # Create some test data
        self.test_data = np.array([-10.5, -5.2, 0.0, 3.7, 8.9], dtype=np.float32)
        self.test_array_2d = np.array([[-5.0, -2.5, 0.0], [1.5, 4.0, 7.5]], dtype=np.float32)
        self.test_array_3d = np.random.uniform(-10, 10, (2, 3, 4)).astype(np.float32)
        
        # Create data for per-channel tests
        self.channel_data = np.random.uniform(-10, 10, (3, 5, 5)).astype(np.float32)

    def test_initialization(self):
        """Test that the quantizer initializes correctly with default and custom configs."""
        # Test with default config
        q1 = Quantizer()
        self.assertEqual(q1.config.quantization_type, QuantizationType.INT8)
        self.assertEqual(q1.config.quantization_mode, QuantizationMode.DYNAMIC)
        
        # Test with custom config
        custom_config = QuantizationConfig(
            quantization_type=QuantizationType.UINT8,
            quantization_mode=QuantizationMode.SYMMETRIC,
            num_bits=8
        )
        q2 = Quantizer(custom_config)
        self.assertEqual(q2.config.quantization_type, QuantizationType.UINT8)
        self.assertEqual(q2.config.quantization_mode, QuantizationMode.SYMMETRIC)

    def test_update_config(self):
        """Test updating the quantizer configuration."""
        # Initial config
        self.assertEqual(self.quantizer.config.quantization_type, QuantizationType.INT8)
        
        # Update config
        new_config = QuantizationConfig(
            quantization_type=QuantizationType.INT16,
            num_bits=16
        )
        self.quantizer.update_config(new_config)
        
        # Verify the update
        self.assertEqual(self.quantizer.config.quantization_type, QuantizationType.INT16)
        self.assertEqual(self.quantizer.config.num_bits, 16)

    def test_get_config(self):
        """Test retrieving the current configuration."""
        config_dict = self.quantizer.get_config()
        
        # Verify the dictionary contains expected keys
        self.assertIn('quantization_type', config_dict)
        self.assertIn('quantization_mode', config_dict)
        self.assertIn('num_bits', config_dict)
        
        # Verify values match the current config
        self.assertEqual(config_dict['quantization_type'], self.quantizer.config.quantization_type)

    def test_basic_quantization_int8(self):
        """Test basic quantization with INT8 type."""
        # Configure quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.DYNAMIC
        )
        quantizer = Quantizer(config)
        
        # Quantize data
        quantized = quantizer.quantize(self.test_data)
        
        # Check type and range
        self.assertEqual(quantized.dtype, np.int8)
        self.assertTrue(np.all(quantized >= -128))
        self.assertTrue(np.all(quantized <= 127))

    def test_basic_quantization_uint8(self):
        """Test basic quantization with UINT8 type."""
        # Configure quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.UINT8,
            quantization_mode=QuantizationMode.DYNAMIC
        )
        quantizer = Quantizer(config)
        
        # Quantize data
        quantized = quantizer.quantize(self.test_data)
        
        # Check type and range
        self.assertEqual(quantized.dtype, np.uint8)
        self.assertTrue(np.all(quantized >= 0))
        self.assertTrue(np.all(quantized <= 255))

    def test_basic_quantization_int16(self):
        """Test basic quantization with INT16 type."""
        # Configure quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT16,
            quantization_mode=QuantizationMode.DYNAMIC,
            num_bits=16
        )
        quantizer = Quantizer(config)
        
        # Quantize data
        quantized = quantizer.quantize(self.test_data)
        
        # Check type and range
        self.assertEqual(quantized.dtype, np.int16)
        self.assertTrue(np.all(quantized >= -32768))
        self.assertTrue(np.all(quantized <= 32767))

    def test_basic_quantization_float16(self):
        """Test basic quantization with FLOAT16 type."""
        # Configure quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.FLOAT16
        )
        quantizer = Quantizer(config)
        
        # Quantize data
        quantized = quantizer.quantize(self.test_data)
        
        # Check type
        self.assertEqual(quantized.dtype, np.float16)

    def test_dequantization(self):
        """Test dequantization returns values close to the original."""
        # Configure quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.DYNAMIC
        )
        quantizer = Quantizer(config)
        
        # Quantize and then dequantize
        quantized = quantizer.quantize(self.test_data)
        dequantized = quantizer.dequantize(quantized)
        
        # Check type
        self.assertEqual(dequantized.dtype, np.float32)
        
        # We expect some loss due to quantization
        # For INT8, let's say we expect values within ±0.5 of the original
        max_diff = np.max(np.abs(dequantized - self.test_data))
        self.assertLessEqual(max_diff, 1.0)

    def test_quantize_dequantize(self):
        """Test the combined quantize_dequantize method."""
        # Configure quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.DYNAMIC
        )
        quantizer = Quantizer(config)
        
        # Apply quantize_dequantize
        result = quantizer.quantize_dequantize(self.test_data)
        
        # Check type
        self.assertEqual(result.dtype, np.float32)
        
        # We expect some loss due to quantization
        max_diff = np.max(np.abs(result - self.test_data))
        self.assertLessEqual(max_diff, 1.0)

    def test_calibration(self):
        """Test calibration with representative data."""
        # Configure quantizer for calibration
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.CALIBRATED
        )
        quantizer = Quantizer(config)
        
        # Create calibration data
        calib_data = [
            np.array([-10.0, -8.0, -6.0], dtype=np.float32),
            np.array([4.0, 6.0, 8.0], dtype=np.float32)
        ]
        
        # Calibrate
        quantizer.calibrate(calib_data)
        
        # Check that calibration flag is set
        self.assertTrue(quantizer._is_calibrated)
        
        # Quantize data and verify it uses calibration
        quantized = quantizer.quantize(self.test_data)
        self.assertEqual(quantized.dtype, np.int8)

    def test_stats_tracking(self):
        """Test that the quantizer tracks statistics."""
        # Reset stats
        self.quantizer.reset_stats()
        
        # Perform some operations to generate stats
        self.quantizer.quantize(self.test_data)
        self.quantizer.dequantize(self.quantizer.quantize(self.test_data))
        
        # Get stats
        stats = self.quantizer.get_stats()
        
        # Check stats tracking
        self.assertGreater(stats['quantize_calls'], 0)
        self.assertGreater(stats['dequantize_calls'], 0)
        self.assertGreater(stats['total_values'], 0)

    def test_symmetric_quantization(self):
        """Test symmetric quantization mode."""
        # Configure quantizer for symmetric quantization
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.SYMMETRIC,
            symmetric=True
        )
        quantizer = Quantizer(config)
        
        # Verify zero point is 0 for symmetric quantization
        quantizer.compute_scale_and_zero_point(self.test_data)
        self.assertEqual(quantizer.zero_point, 0.0)
        
        # Quantize and check some properties
        quantized = quantizer.quantize(self.test_data)
        
        # For symmetric, zero should map to zero
        zero_idx = np.where(self.test_data == 0.0)[0][0]
        self.assertEqual(quantized[zero_idx], 0)

    def test_asymmetric_quantization(self):
        """Test asymmetric quantization mode."""
        # Configure quantizer for asymmetric quantization
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.DYNAMIC,
            symmetric=False
        )
        quantizer = Quantizer(config)
        
        # For asymmetric, zero point may not be 0
        quantizer.compute_scale_and_zero_point(self.test_data)
        
        # Quantize and dequantize to verify round trip
        quantized = quantizer.quantize(self.test_data)
        dequantized = quantizer.dequantize(quantized)
        
        # Check general preservation of values
        max_diff = np.max(np.abs(dequantized - self.test_data))
        self.assertLessEqual(max_diff, 1.0)

    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        # Configure quantizer for per-channel quantization
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.DYNAMIC_PER_CHANNEL,
            per_channel=True
        )
        quantizer = Quantizer(config)
        
        # Quantize with channel dimension specified
        channel_dim = 0
        quantized = quantizer.quantize(self.channel_data, channel_dim=channel_dim)
        
        # Dequantize back
        dequantized = quantizer.dequantize(quantized, channel_dim=channel_dim)
        
        # Check shape preservation
        self.assertEqual(dequantized.shape, self.channel_data.shape)
        
        # Check general preservation of values
        max_diff = np.max(np.abs(dequantized - self.channel_data))
        self.assertLessEqual(max_diff, 2.0)  # Per-channel might have larger differences

    def test_export_import_parameters(self):
        """Test exporting and importing quantization parameters."""
        # Configure and calibrate quantizer
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            quantization_mode=QuantizationMode.CALIBRATED
        )
        quantizer = Quantizer(config)
        
        # Calibrate
        calib_data = [
            np.array([-10.0, -8.0, -6.0], dtype=np.float32),
            np.array([4.0, 6.0, 8.0], dtype=np.float32)
        ]
        quantizer.calibrate(calib_data)
        
        # Export parameters
        params = quantizer.export_import_parameters()
        
        # Create a new quantizer and import parameters
        new_quantizer = Quantizer(config)
        new_quantizer.load_parameters(params)
        
        # Verify imported parameters match
        self.assertEqual(new_quantizer._global_scale, quantizer._global_scale)
        self.assertEqual(new_quantizer._global_zero_point, quantizer._global_zero_point)
        self.assertEqual(new_quantizer._is_calibrated, quantizer._is_calibrated)

    def test_mixed_precision_quantization(self):
        """Test mixed precision quantization if enabled."""
        # Configure for mixed precision
        config = QuantizationConfig(
            quantization_type=QuantizationType.MIXED,
            enable_mixed_precision=True,
            mixed_precision_layers=['layer1', 'layer2']
        )
        
        # Set custom quantization config
        if hasattr(config, 'custom_quantization_config'):
            config.custom_quantization_config = {
                'layer1': {'bits': 16, 'type': QuantizationType.INT16},
                'layer2': {'bits': 8, 'type': QuantizationType.INT8}
            }
        
        quantizer = Quantizer(config)
        
        # Test with layer name to trigger mixed precision handling
        quantized1 = quantizer.quantize(self.test_data, layer_name='layer1')
        
        # Verify should_quantize_layer function
        should_quantize = quantizer.should_quantize_layer('layer1')
        self.assertTrue(should_quantize)
        
        # Test get_layer_quantization_params
        layer_params = quantizer.get_layer_quantization_params('layer1')
        if hasattr(config, 'custom_quantization_config'):
            self.assertEqual(layer_params['bits'], 16)
        else:
            # Without custom config, should fall back to defaults
            self.assertEqual(layer_params['bits'], 8)

    def test_cache_functionality(self):
        """Test the caching functionality."""
        # Configure with cache enabled
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            enable_cache=True,
            cache_size=100
        )
        quantizer = Quantizer(config)
        
        # Quantize and dequantize data multiple times to test cache
        for _ in range(5):
            quantized = quantizer.quantize(self.test_data)
            quantizer.dequantize(quantized)
        
        # Check stats for cache hits
        stats = quantizer.get_stats()
        self.assertGreaterEqual(stats['cache_hits'], 3)  # Should have some cache hits
        
        # Clear cache and verify
        quantizer.clear_cache()
        self.assertEqual(len(quantizer._value_cache), 0)

    def test_hash_array(self):
        """Test the hash_array static method."""
        # Create some test arrays
        small_array = np.array([1, 2, 3], dtype=np.float32)
        large_array = np.random.uniform(-10, 10, (100, 100)).astype(np.float32)
        
        # Test hash for small array
        hash1 = Quantizer.hash_array(small_array)
        self.assertIsInstance(hash1, str)
        
        # Test hash for large array
        hash2 = Quantizer.hash_array(large_array)
        self.assertIsInstance(hash2, str)
        
        # Hashes should be different
        self.assertNotEqual(hash1, hash2)
        
        # Same array should give same hash
        hash3 = Quantizer.hash_array(small_array)
        self.assertEqual(hash1, hash3)
        
        # Test error for non-ndarray
        with self.assertRaises(TypeError):
            Quantizer.hash_array([1, 2, 3])

    def test_multithreading_safety(self):
        """Test thread safety of the quantizer (basic test)."""
        import threading
        
        # Configure quantizer
        quantizer = Quantizer()
        
        # Create a function to run in multiple threads
        def quantize_in_thread():
            for _ in range(10):
                data = np.random.uniform(-10, 10, (5,)).astype(np.float32)
                quantized = quantizer.quantize(data)
                dequantized = quantizer.dequantize(quantized)
        
        # Create and start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=quantize_in_thread)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # If we got here without exceptions, that's good
        # We can also check that stats were updated
        stats = quantizer.get_stats()
        self.assertGreaterEqual(stats['quantize_calls'], 50)  # 5 threads * 10 calls
        self.assertGreaterEqual(stats['dequantize_calls'], 50)


if __name__ == '__main__':
    unittest.main()