import unittest
import numpy as np
import tempfile
import threading
import time
import os
import json

from modules.configs import QuantizationConfig, QuantizationType, QuantizationMode
from modules.engine.quantizer import Quantizer


class TestQuantizer(unittest.TestCase):
    def setUp(self):
        # Create a default config for most tests
        self.default_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
            enable_cache=True,
            cache_size=256,
            min_percentile=0.1,
            max_percentile=99.9,
            buffer_size=1024,
            outlier_threshold=3.0,
            # Additional parameters for validation
            error_on_nan=False,
            error_on_inf=False,
            use_percentile=False,
            optimize_memory=True,
            symmetric=False,
            per_channel=False,
            enable_mixed_precision=False,
            quantize_bias=True,
            quantize_weights_only=False,
            enable_requantization=False,
            mixed_precision_layers=[],
            skip_layers=[],
            custom_quantization_config={}
        )
        self.quantizer = Quantizer(self.default_config)

    def test_get_config(self):
        config_dict = self.quantizer.get_config()
        self.assertEqual(config_dict['quantization_type'], self.default_config.quantization_type)
        self.assertEqual(config_dict['quantization_mode'], self.default_config.quantization_mode)
        self.assertEqual(config_dict['buffer_size'], self.default_config.buffer_size)

    def test_compute_scale_and_zero_point_empty(self):
        empty_data = np.array([], dtype=np.float32)
        scale, zero_point = self.quantizer.compute_scale_and_zero_point(empty_data)
        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(zero_point, 0.0)
        
    def test_compute_scale_and_zero_point_symmetric(self):
        # Create a symmetric config
        sym_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.SYMMETRIC.value,
            symmetric=True
        )
        sym_quantizer = Quantizer(sym_config)
        
        data = np.array([-5.0, 3.0, -2.0, 4.0], dtype=np.float32)
        scale, zero_point = sym_quantizer.compute_scale_and_zero_point(data)
        
        # For symmetric quantization, zero_point should be 0.0
        self.assertAlmostEqual(zero_point, 0.0)
        # Scale should relate to the max abs value
        abs_max = max(abs(min(data)), abs(max(data))) * 1.01  # Add 1% margin
        expected_scale = abs_max / 127.0  # Max abs value for int8
        self.assertAlmostEqual(scale, expected_scale, places=5)

    def test_compute_scale_and_zero_point_asymmetric(self):
        # Create an asymmetric config
        asym_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.ASYMMETRIC.value,
            symmetric=False
        )
        asym_quantizer = Quantizer(asym_config)
        
        data = np.array([-5.0, 3.0, -2.0, 4.0], dtype=np.float32)
        scale, zero_point = asym_quantizer.compute_scale_and_zero_point(data)
        
        # Zero point should not be 0 for asymmetric
        self.assertNotEqual(zero_point, 0.0)
        
    def test_compute_scale_with_outlier_threshold(self):
        # Create config with outlier threshold
        outlier_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC.value,
            outlier_threshold=2.0  # Standard deviations for clipping
        )
        outlier_quantizer = Quantizer(outlier_config)
        
        # Create data with outliers
        base_data = np.random.normal(0, 1, 100).astype(np.float32)
        data_with_outliers = np.copy(base_data)
        data_with_outliers[0] = 10.0  # Add an extreme outlier
        
        scale1, _ = outlier_quantizer.compute_scale_and_zero_point(base_data)
        scale2, _ = outlier_quantizer.compute_scale_and_zero_point(data_with_outliers)
        
        # Scales should be similar due to outlier handling
        self.assertLess(abs(scale1 - scale2), 0.1)

    def test_compute_scale_with_percentile(self):
        # Create config with percentile option
        percentile_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC.value,
            use_percentile=True,
            min_percentile=1.0,
            max_percentile=99.0
        )
        percentile_quantizer = Quantizer(percentile_config)
        
        # Create data with outliers
        data = np.random.normal(0, 1, 1000).astype(np.float32)
        data[0] = 10.0  # Add outliers
        data[1] = -10.0
        
        scale, _ = percentile_quantizer.compute_scale_and_zero_point(data)
        
        # The scale should be influenced by the percentile settings
        # and less affected by the extreme outliers
        expected_range = np.percentile(data, 99.0) - np.percentile(data, 1.0)
        expected_scale = expected_range / 255.0  # Full int8 range
        
        self.assertLess(abs(scale - expected_scale), 0.1)

    def test_quantize_dequantize_consistency(self):
        # Create a simple linearly spaced array
        data = np.linspace(-1, 1, 100, dtype=np.float32)
        q_data = self.quantizer.quantize(data)
        dq_data = self.quantizer.dequantize(q_data)
        # The reconstruction error should be reasonably small
        mse = np.mean((data - dq_data) ** 2)
        self.assertLess(mse, 0.1)

    def test_quantize_invalid_input(self):
        # Passing a non-ndarray should raise a TypeError (via input validation)
        with self.assertRaises(TypeError):
            self.quantizer.quantize("not an array")

        # Passing an array with wrong dtype: the validation only warns,
        # so converting to float32 should allow quantization to proceed.
        data_int = np.array([1, 2, 3], dtype=np.int32)
        try:
            q = self.quantizer.quantize(data_int.astype(np.float32))
            self.assertIsInstance(q, np.ndarray)
        except Exception as e:
            self.fail("quantize raised an unexpected exception: " + str(e))

    def test_calibrate(self):
        # Generate some random calibration data
        data = np.random.uniform(-1, 1, size=(100,)).astype(np.float32)
        calibration_results = self.quantizer.calibrate(data)
        expected_keys = {
            'data_min', 'data_max', 'data_mean', 'data_std',
            'scale', 'zero_point', 'mse', 'mae', 'max_error', 'snr_db', 'error_hist'
        }
        self.assertTrue(expected_keys.issubset(set(calibration_results.keys())))

    def test_export_import_parameters(self):
        # Run quantization to update parameters
        data = np.random.uniform(-1, 1, size=(100,)).astype(np.float32)
        _ = self.quantizer.quantize(data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            file_path = tmp.name

        try:
            self.quantizer.export_parameters(file_path)

            # Create a new quantizer and import the parameters
            new_quantizer = Quantizer(self.default_config)
            new_quantizer.import_parameters(file_path)

            params_orig = self.quantizer.get_parameters()
            params_new = new_quantizer.get_parameters()

            self.assertAlmostEqual(params_orig['global']['scale'], params_new['global']['scale'])
            self.assertAlmostEqual(params_orig['global']['zero_point'], params_new['global']['zero_point'])
        finally:
            os.remove(file_path)

    def test_reset_stats(self):
        data = np.random.uniform(-1, 1, size=(50,)).astype(np.float32)
        _ = self.quantizer.quantize(data)
        _ = self.quantizer.dequantize(self.quantizer.quantize(data))
        stats_before = self.quantizer.get_quantization_stats()
        self.quantizer.reset_stats()
        stats_after = self.quantizer.get_quantization_stats()
        self.assertNotEqual(stats_before['quantize_calls'], stats_after['quantize_calls'])
        self.assertEqual(stats_after['quantize_calls'], 0)
        self.assertEqual(stats_after['dequantize_calls'], 0)

    def test_clear_cache(self):
        # Use a small array to trigger caching
        data = np.array([0.5, -0.5], dtype=np.float32)
        q_data = self.quantizer.quantize(data)
        _ = self.quantizer.dequantize(q_data)
        # Now clear the cache and verify that cached_values_count is zero.
        self.quantizer.clear_cache()
        stats = self.quantizer.get_quantization_stats()
        self.assertEqual(stats['cached_values_count'], 0)

    def test_per_channel_quantization(self):
        # Create per-channel config
        per_channel_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC_PER_CHANNEL.value,
            per_channel=True
        )
        per_channel_quantizer = Quantizer(per_channel_config)
        
        # Create sample data with shape (channels, features)
        data = np.random.uniform(-2, 2, size=(3, 10)).astype(np.float32)
        q_channel = per_channel_quantizer.quantize(data, channel_dim=0)
        dq_channel = per_channel_quantizer.dequantize(q_channel, channel_dim=0)
        mse_channel = np.mean((data - dq_channel) ** 2)
        self.assertLess(mse_channel, 0.5)

        # Test that an out-of-bound channel dimension raises an error
        with self.assertRaises(ValueError):
            per_channel_quantizer.quantize(data, channel_dim=5)
        with self.assertRaises(ValueError):
            per_channel_quantizer.dequantize(q_channel, channel_dim=5)

    def test_different_quantization_types(self):
        # Test int8
        int8_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC.value
        )
        int8_quantizer = Quantizer(int8_config)
        
        # Test uint8
        uint8_config = QuantizationConfig(
            quantization_type=QuantizationType.UINT8.value,
            quantization_mode=QuantizationMode.DYNAMIC.value
        )
        uint8_quantizer = Quantizer(uint8_config)
        
        # Test int16
        int16_config = QuantizationConfig(
            quantization_type=QuantizationType.INT16.value,
            quantization_mode=QuantizationMode.DYNAMIC.value
        )
        int16_quantizer = Quantizer(int16_config)
        
        # Test float16
        float16_config = QuantizationConfig(
            quantization_type=QuantizationType.FLOAT16.value,
            quantization_mode=QuantizationMode.DYNAMIC.value
        )
        float16_quantizer = Quantizer(float16_config)
        
        data = np.random.uniform(-1, 1, size=(50,)).astype(np.float32)
        
        # Check each quantizer returns the correct dtype
        self.assertEqual(int8_quantizer.quantize(data).dtype, np.int8)
        self.assertEqual(uint8_quantizer.quantize(data).dtype, np.uint8)
        self.assertEqual(int16_quantizer.quantize(data).dtype, np.int16)
        self.assertEqual(float16_quantizer.quantize(data).dtype, np.float16)
        
        # Check roundtrip consistency for each type
        for quantizer in [int8_quantizer, uint8_quantizer, int16_quantizer, float16_quantizer]:
            q_data = quantizer.quantize(data)
            dq_data = quantizer.dequantize(q_data)
            mse = np.mean((data - dq_data) ** 2)
            self.assertLess(mse, 0.5)  # Different types have different precision
            
    def test_none_quantization_type(self):
        # Test when no quantization is requested
        none_config = QuantizationConfig(
            quantization_type=QuantizationType.NONE.value
        )
        none_quantizer = Quantizer(none_config)
        
        data = np.random.uniform(-1, 1, size=(50,)).astype(np.float32)
        q_data = none_quantizer.quantize(data)
        
        # Should return the original data unchanged
        np.testing.assert_array_equal(data, q_data)
        self.assertEqual(q_data.dtype, np.float32)

    def test_update_config(self):
        new_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.ASYMMETRIC.value,
            enable_cache=False,
            cache_size=128,
            min_percentile=0.1,
            max_percentile=99.9,
            buffer_size=512,
            outlier_threshold=2.0,
            error_on_nan=False,
            error_on_inf=False,
            use_percentile=True,
            optimize_memory=True,
            symmetric=False,
            per_channel=True,
            enable_mixed_precision=False,
            quantize_bias=False,
            quantize_weights_only=True,
            enable_requantization=False
        )
        self.quantizer.update_config(new_config)
        config_dict = self.quantizer.get_config()
        self.assertEqual(config_dict['quantization_mode'], new_config.quantization_mode)
        self.assertEqual(config_dict['cache_size'], new_config.cache_size)
        self.assertEqual(config_dict['buffer_size'], new_config.buffer_size)
        self.assertEqual(config_dict['quantize_bias'], new_config.quantize_bias)
        self.assertEqual(config_dict['quantize_weights_only'], new_config.quantize_weights_only)
        self.assertEqual(config_dict['per_channel'], new_config.per_channel)

    def test_mixed_precision_quantization(self):
        # Create a config with mixed precision enabled
        mixed_config = QuantizationConfig(
            quantization_type=QuantizationType.MIXED.value,
            enable_mixed_precision=True,
            mixed_precision_layers=['dense1', 'conv2'],
            custom_quantization_config={
                'dense1': {'bits': 16, 'type': QuantizationType.INT16.value},
                'conv2': {'bits': 8, 'type': QuantizationType.INT8.value}
            }
        )
        mixed_quantizer = Quantizer(mixed_config)
        
        data = np.random.uniform(-1, 1, size=(50,)).astype(np.float32)
        
        # Check layer-specific quantization params
        dense_params = mixed_quantizer.get_layer_quantization_params('dense1')
        self.assertEqual(dense_params['bits'], 16)
        self.assertEqual(dense_params['type'], QuantizationType.INT16.value)
        
        # Check should_quantize_layer
        self.assertTrue(mixed_quantizer.should_quantize_layer('conv2'))
        self.assertTrue(mixed_quantizer.should_quantize_layer('dense1'))
        
        # Add a skip layer
        mixed_config.skip_layers = ['conv3']
        mixed_quantizer.update_config(mixed_config)
        self.assertFalse(mixed_quantizer.should_quantize_layer('conv3'))
        
        # Check bias quantization
        mixed_config.quantize_bias = False
        mixed_quantizer.update_config(mixed_config)
        self.assertFalse(mixed_quantizer.should_quantize_layer('bias_layer'))
        
        # Check weights-only quantization
        mixed_config.quantize_weights_only = True
        mixed_quantizer.update_config(mixed_config)
        self.assertFalse(mixed_quantizer.should_quantize_layer('activation_layer'))
    
    def test_input_validation_with_nan_inf(self):
        # Create a config that raises errors on NaN and Inf
        config_nan = QuantizationConfig(
            quantization_type=QuantizationType.INT8.value,
            quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH.value,
            enable_cache=True,
            cache_size=256,
            min_percentile=0.1,
            max_percentile=99.9,
            buffer_size=1024,
            outlier_threshold=3.0,
            error_on_nan=True,
            error_on_inf=True,
            use_percentile=False,
            optimize_memory=True,
        )
        quantizer_nan = Quantizer(config_nan)
        data_nan = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = quantizer_nan.quantize(data_nan, validate=True)
        data_inf = np.array([1.0, np.inf, 2.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = quantizer_nan.quantize(data_inf, validate=True)
            
    def test_hash_array(self):
        # Test the hash_array static method
        small_arr = np.array([1, 2, 3, 4], dtype=np.float32)
        hash1 = Quantizer.hash_array(small_arr)
        self.assertIsInstance(hash1, str)
        
        # Same array should produce same hash
        hash2 = Quantizer.hash_array(small_arr)
        self.assertEqual(hash1, hash2)
        
        # Different array should produce different hash
        different_arr = np.array([1, 2, 3, 5], dtype=np.float32)
        hash3 = Quantizer.hash_array(different_arr)
        self.assertNotEqual(hash1, hash3)
        
        # Test with large array
        large_arr = np.random.uniform(-1, 1, size=(200,)).astype(np.float32)
        hash_large = Quantizer.hash_array(large_arr)
        self.assertIsInstance(hash_large, str)
        
        # Test with non-ndarray input
        with self.assertRaises(TypeError):
            Quantizer.hash_array([1, 2, 3])
        
    def test_quantize_dequantize_function(self):
        data = np.random.uniform(-1, 1, size=(100,)).astype(np.float32)
        dq_data = self.quantizer.quantize_dequantize(data)
        mse = np.mean((data - dq_data) ** 2)
        self.assertLess(mse, 0.1)

    def test_quantize_large_arrays(self):
        # Create a large array to test vectorized performance
        large_data = np.random.uniform(-1, 1, size=(1000, 1000)).astype(np.float32)
        start_time = time.time()
        q_data = self.quantizer.quantize(large_data)
        dq_data = self.quantizer.dequantize(q_data)
        end_time = time.time()
        
        # Check that results are reasonable
        mse = np.mean((large_data - dq_data) ** 2)
        self.assertLess(mse, 0.1)
        
        # Check that the processing time is recorded in stats
        stats = self.quantizer.get_quantization_stats()
        self.assertGreater(stats['processing_time_ms'], 0)
        
        # Check that the elapsed time makes sense (should take some time)
        elapsed = end_time - start_time
        print(f"Large array quantization+dequantization took {elapsed:.4f} seconds")
        self.assertGreater(elapsed, 0)  # Should take some time

    def test_thread_safety(self):
        # Test concurrent quantization and dequantization calls
        errors = []

        def worker():
            try:
                for _ in range(10):
                    data = np.random.uniform(-1, 1, size=(50, 10)).astype(np.float32)
                    q = self.quantizer.quantize(data)
                    dq = self.quantizer.dequantize(q)
                    err = np.mean((data - dq) ** 2)
                    if err >= 1.0:
                        errors.append(err)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred in threads: {errors}")


if __name__ == "__main__":
    unittest.main()