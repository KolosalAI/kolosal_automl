import unittest
import numpy as np
import tempfile
import threading
import time
import os

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
        # Create sample data with shape (channels, features)
        data = np.random.uniform(-2, 2, size=(3, 10)).astype(np.float32)
        q_channel = self.quantizer.quantize(data, channel_dim=0)
        dq_channel = self.quantizer.dequantize(q_channel, channel_dim=0)
        mse_channel = np.mean((data - dq_channel) ** 2)
        self.assertLess(mse_channel, 0.5)

        # Test that an out-of-bound channel dimension raises an error
        with self.assertRaises(ValueError):
            self.quantizer.quantize(data, channel_dim=5)
        with self.assertRaises(ValueError):
            self.quantizer.dequantize(q_channel, channel_dim=5)

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
        )
        self.quantizer.update_config(new_config)
        config_dict = self.quantizer.get_config()
        self.assertEqual(config_dict['quantization_mode'], new_config.quantization_mode)
        self.assertEqual(config_dict['cache_size'], new_config.cache_size)
        self.assertEqual(config_dict['buffer_size'], new_config.buffer_size)

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

    def test_quantize_dequantize_function(self):
        data = np.random.uniform(-1, 1, size=(100,)).astype(np.float32)
        dq_data = self.quantizer.quantize_dequantize(data)
        mse = np.mean((data - dq_data) ** 2)
        self.assertLess(mse, 0.1)

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
