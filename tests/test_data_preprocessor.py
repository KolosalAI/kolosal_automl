import unittest
import numpy as np
import os
import tempfile
import shutil

from typing import Dict, Any

# Make sure to import your DataPreprocessor, PreprocessorConfig, NormalizationType, etc.
from modules.engine.data_preprocessor import DataPreprocessor
from modules.configs import PreprocessorConfig, NormalizationType


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """
        Create test fixture for each test method.
        This method is called before every test_xxx method.
        """
        self.config = PreprocessorConfig(
            normalization=NormalizationType.STANDARD,
            debug_mode=True,  # Enable debug mode for more verbose logging
            detect_outliers=True,
            outlier_method="iqr",
            outlier_params={"threshold": 1.5, "clip": True},
            handle_nan=True,
            handle_inf=True,
            nan_strategy="mean",
            inf_strategy="mean",
            parallel_processing=False,
            chunk_size=None,  # For simplicity, disable chunk processing in these tests
            cache_enabled=True
        )
        self.preprocessor = DataPreprocessor(self.config)

        # Generate a small synthetic dataset
        np.random.seed(42)
        self.X = np.random.randn(100, 5)  # 100 samples, 5 features

        # Intentionally add a few NaN/Inf values to test data cleaning
        self.X[0, 0] = np.nan
        self.X[1, 1] = np.inf
        self.X[2, 2] = -np.inf

    def tearDown(self):
        """
        Clean up after each test method.
        This method is called after every test_xxx method.
        """
        pass

    def test_fit_transform_standard(self):
        """
        Test standard (z-score) normalization: 
        - Fit the preprocessor on data with NaN/inf
        - Transform the data
        - Check that the transformed data has mean approx 0 and std approx 1
        - Ensure no NaNs or inf remain in the output
        """
        self.preprocessor.fit(self.X)

        # Check that the preprocessor is fitted
        self.assertTrue(self.preprocessor._fitted, "Preprocessor should be marked as fitted")

        # Check stats keys
        self.assertIn('mean', self.preprocessor._stats)
        self.assertIn('std', self.preprocessor._stats)

        # Transform data
        X_transformed = self.preprocessor.transform(self.X)

        # Check shape
        self.assertEqual(X_transformed.shape, self.X.shape)

        # Check for NaNs or inf
        self.assertFalse(np.isnan(X_transformed).any(), "Transformed data should not contain NaN")
        self.assertTrue(np.all(np.isfinite(X_transformed)), "Transformed data should not contain inf")

        # Check mean/std of each feature is approximately 0 and 1
        means = np.mean(X_transformed, axis=0)
        stds = np.std(X_transformed, axis=0)
        for mean_val, std_val in zip(means, stds):
            self.assertAlmostEqual(mean_val, 0.0, places=5)
            self.assertAlmostEqual(std_val, 1.0, places=5)

    def test_fit_transform_minmax(self):
        """
        Test min-max normalization.
        """
        # Change config to min-max
        self.preprocessor.config.normalization = NormalizationType.MINMAX
        self.preprocessor.fit(self.X)
        X_transformed = self.preprocessor.transform(self.X)

        # Check stats
        self.assertIn('min', self.preprocessor._stats)
        self.assertIn('max', self.preprocessor._stats)
        self.assertIn('scale', self.preprocessor._stats)

        # Check range in [0, 1]
        self.assertTrue(np.all(X_transformed >= 0.0))
        self.assertTrue(np.all(X_transformed <= 1.0))

    def test_outlier_detection_iqr(self):
        """
        Test outlier handling with IQR. 
        We expect outliers to be clipped to the lower/upper IQR range.
        """
        # Fit with the current config (IQR outliers detection is enabled)
        self.preprocessor.fit(self.X)
        X_transformed = self.preprocessor.transform(self.X)

        # Check that the iqr boundaries are computed
        self.assertIn('outlier_lower', self.preprocessor._stats)
        self.assertIn('outlier_upper', self.preprocessor._stats)

        lower_bound = self.preprocessor._stats['outlier_lower']
        upper_bound = self.preprocessor._stats['outlier_upper']

        # Check that all values are clipped within the computed bounds
        self.assertTrue(np.all(X_transformed >= lower_bound - 1e-9))
        self.assertTrue(np.all(X_transformed <= upper_bound + 1e-9))

    def test_partial_fit(self):
        """
        Test partial fitting with multiple batches of data.
        We'll simulate streaming data in two batches.
        """
        # Split data into two batches
        X_batch1 = self.X[:50]
        X_batch2 = self.X[50:]

        # Perform partial_fit on first batch
        self.preprocessor.partial_fit(X_batch1)

        # Check that preprocessor is fitted after first batch
        self.assertTrue(self.preprocessor._fitted)

        # Stats after first batch
        stats_after_first_batch = self.preprocessor.get_stats()

        # Partial fit again with the second batch
        self.preprocessor.partial_fit(X_batch2)

        # Stats after second batch
        stats_after_second_batch = self.preprocessor.get_stats()

        # We expect some differences in statistics (unless data is identical)
        # For instance, the mean should generally shift after seeing more data
        if 'mean' in stats_after_first_batch:
            self.assertTrue(
                not np.allclose(stats_after_first_batch['mean'],
                                stats_after_second_batch['mean']),
                "Means after second batch should differ from the first batch in partial fit"
            )

        # Transform entire dataset after partial fit
        X_transformed = self.preprocessor.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)

    def test_inverse_transform_standard(self):
        """
        Test that inverse_transform roughly recovers the original data for 
        Standard normalization.
        """
        self.preprocessor.fit(self.X)
        X_transformed = self.preprocessor.transform(self.X)

        # Now invert
        X_inverted = self.preprocessor.inverse_transform(X_transformed)

        # Check shape
        self.assertEqual(X_inverted.shape, self.X.shape)

        # Because outliers are clipped and NaNs replaced, 
        # the recovered data won't be identical, 
        # but it should be close where data wasn't NaN/Inf or clipped.
        # We'll test the approximate difference on a subset that wasn't 
        # originally NaN or inf or out of normal range.

        # Filter out the rows with special values
        valid_mask = np.isfinite(self.X).all(axis=1)
        X_valid = self.X[valid_mask]
        X_inverted_valid = X_inverted[valid_mask]

        # Compare some statistic (e.g., means) to see if inversion is close
        original_mean = np.mean(X_valid, axis=0)
        inverted_mean = np.mean(X_inverted_valid, axis=0)
        for om, im in zip(original_mean, inverted_mean):
            # Allow some tolerance
            self.assertAlmostEqual(om, im, places=1)

    def test_serialization(self):
        """
        Test saving and loading the preprocessor from disk.
        """
        self.preprocessor.fit(self.X)

        # Create a temporary directory to store the model
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "preprocessor.json")
            self.preprocessor.save(model_path)

            # Load a new preprocessor from disk
            loaded_preprocessor = DataPreprocessor.load(model_path)
            self.assertTrue(loaded_preprocessor._fitted)

            # Check that key stats match
            orig_stats = self.preprocessor.get_stats()
            new_stats = loaded_preprocessor.get_stats()

            # Compare means in standard mode
            if 'mean' in orig_stats and 'mean' in new_stats:
                self.assertTrue(np.allclose(orig_stats['mean'], new_stats['mean'], atol=1e-6))

            # Transform with the loaded preprocessor
            X_transformed_loaded = loaded_preprocessor.transform(self.X)
            X_transformed = self.preprocessor.transform(self.X)
            self.assertTrue(np.allclose(X_transformed_loaded, X_transformed, atol=1e-6))

        finally:
            # Remove temp directory
            shutil.rmtree(temp_dir)

    def test_exception_unfitted_transform(self):
        """
        Test that calling transform on an unfitted preprocessor raises an error.
        """
        with self.assertRaises(RuntimeError):
            _ = self.preprocessor.transform(self.X)

    def test_exception_dimension_mismatch(self):
        """
        Test that transforming data with different feature dimension raises an exception.
        """
        # Fit with 5 features
        self.preprocessor.fit(self.X)
        X_wrong_dim = np.random.randn(10, 3)  # Only 3 features
        with self.assertRaisesRegex(InputValidationError, "features"):
            _ = self.preprocessor.transform(X_wrong_dim)

    def test_reset(self):
        """
        Test that reset() clears out fitted state.
        """
        self.preprocessor.fit(self.X)
        self.assertTrue(self.preprocessor._fitted)

        self.preprocessor.reset()
        self.assertFalse(self.preprocessor._fitted)
        self.assertEqual(self.preprocessor._stats, {})

    def test_copy(self):
        """
        Test copying a fitted preprocessor.
        """
        self.preprocessor.fit(self.X)
        preprocessor_copy = self.preprocessor.copy()

        # They should not be the same object
        self.assertNotEqual(id(self.preprocessor), id(preprocessor_copy))

        # But they should share the same fitted stats (by value)
        orig_stats = self.preprocessor.get_stats()
        copy_stats = preprocessor_copy.get_stats()

        if 'mean' in orig_stats and 'mean' in copy_stats:
            self.assertTrue(np.allclose(orig_stats['mean'], copy_stats['mean']))
        
        # Transform the same data and compare results
        X_orig_trans = self.preprocessor.transform(self.X)
        X_copy_trans = preprocessor_copy.transform(self.X)
        self.assertTrue(np.allclose(X_orig_trans, X_copy_trans, atol=1e-6))


# If you want to run this suite directly (e.g., `python test_datapreprocessor.py`)
if __name__ == '__main__':
    unittest.main()
