use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Precision mode for computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Full precision (f64)
    Full,
    /// Half precision (f16 simulated as f32 with reduced range)
    Half,
    /// BFloat16 (brain floating point)
    BFloat16,
    /// Mixed precision (auto-select per operation)
    Mixed,
    /// Auto-detect best precision based on data and hardware
    Auto,
}

impl Default for PrecisionMode {
    fn default() -> Self {
        PrecisionMode::Full
    }
}

/// Configuration for mixed precision training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    pub enabled: bool,
    pub mode: PrecisionMode,
    pub loss_scale: f64,
    pub dynamic_loss_scaling: bool,
    /// Multiply loss scale by this factor on successful steps
    pub loss_scale_factor: f64,
    /// Divide loss scale by this factor on overflow
    pub loss_scale_backoff: f64,
    /// Number of successful steps before increasing loss scale
    pub loss_scale_window: usize,
    pub max_loss_scale: f64,
    pub min_loss_scale: f64,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: PrecisionMode::Mixed,
            loss_scale: 65536.0,
            dynamic_loss_scaling: true,
            loss_scale_factor: 2.0,
            loss_scale_backoff: 0.5,
            loss_scale_window: 2000,
            max_loss_scale: 2.0_f64.powi(24),
            min_loss_scale: 1.0,
        }
    }
}

/// Statistics for mixed precision operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionStats {
    pub operations_count: u64,
    pub fp32_ops: u64,
    pub fp16_ops: u64,
    pub bf16_ops: u64,
    pub overflow_count: u64,
    pub underflow_count: u64,
    pub total_speedup_estimate: f64,
    pub memory_savings_bytes: u64,
}

impl Default for MixedPrecisionStats {
    fn default() -> Self {
        Self {
            operations_count: 0,
            fp32_ops: 0,
            fp16_ops: 0,
            bf16_ops: 0,
            overflow_count: 0,
            underflow_count: 0,
            total_speedup_estimate: 1.0,
            memory_savings_bytes: 0,
        }
    }
}

/// Manager for mixed precision computation.
///
/// Handles precision selection, loss scaling, and gradient management
/// for efficient mixed-precision training workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionManager {
    config: MixedPrecisionConfig,
    stats: MixedPrecisionStats,
    /// Current dynamic loss scale
    current_loss_scale: f64,
    /// Steps since last overflow (for dynamic scaling window)
    steps_since_overflow: usize,
}

impl MixedPrecisionManager {
    /// Create a new MixedPrecisionManager with the given configuration.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let current_loss_scale = config.loss_scale;
        Self {
            config,
            stats: MixedPrecisionStats::default(),
            current_loss_scale,
            steps_since_overflow: 0,
        }
    }

    /// Detect whether the platform has FP16 support.
    ///
    /// On x86_64, checks for F16C instruction set.
    /// On aarch64 (Apple Silicon / ARM), FP16 is generally supported.
    pub fn detect_fp16_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "f16c")]
            return true;
            #[cfg(not(target_feature = "f16c"))]
            return false;
        }
        #[cfg(target_arch = "aarch64")]
        {
            return true;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Detect whether the platform has BF16 support.
    ///
    /// BF16 is natively supported on newer x86_64 (AVX-512 BF16) and aarch64 CPUs.
    pub fn detect_bf16_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "avx512bf16")]
            return true;
            #[cfg(not(target_feature = "avx512bf16"))]
            return false;
        }
        #[cfg(target_arch = "aarch64")]
        {
            return true;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Downcast f64 data to f32 with precision checks.
    ///
    /// Values outside the f16 representable range (~65504) are clamped.
    /// Tracks overflow/underflow statistics.
    pub fn optimize_precision(&mut self, data: &[f64]) -> Result<Vec<f32>, String> {
        if !self.config.enabled {
            return Ok(data.iter().map(|&v| v as f32).collect());
        }

        let half_max: f64 = 65504.0;
        let half_min_positive: f64 = 6.1e-5;

        let mode = if self.config.mode == PrecisionMode::Auto {
            self.select_precision(data)
        } else {
            self.config.mode.clone()
        };

        let result: Vec<f32> = data
            .iter()
            .map(|&v| {
                self.stats.operations_count += 1;
                match mode {
                    PrecisionMode::Half => {
                        self.stats.fp16_ops += 1;
                        if v.is_nan() || v.is_infinite() {
                            self.stats.overflow_count += 1;
                            return v as f32;
                        }
                        let abs_v = v.abs();
                        if abs_v > half_max {
                            self.stats.overflow_count += 1;
                            (v.signum() * half_max) as f32
                        } else if abs_v > 0.0 && abs_v < half_min_positive {
                            self.stats.underflow_count += 1;
                            0.0_f32
                        } else {
                            v as f32
                        }
                    }
                    PrecisionMode::BFloat16 => {
                        self.stats.bf16_ops += 1;
                        // BF16 has same range as f32 but lower mantissa precision (8 bits)
                        // Simulate by rounding to ~3 decimal digits of precision
                        let f = v as f32;
                        let bits = f.to_bits();
                        // Zero out lowest 16 bits of mantissa to simulate bf16 truncation
                        let truncated = bits & 0xFFFF_0000;
                        f32::from_bits(truncated)
                    }
                    _ => {
                        self.stats.fp32_ops += 1;
                        v as f32
                    }
                }
            })
            .collect();

        // Estimate memory savings: f64 (8 bytes) -> f32 (4 bytes) per element
        self.stats.memory_savings_bytes += (data.len() * 4) as u64;

        // Estimate speedup (fp16/bf16 ops are ~2x faster on supported hardware)
        let total = self.stats.operations_count as f64;
        let fast_ops = (self.stats.fp16_ops + self.stats.bf16_ops) as f64;
        if total > 0.0 {
            let fast_ratio = fast_ops / total;
            self.stats.total_speedup_estimate = 1.0 + fast_ratio;
        }

        Ok(result)
    }

    /// Upcast f32 data back to f64.
    pub fn restore_precision(&mut self, data: &[f32]) -> Vec<f64> {
        self.stats.operations_count += data.len() as u64;
        data.iter().map(|&v| v as f64).collect()
    }

    /// Apply loss scaling to the given loss value.
    pub fn scale_loss(&self, loss: f64) -> f64 {
        if !self.config.enabled || !self.config.dynamic_loss_scaling {
            return loss;
        }
        loss * self.current_loss_scale
    }

    /// Reverse loss scaling on gradients (divide by current loss scale).
    pub fn unscale_gradients(&self, gradients: &mut [f64]) {
        if !self.config.enabled || !self.config.dynamic_loss_scaling {
            return;
        }
        let inv_scale = 1.0 / self.current_loss_scale;
        for g in gradients.iter_mut() {
            *g *= inv_scale;
        }
    }

    /// Update loss scale after an optimizer step.
    ///
    /// If no overflow has occurred for `loss_scale_window` consecutive steps,
    /// the loss scale is increased. If overflow is detected, the scale is
    /// reduced by `loss_scale_backoff`.
    pub fn step_optimizer(&mut self) {
        if !self.config.enabled || !self.config.dynamic_loss_scaling {
            return;
        }

        self.steps_since_overflow += 1;

        if self.steps_since_overflow >= self.config.loss_scale_window {
            // Increase loss scale
            let new_scale = self.current_loss_scale * self.config.loss_scale_factor;
            self.current_loss_scale = new_scale.min(self.config.max_loss_scale);
            self.steps_since_overflow = 0;
        }
    }

    /// Record an overflow event and reduce the loss scale.
    pub fn record_overflow(&mut self) {
        if !self.config.dynamic_loss_scaling {
            return;
        }
        self.stats.overflow_count += 1;
        self.current_loss_scale =
            (self.current_loss_scale * self.config.loss_scale_backoff).max(self.config.min_loss_scale);
        self.steps_since_overflow = 0;
    }

    /// Check whether any values in the slice are inf or NaN.
    pub fn check_overflow(values: &[f64]) -> bool {
        values.iter().any(|v| v.is_nan() || v.is_infinite())
    }

    /// Auto-select the best precision mode based on the data range.
    ///
    /// - If all values fit within f16 range, use Half.
    /// - If values exceed f16 range but fit f32, use BFloat16 (wider range).
    /// - Otherwise, use Full precision.
    pub fn select_precision(&self, data: &[f64]) -> PrecisionMode {
        if data.is_empty() {
            return PrecisionMode::Full;
        }

        let half_max: f64 = 65504.0;
        let half_min_positive: f64 = 6.1e-5;

        let mut max_abs: f64 = 0.0;
        let mut min_abs_nonzero: f64 = f64::MAX;
        let mut has_nonzero = false;

        for &v in data {
            if v.is_nan() || v.is_infinite() {
                return PrecisionMode::Full;
            }
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
            if abs_v > 0.0 && abs_v < min_abs_nonzero {
                min_abs_nonzero = abs_v;
                has_nonzero = true;
            }
        }

        if max_abs <= half_max && (!has_nonzero || min_abs_nonzero >= half_min_positive) {
            PrecisionMode::Half
        } else if max_abs <= f32::MAX as f64 {
            PrecisionMode::BFloat16
        } else {
            PrecisionMode::Full
        }
    }

    /// Benchmark different precision modes on the given data.
    ///
    /// Returns a map of mode name -> estimated throughput (elements/second).
    pub fn benchmark_precision(&mut self, data: &[f64]) -> HashMap<String, f64> {
        let mut results = HashMap::new();

        // Benchmark Full precision (f64 -> f64 copy)
        let start = Instant::now();
        let _full: Vec<f64> = data.to_vec();
        let elapsed = start.elapsed().as_secs_f64().max(1e-12);
        results.insert("Full".to_string(), data.len() as f64 / elapsed);

        // Benchmark Half (f64 -> f32 with clamping)
        let half_max: f64 = 65504.0;
        let start = Instant::now();
        let _half: Vec<f32> = data
            .iter()
            .map(|&v| {
                let clamped = v.clamp(-half_max, half_max);
                clamped as f32
            })
            .collect();
        let elapsed = start.elapsed().as_secs_f64().max(1e-12);
        results.insert("Half".to_string(), data.len() as f64 / elapsed);

        // Benchmark BFloat16 (f64 -> truncated f32)
        let start = Instant::now();
        let _bf16: Vec<f32> = data
            .iter()
            .map(|&v| {
                let f = v as f32;
                let bits = f.to_bits() & 0xFFFF_0000;
                f32::from_bits(bits)
            })
            .collect();
        let elapsed = start.elapsed().as_secs_f64().max(1e-12);
        results.insert("BFloat16".to_string(), data.len() as f64 / elapsed);

        // Benchmark simple f32 cast (Mixed/Auto baseline)
        let start = Instant::now();
        let _mixed: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let elapsed = start.elapsed().as_secs_f64().max(1e-12);
        results.insert("Mixed".to_string(), data.len() as f64 / elapsed);

        results
    }

    /// Get a reference to the current statistics.
    pub fn get_stats(&self) -> &MixedPrecisionStats {
        &self.stats
    }

    /// Reset all statistics to defaults.
    pub fn reset_stats(&mut self) {
        self.stats = MixedPrecisionStats::default();
    }

    /// Enable mixed precision.
    pub fn enable(&mut self) {
        self.config.enabled = true;
    }

    /// Disable mixed precision.
    pub fn disable(&mut self) {
        self.config.enabled = false;
    }

    /// Get the current dynamic loss scale.
    pub fn current_loss_scale(&self) -> f64 {
        self.current_loss_scale
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &MixedPrecisionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> MixedPrecisionManager {
        MixedPrecisionManager::new(MixedPrecisionConfig::default())
    }

    #[test]
    fn test_optimize_and_restore() {
        let mut mgr = default_manager();
        let data = vec![1.0, 2.5, -3.0, 0.0, 100.0];
        let optimized = mgr.optimize_precision(&data).unwrap();
        assert_eq!(optimized.len(), data.len());
        let restored = mgr.restore_precision(&optimized);
        for (orig, rest) in data.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 1e-4);
        }
    }

    #[test]
    fn test_loss_scaling() {
        let mgr = default_manager();
        let loss = 0.5;
        let scaled = mgr.scale_loss(loss);
        assert!((scaled - loss * 65536.0).abs() < 1e-6);
    }

    #[test]
    fn test_unscale_gradients() {
        let mgr = default_manager();
        let mut grads = vec![65536.0, 131072.0];
        mgr.unscale_gradients(&mut grads);
        assert!((grads[0] - 1.0).abs() < 1e-6);
        assert!((grads[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_check_overflow() {
        assert!(MixedPrecisionManager::check_overflow(&[1.0, f64::NAN]));
        assert!(MixedPrecisionManager::check_overflow(&[f64::INFINITY]));
        assert!(!MixedPrecisionManager::check_overflow(&[1.0, 2.0]));
    }

    #[test]
    fn test_select_precision_half() {
        let mgr = default_manager();
        let data = vec![1.0, -2.0, 100.0];
        assert_eq!(mgr.select_precision(&data), PrecisionMode::Half);
    }

    #[test]
    fn test_select_precision_full_on_overflow() {
        let mgr = default_manager();
        let data = vec![1.0, f64::NAN];
        assert_eq!(mgr.select_precision(&data), PrecisionMode::Full);
    }

    #[test]
    fn test_dynamic_loss_scale_increase() {
        let mut config = MixedPrecisionConfig::default();
        config.loss_scale_window = 2;
        let mut mgr = MixedPrecisionManager::new(config);
        let initial = mgr.current_loss_scale();
        mgr.step_optimizer();
        mgr.step_optimizer();
        assert!(mgr.current_loss_scale() > initial);
    }

    #[test]
    fn test_record_overflow_decreases_scale() {
        let mut mgr = default_manager();
        let initial = mgr.current_loss_scale();
        mgr.record_overflow();
        assert!(mgr.current_loss_scale() < initial);
    }

    #[test]
    fn test_enable_disable() {
        let mut mgr = default_manager();
        mgr.disable();
        assert!(!mgr.config().enabled);
        let scaled = mgr.scale_loss(1.0);
        assert!((scaled - 1.0).abs() < 1e-12);
        mgr.enable();
        assert!(mgr.config().enabled);
    }

    #[test]
    fn test_benchmark_returns_all_modes() {
        let mut mgr = default_manager();
        let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let results = mgr.benchmark_precision(&data);
        assert!(results.contains_key("Full"));
        assert!(results.contains_key("Half"));
        assert!(results.contains_key("BFloat16"));
        assert!(results.contains_key("Mixed"));
    }

    #[test]
    fn test_reset_stats() {
        let mut mgr = default_manager();
        let data = vec![1.0, 2.0, 3.0];
        let _ = mgr.optimize_precision(&data);
        assert!(mgr.get_stats().operations_count > 0);
        mgr.reset_stats();
        assert_eq!(mgr.get_stats().operations_count, 0);
    }
}
