//! Quantization Module
//!
//! Provides data quantization for reduced memory footprint
//! and faster inference.

mod quantizer;
mod calibration;

pub use quantizer::{Quantizer, QuantizationConfig, QuantizationType, QuantizationMode, QuantizedData};
pub use calibration::{CalibrationMethod, QuantizationCalibrator};
