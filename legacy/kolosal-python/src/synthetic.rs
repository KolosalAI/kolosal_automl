//! Python bindings for synthetic data generation and oversampling

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::prelude::*;

/// SMOTE for Python
#[pyclass(name = "SMOTE")]
pub struct PySMOTE {
    k_neighbors: usize,
    sampling_strategy: f64,
    seed: Option<u64>,
}

#[pymethods]
impl PySMOTE {
    #[new]
    #[pyo3(signature = (k_neighbors=5, sampling_strategy=1.0, random_state=None))]
    fn new(k_neighbors: usize, sampling_strategy: f64, random_state: Option<u64>) -> Self {
        Self {
            k_neighbors,
            sampling_strategy,
            seed: random_state,
        }
    }

    /// Fit and resample
    fn fit_resample(
        &self,
        py: Python<'_>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let y_list: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        // Find minority class
        let class_counts = self.count_classes(&y_list);
        let (&minority_class, &minority_count) = class_counts.iter()
            .min_by_key(|(_, c)| *c)
            .unwrap();
        let (&_majority_class, &majority_count) = class_counts.iter()
            .max_by_key(|(_, c)| *c)
            .unwrap();
        
        let minority_class_f64 = minority_class as f64;
        
        // Calculate how many samples to generate
        let target_count = ((majority_count as f64) * self.sampling_strategy) as usize;
        let n_to_generate = target_count.saturating_sub(minority_count);
        
        if n_to_generate == 0 {
            let numpy = py.import("numpy")?;
            let x_arr = numpy.call_method1("array", (x_list,))?;
            let y_arr = numpy.call_method1("array", (y_list,))?;
            return Ok((x_arr.into(), y_arr.into()));
        }
        
        // Get minority samples
        let minority_indices: Vec<usize> = y_list.iter()
            .enumerate()
            .filter(|(_, &c)| (c - minority_class_f64).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect();
        
        let minority_samples: Vec<&Vec<f64>> = minority_indices.iter()
            .map(|&i| &x_list[i])
            .collect();
        
        // Generate synthetic samples
        let mut synthetic_X: Vec<Vec<f64>> = Vec::with_capacity(n_to_generate);
        let mut synthetic_y: Vec<f64> = Vec::with_capacity(n_to_generate);
        
        for _ in 0..n_to_generate {
            let idx = rng.gen_range(0..minority_samples.len());
            let sample = minority_samples[idx];
            
            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = minority_samples.iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(i, s)| {
                    let dist: f64 = sample.iter().zip(s.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (i, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let k = self.k_neighbors.min(distances.len());
            if k == 0 {
                continue;
            }
            
            let neighbor_idx = distances[rng.gen_range(0..k)].0;
            let neighbor = minority_samples[neighbor_idx];
            
            // Interpolate
            let alpha: f64 = rng.gen();
            let new_sample: Vec<f64> = sample.iter().zip(neighbor.iter())
                .map(|(a, b)| a + alpha * (b - a))
                .collect();
            
            synthetic_X.push(new_sample);
            synthetic_y.push(minority_class_f64);
        }
        
        // Combine original and synthetic
        let mut result_X = x_list;
        let mut result_y = y_list;
        result_X.extend(synthetic_X);
        result_y.extend(synthetic_y);
        
        let numpy = py.import("numpy")?;
        let x_arr = numpy.call_method1("array", (result_X,))?;
        let y_arr = numpy.call_method1("array", (result_y,))?;
        
        Ok((x_arr.into(), y_arr.into()))
    }

    fn k_neighbors(&self) -> usize {
        self.k_neighbors
    }
}

impl PySMOTE {
    fn count_classes(&self, y: &[f64]) -> std::collections::HashMap<i64, usize> {
        let mut counts = std::collections::HashMap::new();
        for &c in y {
            *counts.entry(c as i64).or_insert(0) += 1;
        }
        // Convert back to f64 keys
        counts.into_iter().map(|(k, v)| (k, v)).collect::<std::collections::HashMap<i64, usize>>()
            .into_iter()
            .map(|(k, v)| (k as f64, v))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(k, v)| (k as i64, v))
            .collect()
    }
}

/// ADASYN for Python
#[pyclass(name = "ADASYN")]
pub struct PyADASYN {
    n_neighbors: usize,
    sampling_strategy: f64,
    seed: Option<u64>,
}

#[pymethods]
impl PyADASYN {
    #[new]
    #[pyo3(signature = (n_neighbors=5, sampling_strategy=1.0, random_state=None))]
    fn new(n_neighbors: usize, sampling_strategy: f64, random_state: Option<u64>) -> Self {
        Self {
            n_neighbors,
            sampling_strategy,
            seed: random_state,
        }
    }

    /// Fit and resample (simplified ADASYN)
    fn fit_resample(
        &self,
        py: Python<'_>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        // For simplicity, delegate to SMOTE-like behavior
        let smote = PySMOTE::new(self.n_neighbors, self.sampling_strategy, self.seed);
        smote.fit_resample(py, X, y)
    }

    fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }
}

/// Random Over Sampler for Python
#[pyclass(name = "RandomOverSampler")]
pub struct PyRandomOverSampler {
    sampling_strategy: f64,
    seed: Option<u64>,
}

#[pymethods]
impl PyRandomOverSampler {
    #[new]
    #[pyo3(signature = (sampling_strategy=1.0, random_state=None))]
    fn new(sampling_strategy: f64, random_state: Option<u64>) -> Self {
        Self {
            sampling_strategy,
            seed: random_state,
        }
    }

    /// Fit and resample by random duplication
    fn fit_resample(
        &self,
        py: Python<'_>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let y_list: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        // Find minority class
        let mut class_counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for &c in &y_list {
            *class_counts.entry(c as i64).or_insert(0) += 1;
        }
        
        let minority_class = *class_counts.iter().min_by_key(|(_, c)| *c).unwrap().0;
        let majority_count = *class_counts.values().max().unwrap();
        
        let target = (majority_count as f64 * self.sampling_strategy) as usize;
        let minority_count = class_counts[&minority_class];
        let n_to_add = target.saturating_sub(minority_count);
        
        // Get minority indices
        let minority_indices: Vec<usize> = y_list.iter()
            .enumerate()
            .filter(|(_, &c)| c as i64 == minority_class)
            .map(|(i, _)| i)
            .collect();
        
        let mut result_X = x_list;
        let mut result_y = y_list;
        
        for _ in 0..n_to_add {
            let idx = minority_indices[rng.gen_range(0..minority_indices.len())];
            result_X.push(result_X[idx].clone());
            result_y.push(minority_class as f64);
        }
        
        let numpy = py.import("numpy")?;
        let x_arr = numpy.call_method1("array", (result_X,))?;
        let y_arr = numpy.call_method1("array", (result_y,))?;
        
        Ok((x_arr.into(), y_arr.into()))
    }
}

/// Random Under Sampler for Python
#[pyclass(name = "RandomUnderSampler")]
pub struct PyRandomUnderSampler {
    sampling_strategy: f64,
    seed: Option<u64>,
}

#[pymethods]
impl PyRandomUnderSampler {
    #[new]
    #[pyo3(signature = (sampling_strategy=1.0, random_state=None))]
    fn new(sampling_strategy: f64, random_state: Option<u64>) -> Self {
        Self {
            sampling_strategy,
            seed: random_state,
        }
    }

    /// Fit and resample by random removal
    fn fit_resample(
        &self,
        py: Python<'_>,
        X: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        let y_list: Vec<f64> = y.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        // Find majority class
        let mut class_counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for &c in &y_list {
            *class_counts.entry(c as i64).or_insert(0) += 1;
        }
        
        let majority_class = *class_counts.iter().max_by_key(|(_, c)| *c).unwrap().0;
        let minority_count = *class_counts.values().min().unwrap();
        
        let target = (minority_count as f64 / self.sampling_strategy) as usize;
        
        // Get majority indices and shuffle
        let mut majority_indices: Vec<usize> = y_list.iter()
            .enumerate()
            .filter(|(_, &c)| c as i64 == majority_class)
            .map(|(i, _)| i)
            .collect();
        majority_indices.shuffle(&mut rng);
        
        // Keep only target number of majority samples
        let keep_indices: std::collections::HashSet<usize> = majority_indices
            .into_iter()
            .take(target)
            .collect();
        
        let mut result_X = Vec::new();
        let mut result_y = Vec::new();
        
        for (i, (x, y)) in x_list.iter().zip(y_list.iter()).enumerate() {
            if *y as i64 != majority_class || keep_indices.contains(&i) {
                result_X.push(x.clone());
                result_y.push(*y);
            }
        }
        
        let numpy = py.import("numpy")?;
        let x_arr = numpy.call_method1("array", (result_X,))?;
        let y_arr = numpy.call_method1("array", (result_y,))?;
        
        Ok((x_arr.into(), y_arr.into()))
    }
}
