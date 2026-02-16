//! SMOTE and variants

use crate::error::{KolosalError, Result};
use crate::synthetic::{Sampler, ResampleResult, class_counts, class_indices};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::cmp::Ordering;

/// Ordered float for BinaryHeap-based partial sort
#[derive(Debug, Clone, Copy)]
struct DistIdx(f64, usize);

impl PartialEq for DistIdx {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}
impl Eq for DistIdx {}
impl PartialOrd for DistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for DistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// SMOTE variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SMOTEVariant {
    /// Standard SMOTE
    Regular,
    /// Borderline SMOTE (type 1)
    Borderline1,
    /// Borderline SMOTE (type 2)  
    Borderline2,
}

/// SMOTE (Synthetic Minority Over-sampling Technique)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMOTE {
    /// Number of nearest neighbors
    k_neighbors: usize,
    /// Sampling strategy (ratio of minority to majority)
    sampling_strategy: f64,
    /// Random seed
    seed: Option<u64>,
    /// Target samples per class
    target_counts: Option<HashMap<i64, usize>>,
}

impl SMOTE {
    /// Create new SMOTE sampler
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            sampling_strategy: 1.0, // Balance classes
            seed: None,
            target_counts: None,
        }
    }

    /// Set number of neighbors
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.max(1);
        self
    }

    /// Set sampling strategy (ratio)
    pub fn with_sampling_strategy(mut self, ratio: f64) -> Self {
        self.sampling_strategy = ratio.clamp(0.1, 10.0);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Euclidean distance
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Find k nearest neighbors using BinaryHeap (O(n log k) instead of O(n log n))
    fn find_neighbors(&self, point: &[f64], data: &[Vec<f64>], k: usize) -> Vec<usize> {
        let mut heap: BinaryHeap<DistIdx> = BinaryHeap::with_capacity(k + 1);

        for (i, d) in data.iter().enumerate() {
            let dist = Self::distance(point, d);
            if dist <= 0.0 {
                continue; // Exclude self
            }
            if heap.len() < k {
                heap.push(DistIdx(dist, i));
            } else if let Some(&DistIdx(max_dist, _)) = heap.peek() {
                if dist < max_dist {
                    heap.pop();
                    heap.push(DistIdx(dist, i));
                }
            }
        }

        heap.into_iter().map(|DistIdx(_, i)| i).collect()
    }

    /// Generate synthetic sample between two points
    fn generate_sample(&self, point: &[f64], neighbor: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let gap: f64 = rng.gen();
        point.iter()
            .zip(neighbor.iter())
            .map(|(&p, &n)| p + gap * (n - p))
            .collect()
    }
}

impl Default for SMOTE {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SMOTE {
    fn fit(&mut self, _x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        let counts = class_counts(y);
        
        if counts.len() < 2 {
            return Err(KolosalError::ValidationError(
                "Need at least 2 classes for SMOTE".to_string()
            ));
        }

        // Find majority class count
        let max_count = *counts.values().max().unwrap();

        // Calculate target counts
        let mut targets = HashMap::new();
        for (&class, &count) in &counts {
            let target = (max_count as f64 * self.sampling_strategy) as usize;
            targets.insert(class, target.max(count));
        }

        self.target_counts = Some(targets);
        Ok(())
    }

    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        let targets = self.target_counts.as_ref().ok_or_else(|| {
            KolosalError::ValidationError("SMOTE not fitted".to_string())
        })?;

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let indices = class_indices(y);
        let counts = class_counts(y);
        let n_features = x.ncols();

        // Collect only synthetic samples (original data reused from x directly)
        let mut synthetic_x: Vec<Vec<f64>> = Vec::new();
        let mut synthetic_y: Vec<i64> = Vec::new();
        let mut n_synthetic = Vec::new();

        for (&class, &target_count) in targets {
            let current_count = counts.get(&class).copied().unwrap_or(0);
            let n_to_generate = target_count.saturating_sub(current_count);

            if n_to_generate == 0 {
                n_synthetic.push(0);
                continue;
            }

            // Get samples for this class
            let class_idx = indices.get(&class).unwrap();
            let class_samples: Vec<Vec<f64>> = class_idx.iter()
                .map(|&i| x.row(i).iter().copied().collect())
                .collect();

            let k = self.k_neighbors.min(class_samples.len() - 1).max(1);

            // Generate synthetic samples
            let mut generated = 0;
            while generated < n_to_generate {
                let idx = rng.gen_range(0..class_samples.len());
                let sample = &class_samples[idx];

                let neighbors = self.find_neighbors(sample, &class_samples, k);

                if neighbors.is_empty() {
                    continue;
                }

                let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                let neighbor = &class_samples[neighbor_idx];

                synthetic_x.push(self.generate_sample(sample, neighbor, &mut rng));
                synthetic_y.push(class);
                generated += 1;
            }

            n_synthetic.push(n_to_generate);
        }

        // Build result: original rows + synthetic rows using from_shape_fn
        let n_original = x.nrows();
        let n_total = n_original + synthetic_x.len();
        let result_x = Array2::from_shape_fn((n_total, n_features), |(i, j)| {
            if i < n_original {
                x[[i, j]]
            } else {
                synthetic_x[i - n_original][j]
            }
        });

        let mut all_y: Vec<i64> = y.iter().copied().collect();
        all_y.extend_from_slice(&synthetic_y);
        let result_y = Array1::from_vec(all_y);

        Ok(ResampleResult {
            x: result_x,
            y: result_y,
            n_synthetic,
        })
    }
}

/// Borderline SMOTE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderlineSMOTE {
    /// Base SMOTE
    smote: SMOTE,
    /// Variant type
    variant: SMOTEVariant,
    /// Number of neighbors for borderline detection
    m_neighbors: usize,
}

impl BorderlineSMOTE {
    /// Create new Borderline SMOTE
    pub fn new(variant: SMOTEVariant) -> Self {
        Self {
            smote: SMOTE::new(),
            variant,
            m_neighbors: 10,
        }
    }

    /// Set k neighbors for SMOTE
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.smote = self.smote.with_k_neighbors(k);
        self
    }

    /// Set m neighbors for borderline detection
    pub fn with_m_neighbors(mut self, m: usize) -> Self {
        self.m_neighbors = m.max(1);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.smote = self.smote.with_seed(seed);
        self
    }

    /// Check if a point is borderline.
    /// Uses BinaryHeap for O(n log m) partial sort instead of O(n log n) full sort.
    /// Operates on Array2 row views directly — no per-call Vec<Vec<f64>> allocation.
    fn is_borderline(&self, point_idx: usize, x: &Array2<f64>, y: &Array1<i64>) -> bool {
        let point = x.row(point_idx);
        let point_slice = point.as_slice().unwrap();
        let point_class = y[point_idx];
        let m = self.m_neighbors;

        // Find m nearest neighbors using BinaryHeap
        let mut heap: BinaryHeap<DistIdx> = BinaryHeap::with_capacity(m + 1);
        for (i, row) in x.rows().into_iter().enumerate() {
            if i == point_idx {
                continue;
            }
            let dist = SMOTE::distance(point_slice, row.as_slice().unwrap());
            if heap.len() < m {
                heap.push(DistIdx(dist, i));
            } else if let Some(&DistIdx(max_dist, _)) = heap.peek() {
                if dist < max_dist {
                    heap.pop();
                    heap.push(DistIdx(dist, i));
                }
            }
        }

        // Count neighbors from different class
        let n_different = heap.into_iter()
            .filter(|&DistIdx(_, i)| y[i] != point_class)
            .count();

        let ratio = n_different as f64 / m as f64;
        ratio > 0.3 && ratio < 0.7
    }
}

impl Sampler for BorderlineSMOTE {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        self.smote.fit(x, y)
    }

    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        let targets = self.smote.target_counts.as_ref().ok_or_else(|| {
            KolosalError::ValidationError("Borderline SMOTE not fitted".to_string())
        })?;

        let mut rng = match self.smote.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let indices = class_indices(y);
        let counts = class_counts(y);
        let n_features = x.ncols();

        // Find minority classes
        let max_count = *counts.values().max().unwrap();
        let minority_classes: Vec<i64> = counts.iter()
            .filter(|(_, &c)| c < max_count)
            .map(|(&k, _)| k)
            .collect();

        // Find borderline samples for minority classes
        let mut borderline_indices: HashMap<i64, Vec<usize>> = HashMap::new();
        for &class in &minority_classes {
            let class_idx = indices.get(&class).unwrap();
            let borderline: Vec<usize> = class_idx.iter()
                .filter(|&&i| self.is_borderline(i, x, y))
                .copied()
                .collect();
            borderline_indices.insert(class, borderline);
        }

        // Collect only synthetic samples
        let mut synthetic_x: Vec<Vec<f64>> = Vec::new();
        let mut synthetic_y: Vec<i64> = Vec::new();
        let mut n_synthetic = Vec::new();

        // Generate synthetic samples only from borderline samples
        for (&class, &target_count) in targets {
            let current_count = counts.get(&class).copied().unwrap_or(0);
            let n_to_generate = target_count.saturating_sub(current_count);

            if n_to_generate == 0 {
                n_synthetic.push(0);
                continue;
            }

            let class_idx = indices.get(&class).unwrap();
            let borderline = borderline_indices.get(&class)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let source_indices = if borderline.is_empty() {
                class_idx.as_slice()
            } else {
                borderline
            };

            let class_samples: Vec<Vec<f64>> = class_idx.iter()
                .map(|&i| x.row(i).iter().copied().collect())
                .collect();

            let k = self.smote.k_neighbors.min(class_samples.len() - 1).max(1);

            let mut generated = 0;
            while generated < n_to_generate {
                let idx = source_indices[rng.gen_range(0..source_indices.len())];
                let sample: Vec<f64> = x.row(idx).iter().copied().collect();

                let neighbors = self.smote.find_neighbors(&sample, &class_samples, k);

                if neighbors.is_empty() {
                    continue;
                }

                let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                let neighbor = &class_samples[neighbor_idx];

                synthetic_x.push(self.smote.generate_sample(&sample, neighbor, &mut rng));
                synthetic_y.push(class);
                generated += 1;
            }

            n_synthetic.push(n_to_generate);
        }

        // Build result: original rows + synthetic rows
        let n_original = x.nrows();
        let n_total = n_original + synthetic_x.len();
        let result_x = Array2::from_shape_fn((n_total, n_features), |(i, j)| {
            if i < n_original {
                x[[i, j]]
            } else {
                synthetic_x[i - n_original][j]
            }
        });

        let mut all_y: Vec<i64> = y.iter().copied().collect();
        all_y.extend_from_slice(&synthetic_y);

        Ok(ResampleResult {
            x: result_x,
            y: Array1::from_vec(all_y),
            n_synthetic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_imbalanced_data() -> (Array2<f64>, Array1<i64>) {
        // Create imbalanced dataset: 20 majority, 5 minority
        let mut data = Vec::new();
        let mut labels = Vec::new();

        // Majority class (0) around (0, 0)
        for i in 0..20 {
            data.push((i % 5) as f64);
            data.push((i / 5) as f64);
            labels.push(0i64);
        }

        // Minority class (1) around (10, 10)
        for i in 0..5 {
            data.push(10.0 + (i % 3) as f64);
            data.push(10.0 + (i / 3) as f64);
            labels.push(1i64);
        }

        let x = Array2::from_shape_vec((25, 2), data).unwrap();
        let y = Array1::from_vec(labels);

        (x, y)
    }

    #[test]
    fn test_smote_basic() {
        let (x, y) = create_imbalanced_data();
        
        let mut smote = SMOTE::new()
            .with_k_neighbors(3)
            .with_seed(42);

        let result = smote.fit_resample(&x, &y).unwrap();

        // Check that we have more samples
        assert!(result.x.nrows() > x.nrows());
        assert!(result.y.len() > y.len());

        // Check class balance improved
        let new_counts = class_counts(&result.y);
        let count_0 = new_counts.get(&0).copied().unwrap_or(0);
        let count_1 = new_counts.get(&1).copied().unwrap_or(0);
        
        // Classes should be more balanced
        assert!(count_1 > 5); // More minority samples
    }

    #[test]
    fn test_smote_preserves_original() {
        let (x, y) = create_imbalanced_data();
        let original_rows = x.nrows();
        
        let mut smote = SMOTE::new().with_seed(42);
        let result = smote.fit_resample(&x, &y).unwrap();

        // First rows should be original data
        for i in 0..original_rows {
            for j in 0..x.ncols() {
                assert_eq!(result.x[[i, j]], x[[i, j]]);
            }
        }
    }

    #[test]
    fn test_borderline_smote() {
        let (x, y) = create_imbalanced_data();
        
        let mut bsmote = BorderlineSMOTE::new(SMOTEVariant::Borderline1)
            .with_k_neighbors(3)
            .with_m_neighbors(5)
            .with_seed(42);

        let result = bsmote.fit_resample(&x, &y).unwrap();

        // Should have generated samples
        assert!(result.x.nrows() >= x.nrows());
    }
}
