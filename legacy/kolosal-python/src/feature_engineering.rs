//! Python bindings for feature engineering

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

/// Polynomial features generator for Python
#[pyclass(name = "PolynomialFeatures")]
pub struct PyPolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
}

#[pymethods]
impl PyPolynomialFeatures {
    #[new]
    #[pyo3(signature = (degree=2, include_bias=true, interaction_only=false))]
    fn new(degree: usize, include_bias: bool, interaction_only: bool) -> Self {
        Self { degree, include_bias, interaction_only }
    }

    /// Transform data to polynomial features
    fn fit_transform(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let n_samples = x_list.len();
        let n_features = x_list[0].len();
        
        let mut result: Vec<Vec<f64>> = vec![Vec::new(); n_samples];
        
        // Add bias
        if self.include_bias {
            for row in &mut result {
                row.push(1.0);
            }
        }
        
        // Add original features (degree 1)
        for i in 0..n_samples {
            for j in 0..n_features {
                result[i].push(x_list[i][j]);
            }
        }
        
        // Add degree 2 features
        if self.degree >= 2 {
            for i in 0..n_features {
                let start = if self.interaction_only { i + 1 } else { i };
                for j in start..n_features {
                    for s in 0..n_samples {
                        result[s].push(x_list[s][i] * x_list[s][j]);
                    }
                }
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    fn degree(&self) -> usize {
        self.degree
    }
}

/// Feature interactions generator for Python
#[pyclass(name = "FeatureInteractions")]
pub struct PyFeatureInteractions {
    interaction_pairs: Option<Vec<(usize, usize)>>,
}

#[pymethods]
impl PyFeatureInteractions {
    #[new]
    #[pyo3(signature = (pairs=None))]
    fn new(pairs: Option<Vec<(usize, usize)>>) -> Self {
        Self { interaction_pairs: pairs }
    }

    /// Create interaction features
    fn fit_transform(&self, py: Python<'_>, X: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_list: Vec<Vec<f64>> = X.call_method0("tolist")?.extract()?;
        
        if x_list.is_empty() {
            return Err(PyValueError::new_err("Empty data"));
        }
        
        let n_samples = x_list.len();
        let n_features = x_list[0].len();
        
        let pairs: Vec<(usize, usize)> = self.interaction_pairs.clone().unwrap_or_else(|| {
            let mut p = Vec::new();
            for i in 0..n_features {
                for j in (i+1)..n_features {
                    p.push((i, j));
                }
            }
            p
        });
        
        let mut result: Vec<Vec<f64>> = x_list.clone();
        
        for (i, j) in &pairs {
            for s in 0..n_samples {
                result[s].push(x_list[s][*i] * x_list[s][*j]);
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }
}

/// TF-IDF Vectorizer for Python
#[pyclass(name = "TfidfVectorizer")]
pub struct PyTfidfVectorizer {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    max_features: Option<usize>,
}

#[pymethods]
impl PyTfidfVectorizer {
    #[new]
    #[pyo3(signature = (max_features=None))]
    fn new(max_features: Option<usize>) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            max_features,
        }
    }

    /// Fit on documents
    fn fit(&mut self, documents: Vec<String>) -> PyResult<()> {
        let n_docs = documents.len() as f64;
        let mut term_doc_counts: HashMap<String, usize> = HashMap::new();
        
        for doc in &documents {
            let tokens: std::collections::HashSet<String> = doc.to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            for token in tokens {
                *term_doc_counts.entry(token).or_insert(0) += 1;
            }
        }
        
        let mut terms: Vec<(String, usize)> = term_doc_counts.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));
        
        if let Some(max) = self.max_features {
            terms.truncate(max);
        }
        
        self.vocabulary.clear();
        self.idf.clear();
        
        for (idx, (term, count)) in terms.into_iter().enumerate() {
            self.vocabulary.insert(term, idx);
            self.idf.push((n_docs / count as f64).ln() + 1.0);
        }
        
        Ok(())
    }

    /// Transform documents to TF-IDF matrix
    fn transform(&self, py: Python<'_>, documents: Vec<String>) -> PyResult<PyObject> {
        if self.vocabulary.is_empty() {
            return Err(PyValueError::new_err("Not fitted"));
        }
        
        let n_docs = documents.len();
        let n_terms = self.vocabulary.len();
        let mut result = vec![vec![0.0; n_terms]; n_docs];
        
        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens: Vec<String> = doc.to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let n_tokens = tokens.len() as f64;
            
            let mut tf: HashMap<String, f64> = HashMap::new();
            for token in &tokens {
                *tf.entry(token.clone()).or_insert(0.0) += 1.0;
            }
            
            for (token, count) in tf {
                if let Some(&idx) = self.vocabulary.get(&token) {
                    result[doc_idx][idx] = (count / n_tokens) * self.idf[idx];
                }
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    /// Fit and transform
    fn fit_transform(&mut self, py: Python<'_>, documents: Vec<String>) -> PyResult<PyObject> {
        self.fit(documents.clone())?;
        self.transform(py, documents)
    }

    fn vocabulary(&self) -> HashMap<String, usize> {
        self.vocabulary.clone()
    }
}

/// Feature hasher for Python
#[pyclass(name = "FeatureHasher")]
pub struct PyFeatureHasher {
    n_features: usize,
}

#[pymethods]
impl PyFeatureHasher {
    #[new]
    #[pyo3(signature = (n_features=1024))]
    fn new(n_features: usize) -> Self {
        Self { n_features }
    }

    /// Hash features to fixed-size vector
    fn transform(&self, py: Python<'_>, features: Vec<HashMap<String, f64>>) -> PyResult<PyObject> {
        let n_samples = features.len();
        let mut result = vec![vec![0.0; self.n_features]; n_samples];
        
        for (i, feat_dict) in features.iter().enumerate() {
            for (key, value) in feat_dict {
                let hash = self.simple_hash(key) % self.n_features;
                result[i][hash] += value;
            }
        }
        
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (result,))?;
        Ok(array.into())
    }

    fn n_features(&self) -> usize {
        self.n_features
    }
}

impl PyFeatureHasher {
    fn simple_hash(&self, s: &str) -> usize {
        s.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize))
    }
}
