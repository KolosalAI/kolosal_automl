//! Shared utilities for Python bindings

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use polars::prelude::*;

/// Convert Python DataFrame (pandas) to Polars DataFrame
pub fn python_df_to_polars(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let columns: Vec<String> = df.getattr("columns")?.call_method0("tolist")?.extract()?;
    
    let mut col_vec: Vec<Column> = Vec::with_capacity(columns.len());
    
    for col in &columns {
        let col_data = df.get_item(col)?;
        let dtype_str: String = col_data.getattr("dtype")?.call_method0("__str__")?.extract()?;
        
        if dtype_str.contains("float") {
            let values: Vec<Option<f64>> = col_data
                .call_method0("tolist")?
                .extract::<Vec<PyObject>>()?
                .into_iter()
                .map(|v| v.extract::<f64>(py).ok())
                .collect();
            col_vec.push(Series::new(col.as_str().into(), values).into());
        } else if dtype_str.contains("int") {
            let values: Vec<Option<i64>> = col_data
                .call_method0("tolist")?
                .extract::<Vec<PyObject>>()?
                .into_iter()
                .map(|v| v.extract::<i64>(py).ok())
                .collect();
            col_vec.push(Series::new(col.as_str().into(), values).into());
        } else if dtype_str.contains("bool") {
            let values: Vec<Option<bool>> = col_data
                .call_method0("tolist")?
                .extract::<Vec<PyObject>>()?
                .into_iter()
                .map(|v| v.extract::<bool>(py).ok())
                .collect();
            col_vec.push(Series::new(col.as_str().into(), values).into());
        } else {
            // Default to string/object
            let values: Vec<Option<String>> = col_data
                .call_method0("tolist")?
                .extract::<Vec<PyObject>>()?
                .into_iter()
                .map(|v| v.extract::<String>(py).ok())
                .collect();
            col_vec.push(Series::new(col.as_str().into(), values).into());
        }
    }
    
    DataFrame::new(col_vec).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert Polars DataFrame to Python DataFrame (pandas)
pub fn polars_to_python_df(py: Python<'_>, df: &DataFrame) -> PyResult<PyObject> {
    let pandas = py.import("pandas")?;
    let dict = pyo3::types::PyDict::new(py);
    
    for col in df.get_columns() {
        let name = col.name().to_string();
        let series = col.as_materialized_series();
        
        match series.dtype() {
            DataType::Float64 | DataType::Float32 => {
                let values: Vec<Option<f64>> = series
                    .f64()
                    .map(|ca| ca.into_iter().collect())
                    .or_else(|_| series.f32().map(|ca| ca.into_iter().map(|v| v.map(|x| x as f64)).collect()))
                    .unwrap_or_default();
                dict.set_item(&name, values)?;
            }
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                let values: Vec<Option<i64>> = series
                    .i64()
                    .map(|ca| ca.into_iter().collect())
                    .or_else(|_| series.i32().map(|ca| ca.into_iter().map(|v| v.map(|x| x as i64)).collect()))
                    .unwrap_or_default();
                dict.set_item(&name, values)?;
            }
            DataType::Boolean => {
                let values: Vec<Option<bool>> = series
                    .bool()
                    .map(|ca| ca.into_iter().collect())
                    .unwrap_or_default();
                dict.set_item(&name, values)?;
            }
            _ => {
                let values: Vec<String> = series
                    .str()
                    .map(|ca| ca.into_iter().map(|v| v.unwrap_or("").to_string()).collect())
                    .unwrap_or_else(|_| (0..series.len()).map(|i| format!("{:?}", series.get(i))).collect());
                dict.set_item(&name, values)?;
            }
        }
    }
    
    let result = pandas.call_method1("DataFrame", (dict,))?;
    Ok(result.into())
}

/// Convert Vec<f64> to numpy array
pub fn vec_to_numpy_1d<'py>(py: Python<'py>, data: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("array", (data,))?;
    Ok(array)
}

/// Convert Vec<Vec<f64>> to numpy 2D array
pub fn vec_to_numpy_2d<'py>(py: Python<'py>, data: Vec<Vec<f64>>) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("array", (data,))?;
    Ok(array)
}

/// Convert numpy array to Vec<f64>
pub fn numpy_to_vec_1d(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let flat = arr.call_method0("flatten")?;
    let list = flat.call_method0("tolist")?;
    list.extract()
}

/// Convert numpy 2D array to Vec<Vec<f64>>
pub fn numpy_to_vec_2d(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let list = arr.call_method0("tolist")?;
    list.extract()
}
