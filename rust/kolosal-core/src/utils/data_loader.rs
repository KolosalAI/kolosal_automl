//! Data loading utilities

use crate::error::{KolosalError, Result};
use polars::prelude::*;
use std::path::Path;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::sync::Arc;

/// Data loader for various file formats
pub struct DataLoader {
    /// Number of threads for parallel loading
    n_threads: Option<usize>,
    /// Low memory mode
    low_memory: bool,
    /// Chunk size for streaming
    chunk_size: Option<usize>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader {
    /// Create a new data loader
    pub fn new() -> Self {
        Self {
            n_threads: None,
            low_memory: false,
            chunk_size: None,
        }
    }

    /// Set number of threads
    pub fn with_n_threads(mut self, n: usize) -> Self {
        self.n_threads = Some(n);
        self
    }

    /// Enable low memory mode
    pub fn with_low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }

    /// Set chunk size for streaming
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Load a CSV file
    pub fn load_csv(&self, path: &str) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let mut reader = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(100))
            .into_reader_with_file_handle(file);

        reader.finish()
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Load a CSV file with specific options
    pub fn load_csv_with_options(
        &self,
        path: &str,
        delimiter: u8,
        has_header: bool,
        skip_rows: usize,
        _columns: Option<Vec<String>>,
    ) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let parse_opts = CsvParseOptions::default()
            .with_separator(delimiter);

        let reader = CsvReadOptions::default()
            .with_has_header(has_header)
            .with_skip_rows(skip_rows)
            .with_infer_schema_length(Some(100))
            .with_parse_options(parse_opts)
            .into_reader_with_file_handle(file);

        reader.finish()
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Load a Parquet file
    pub fn load_parquet(&self, path: &str) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let reader = ParquetReader::new(file);

        reader.finish()
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Load a Parquet file with specific columns
    pub fn load_parquet_columns(&self, path: &str, columns: Vec<String>) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        ParquetReader::new(file)
            .with_columns(Some(columns))
            .finish()
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Load a JSON file (line-delimited)
    pub fn load_json(&self, path: &str) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        JsonReader::new(file)
            .finish()
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Detect file format from extension and load
    pub fn load_auto(&self, path: &str) -> Result<DataFrame> {
        let path_lower = path.to_lowercase();
        
        if path_lower.ends_with(".csv") || path_lower.ends_with(".tsv") {
            let delimiter = if path_lower.ends_with(".tsv") { b'\t' } else { b',' };
            self.load_csv_with_options(path, delimiter, true, 0, None)
        } else if path_lower.ends_with(".parquet") || path_lower.ends_with(".pq") {
            self.load_parquet(path)
        } else if path_lower.ends_with(".json") || path_lower.ends_with(".jsonl") {
            self.load_json(path)
        } else {
            // Try CSV as default
            self.load_csv(path)
        }
    }

    /// Get file info without loading full data
    pub fn get_file_info(&self, path: &str) -> Result<FileInfo> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let file_size = metadata.len();
        
        // Quick row count for CSV
        let (n_rows, n_cols, columns) = if path.to_lowercase().ends_with(".csv") {
            let file = File::open(path)
                .map_err(|e| KolosalError::DataError(e.to_string()))?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();
            
            // Get header
            let header = lines.next()
                .transpose()
                .map_err(|e| KolosalError::DataError(e.to_string()))?
                .unwrap_or_default();
            
            let columns: Vec<String> = header.split(',')
                .map(|s| s.trim().to_string())
                .collect();
            
            let n_cols = columns.len();
            let n_rows = lines.count(); // Count remaining lines
            
            (Some(n_rows), Some(n_cols), Some(columns))
        } else {
            (None, None, None)
        };

        Ok(FileInfo {
            path: path.to_string(),
            file_size,
            n_rows,
            n_cols,
            columns,
        })
    }
}

/// File information
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: String,
    pub file_size: u64,
    pub n_rows: Option<usize>,
    pub n_cols: Option<usize>,
    pub columns: Option<Vec<String>>,
}

/// Chunked data reader for large files
pub struct ChunkedReader {
    path: String,
    chunk_size: usize,
    current_chunk: usize,
    total_rows: Option<usize>,
}

impl ChunkedReader {
    /// Create a new chunked reader
    pub fn new(path: &str, chunk_size: usize) -> Self {
        Self {
            path: path.to_string(),
            chunk_size,
            current_chunk: 0,
            total_rows: None,
        }
    }

    /// Read the next chunk
    pub fn next_chunk(&mut self) -> Result<Option<DataFrame>> {
        let skip_rows = self.current_chunk * self.chunk_size;
        
        let file = File::open(&self.path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        let df = CsvReadOptions::default()
            .with_has_header(self.current_chunk == 0)
            .with_skip_rows(if self.current_chunk > 0 { skip_rows + 1 } else { 0 })
            .with_n_rows(Some(self.chunk_size))
            .into_reader_with_file_handle(file)
            .finish()
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        if df.height() == 0 {
            return Ok(None);
        }

        self.current_chunk += 1;
        Ok(Some(df))
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }

    /// Get current chunk index
    pub fn current_chunk(&self) -> usize {
        self.current_chunk
    }
}

/// Save DataFrame to various formats
pub struct DataSaver;

impl DataSaver {
    /// Save to CSV
    pub fn save_csv(df: &mut DataFrame, path: &str) -> Result<()> {
        let mut file = File::create(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        CsvWriter::new(&mut file)
            .finish(df)
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }

    /// Save to Parquet
    pub fn save_parquet(df: &mut DataFrame, path: &str) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        ParquetWriter::new(file)
            .finish(df)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        Ok(())
    }

    /// Save to JSON
    pub fn save_json(df: &mut DataFrame, path: &str) -> Result<()> {
        let mut file = File::create(path)
            .map_err(|e| KolosalError::DataError(e.to_string()))?;

        JsonWriter::new(&mut file)
            .finish(df)
            .map_err(|e| KolosalError::DataError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "a,b,c").unwrap();
        writeln!(file, "1,2,3").unwrap();
        writeln!(file, "4,5,6").unwrap();
        writeln!(file, "7,8,9").unwrap();
        file
    }

    #[test]
    fn test_load_csv() {
        let file = create_test_csv();
        let loader = DataLoader::new();
        
        let df = loader.load_csv(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 3);
    }

    #[test]
    fn test_get_file_info() {
        let file = create_test_csv();
        let loader = DataLoader::new();
        
        let info = loader.get_file_info(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(info.n_rows, Some(3));
        assert_eq!(info.n_cols, Some(3));
        assert_eq!(info.columns.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_chunked_reader() {
        let file = create_test_csv();
        let mut reader = ChunkedReader::new(file.path().to_str().unwrap(), 2);
        
        let chunk1 = reader.next_chunk().unwrap();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().height(), 2);
        
        let chunk2 = reader.next_chunk().unwrap();
        assert!(chunk2.is_some());
    }

    #[test]
    fn test_save_csv() {
        let mut df = DataFrame::new(vec![
            Column::new("a".into(), &[1, 2, 3]),
            Column::new("b".into(), &[4, 5, 6]),
        ]).unwrap();

        let file = NamedTempFile::new().unwrap();
        DataSaver::save_csv(&mut df, file.path().to_str().unwrap()).unwrap();

        // Reload and verify
        let loader = DataLoader::new();
        let loaded = loader.load_csv(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(loaded.height(), 3);
        assert_eq!(loaded.width(), 2);
    }
}
