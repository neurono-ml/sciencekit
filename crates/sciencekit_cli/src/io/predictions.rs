//! Prediction and transformed-matrix writers.

use std::path::Path;

use crate::error::SKCliError;

/// The prediction and matrix output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKOutputFormat {
    /// Delimited text with a header row.
    Csv,
    /// A JSON array of numbers (predictions) or arrays (matrices).
    Json,
    /// Apache Parquet with one double column per feature.
    Parquet,
}

impl SKOutputFormat {
    /// Parse a configured or command-line format name.
    pub fn from_name(name: &str) -> Result<Self, SKCliError> {
        match name.to_ascii_lowercase().as_str() {
            "csv" => Ok(SKOutputFormat::Csv),
            "json" => Ok(SKOutputFormat::Json),
            "parquet" => Ok(SKOutputFormat::Parquet),
            other => Err(SKCliError::validation(
                "configuration.output_format",
                format!("`{other}` is not an output format; use `csv`, `json` or `parquet`"),
            )),
        }
    }

    /// Infer the format from the extension, defaulting to CSV.
    pub fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("json") => SKOutputFormat::Json,
            Some("parquet") => SKOutputFormat::Parquet,
            _ => SKOutputFormat::Csv,
        }
    }
}

fn sk_format_number(value: f64) -> String {
    format!("{value}")
}

fn sk_write_csv(path: &Path, names: &[String], matrix: &[Vec<f64>]) -> Result<(), SKCliError> {
    let mut text = String::new();
    text.push_str(&names.join(","));
    text.push('\n');
    for row in matrix {
        let values: Vec<String> = row.iter().map(|value| sk_format_number(*value)).collect();
        text.push_str(&values.join(","));
        text.push('\n');
    }
    std::fs::write(path, text)
        .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string()))
}

#[cfg(feature = "parquet")]
fn sk_transpose(matrix: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    (0..width)
        .map(|column| matrix.iter().map(|row| row[column]).collect())
        .collect()
}

fn sk_write_matrix(
    path: &Path,
    format: SKOutputFormat,
    names: &[String],
    matrix: &[Vec<f64>],
) -> Result<(), SKCliError> {
    match format {
        SKOutputFormat::Csv => sk_write_csv(path, names, matrix),
        SKOutputFormat::Json => {
            let text = serde_json::to_string(matrix).map_err(|error| {
                SKCliError::runtime(format!("predictions serialization: {error}"))
            })?;
            std::fs::write(path, text)
                .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string()))
        }
        SKOutputFormat::Parquet => {
            #[cfg(feature = "parquet")]
            {
                let columns = sk_transpose(matrix, names.len());
                super::predictions_parquet::sk_write_parquet_columns(path, names, &columns)
            }
            #[cfg(not(feature = "parquet"))]
            {
                Err(SKCliError::unsupported(
                    "parquet",
                    "a Parquet output artifact",
                ))
            }
        }
    }
}

/// Write a prediction vector with the requested format.
pub fn sk_write_predictions(
    path: &Path,
    format: SKOutputFormat,
    target_name: &str,
    predictions: &[f64],
) -> Result<(), SKCliError> {
    let matrix: Vec<Vec<f64>> = predictions.iter().map(|value| vec![*value]).collect();
    sk_write_matrix(path, format, &[target_name.to_string()], &matrix)
}

/// Write a transformed feature matrix with the requested format.
pub fn sk_write_transformed_matrix(
    path: &Path,
    format: SKOutputFormat,
    feature_names: &[String],
    matrix: &[Vec<f64>],
) -> Result<(), SKCliError> {
    sk_write_matrix(path, format, feature_names, matrix)
}
