//! Dataset adapters over the interop tabular sources.

use std::path::Path;

use ndarray::{Array1, Array2};
use sciencekit::interop::columns::SKColumnData;

use crate::error::SKCliError;

/// A materialized dataset: numeric features, optional targets and names.
#[derive(Debug, Clone)]
pub struct SKDataset {
    /// The feature matrix, one row per observation.
    pub features: Array2<f64>,
    /// The target vector, when the dataset carries a target column.
    pub targets: Option<Array1<f64>>,
    /// The feature column names, in matrix order.
    pub feature_names: Vec<String>,
    /// The target column name, when present.
    pub target_name: Option<String>,
}

impl SKDataset {
    /// The number of observations.
    pub fn number_of_rows(&self) -> usize {
        self.features.nrows()
    }

    /// The number of feature columns.
    pub fn number_of_features(&self) -> usize {
        self.features.ncols()
    }
}

/// The dataset file formats the adapter recognizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKDatasetFormat {
    /// Delimited text read through the interop Arrow adapter.
    Csv,
    /// Apache Parquet read through the interop Arrow adapter.
    Parquet,
}

impl SKDatasetFormat {
    /// Parse a configured format name.
    pub fn from_name(name: &str) -> Result<Self, SKCliError> {
        match name.to_ascii_lowercase().as_str() {
            "csv" => Ok(SKDatasetFormat::Csv),
            "parquet" => Ok(SKDatasetFormat::Parquet),
            other => Err(SKCliError::validation(
                "configuration.dataset_format",
                format!("`{other}` is not a supported dataset format; use `csv` or `parquet`"),
            )),
        }
    }

    /// Infer the format from the file extension, defaulting to CSV.
    pub fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("parquet") => SKDatasetFormat::Parquet,
            _ => SKDatasetFormat::Csv,
        }
    }
}

fn sk_numeric_value(
    column: &SKColumnData,
    row: usize,
    column_name: &str,
    path: &str,
) -> Result<f64, SKCliError> {
    let missing = || {
        SKCliError::validation(
            format!("{path}.row[{row}].{column_name}"),
            "missing values are not supported by this adapter",
        )
    };
    match column {
        SKColumnData::Float32(values) => values.get(row).copied().flatten().map(f64::from),
        SKColumnData::Float64(values) => values.get(row).copied().flatten(),
        SKColumnData::Integer64(values) => values.get(row).copied().flatten().map(|v| v as f64),
        SKColumnData::Boolean(values) => values
            .get(row)
            .copied()
            .flatten()
            .map(|value| if value { 1.0 } else { 0.0 }),
        SKColumnData::Nominal(_) | SKColumnData::CategoryCode(_) => {
            return Err(SKCliError::validation(
                format!("{path}.{column_name}"),
                "nominal columns cannot be used as numeric features",
            ));
        }
    }
    .ok_or_else(missing)
}

#[cfg(any(feature = "csv", feature = "parquet"))]
fn sk_read_arrow(
    path: &Path,
    format: SKDatasetFormat,
    target: Option<&str>,
) -> Result<SKDataset, SKCliError> {
    use sciencekit::interop::arrow::{SKArrowFileFormat, SKArrowFileSource};
    use sciencekit::interop::source::SKTabularSource;

    let arrow_format = match format {
        SKDatasetFormat::Csv => SKArrowFileFormat::Csv,
        SKDatasetFormat::Parquet => SKArrowFileFormat::Parquet,
    };
    let path_text = path.display().to_string();
    let mut source = SKArrowFileSource::new(path, arrow_format).with_batch_size(1_048_576);
    source
        .open()
        .map_err(|error| SKCliError::data(&path_text, error.to_string()))?;

    let mut feature_names: Option<Vec<String>> = None;
    let mut feature_positions: Vec<usize> = Vec::new();
    let mut target_position: Option<usize> = None;
    let mut rows: Vec<f64> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();
    let mut number_of_rows = 0_usize;

    while let Some(batch) = source
        .next_batch()
        .map_err(|error| SKCliError::data(&path_text, error.to_string()))?
    {
        if feature_names.is_none() {
            let names: Vec<String> = batch
                .schema()
                .columns()
                .iter()
                .map(|column| column.name().to_string())
                .collect();
            target_position = match target {
                Some(name) => Some(batch.schema().column_index(name).ok_or_else(|| {
                    SKCliError::validation(
                        "configuration.target",
                        format!(
                            "the dataset does not carry the target column `{name}`; it carries: {}",
                            names.join(", ")
                        ),
                    )
                })?),
                None => None,
            };
            feature_positions = (0..names.len())
                .filter(|position| Some(*position) != target_position)
                .collect();
            feature_names = Some(
                feature_positions
                    .iter()
                    .map(|position| names[*position].clone())
                    .collect(),
            );
        }

        let names = feature_names
            .as_ref()
            .expect("feature names are initialized");
        for row in 0..batch.number_of_rows() {
            for (position, name) in feature_positions.iter().zip(names.iter()) {
                let column = batch.column(*position).ok_or_else(|| {
                    SKCliError::validation(path_text.clone(), "a batch column is missing")
                })?;
                rows.push(sk_numeric_value(column, row, name, &path_text)?);
            }
            if let Some(position) = target_position {
                let column = batch.column(position).ok_or_else(|| {
                    SKCliError::validation(path_text.clone(), "the target column is missing")
                })?;
                targets.push(sk_numeric_value(column, row, "target", &path_text)?);
            }
        }
        number_of_rows += batch.number_of_rows();
    }

    if number_of_rows == 0 {
        return Err(SKCliError::validation(
            "configuration.dataset",
            format!("the dataset `{path_text}` carries no rows"),
        ));
    }
    let names = feature_names.expect("at least one batch was read");
    let features = Array2::from_shape_vec((number_of_rows, names.len()), rows)
        .map_err(|error| SKCliError::runtime(format!("feature matrix assembly: {error}")))?;
    let targets = target_position.map(|_| Array1::from_vec(targets));
    Ok(SKDataset {
        features,
        targets,
        feature_names: names,
        target_name: target.map(str::to_string),
    })
}

/// Read a dataset from disk under the configured target column.
pub fn sk_load_dataset(
    path: &Path,
    format: Option<&str>,
    target: Option<&str>,
) -> Result<SKDataset, SKCliError> {
    let format = match format {
        Some(name) => SKDatasetFormat::from_name(name)?,
        None => SKDatasetFormat::from_path(path),
    };

    match format {
        SKDatasetFormat::Csv => {
            #[cfg(feature = "csv")]
            {
                sk_read_arrow(path, format, target)
            }
            #[cfg(not(feature = "csv"))]
            {
                Err(SKCliError::unsupported("csv", "a CSV dataset"))
            }
        }
        SKDatasetFormat::Parquet => {
            #[cfg(feature = "parquet")]
            {
                sk_read_arrow(path, format, target)
            }
            #[cfg(not(feature = "parquet"))]
            {
                Err(SKCliError::unsupported("parquet", "a Parquet dataset"))
            }
        }
    }
}
