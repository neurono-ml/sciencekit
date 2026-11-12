//! Conversion of a DataFusion `DataFrame` into materialized datasets (spec
//! `tabular-sources`).

use ::datafusion::dataframe::DataFrame;
use sciencekit_common::{SKError, SKFloat};
use tokio::runtime::{Builder, Runtime};

use super::batch_conversion::sk_tabular_batch_from_datafusion_record_batch;
use crate::batch::SKTabularBatch;
use crate::tabular::{SKTabularDataset, SKTabularSourceConfig};

/// Build the current-thread runtime the synchronous entry points use.
///
/// # Errors
/// Returns [`SKError::Conversion`] when Tokio cannot create a runtime.
pub fn sk_datafusion_runtime() -> Result<Runtime, SKError> {
    Builder::new_current_thread()
        .build()
        .map_err(|error| SKError::Conversion(format!("building the Tokio runtime failed: {error}")))
}

/// Collect a DataFusion `DataFrame` into neutral batches, preserving order.
///
/// # Errors
/// Propagates the DataFusion execution failure and the batch conversion
/// failures.
pub async fn sk_tabular_batches_from_dataframe(
    dataframe: DataFrame,
    config: &SKTabularSourceConfig,
) -> Result<Vec<SKTabularBatch>, SKError> {
    let record_batches = dataframe
        .collect()
        .await
        .map_err(|error| SKError::Conversion(format!("DataFusion collect failed: {error}")))?;
    let number_of_batches = record_batches.len();
    record_batches
        .iter()
        .enumerate()
        .map(|(position, record_batch)| {
            sk_tabular_batch_from_datafusion_record_batch(
                record_batch,
                config,
                position,
                position + 1 == number_of_batches,
            )
        })
        .collect()
}

/// Collect a DataFusion `DataFrame` into one owned materialized dataset.
///
/// # Errors
/// Propagates [`sk_tabular_batches_from_dataframe`] and the materialization
/// failures of the configured policy.
pub async fn sk_tabular_dataset_from_dataframe<F: SKFloat>(
    dataframe: DataFrame,
    config: &SKTabularSourceConfig,
) -> Result<SKTabularDataset<F>, SKError> {
    let batches = sk_tabular_batches_from_dataframe(dataframe, config).await?;
    sk_dataset_from_batches::<F>(batches, config)
}

/// Block on [`sk_tabular_dataset_from_dataframe`] with a fresh runtime.
///
/// # Errors
/// Returns [`SKError::UnsupportedOperation`] when called from inside a Tokio
/// runtime, because nesting runtimes panics, and propagates every conversion
/// failure.
pub fn sk_tabular_dataset_from_dataframe_blocking<F: SKFloat>(
    dataframe: DataFrame,
    config: &SKTabularSourceConfig,
) -> Result<SKTabularDataset<F>, SKError> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(SKError::UnsupportedOperation {
            operation: "datafusion_dataframe_blocking",
            reason: "use the async conversion inside a Tokio runtime".to_string(),
        });
    }
    sk_datafusion_runtime()?.block_on(sk_tabular_dataset_from_dataframe::<F>(
        dataframe, config,
    ))
}

impl<F: SKFloat> TryFrom<(&DataFrame, &SKTabularSourceConfig)> for SKTabularDataset<F> {
    type Error = SKError;

    fn try_from(
        (dataframe, config): (&DataFrame, &SKTabularSourceConfig),
    ) -> Result<Self, Self::Error> {
        sk_tabular_dataset_from_dataframe_blocking::<F>(dataframe.clone(), config)
    }
}

pub(crate) fn sk_dataset_from_batches<F: SKFloat>(
    batches: Vec<SKTabularBatch>,
    config: &SKTabularSourceConfig,
) -> Result<SKTabularDataset<F>, SKError> {
    match batches.len() {
        0 => Err(SKError::EmptyInput {
            operation: "datafusion_dataframe_conversion",
        }),
        1 => batches[0].to_tabular_dataset::<F>(config),
        _ => sk_concatenate_batches(&batches)?.to_tabular_dataset::<F>(config),
    }
}

fn sk_concatenate_batches(batches: &[SKTabularBatch]) -> Result<SKTabularBatch, SKError> {
    let first = &batches[0];
    let schema = first.schema().clone();
    let columns = (0..first.number_of_columns())
        .map(|position| {
            let selected: Vec<&crate::columns::SKColumnData> = batches
                .iter()
                .map(|batch| &batch.columns()[position])
                .collect();
            sk_concatenate_column(&selected)
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    SKTabularBatch::new(schema, columns, 0, true)
}

fn sk_concatenate_column(
    columns: &[&crate::columns::SKColumnData],
) -> Result<crate::columns::SKColumnData, SKError> {
    use crate::columns::SKColumnData;

    macro_rules! concatenated {
        ($variant:ident) => {{
            let mut values = Vec::new();
            for column in columns {
                match column {
                    SKColumnData::$variant(column_values) => {
                        values.extend(column_values.iter().cloned());
                    }
                    _ => {
                        return Err(SKError::SchemaMismatch {
                            expected: format!("a {} column", stringify!($variant)),
                            found: column.logical_type().to_string(),
                        });
                    }
                }
            }
            Ok(SKColumnData::$variant(values))
        }};
    }

    match columns[0] {
        SKColumnData::Float32(_) => concatenated!(Float32),
        SKColumnData::Float64(_) => concatenated!(Float64),
        SKColumnData::Integer64(_) => concatenated!(Integer64),
        SKColumnData::Boolean(_) => concatenated!(Boolean),
        SKColumnData::Nominal(_) => concatenated!(Nominal),
        SKColumnData::CategoryCode(_) => concatenated!(CategoryCode),
    }
}
