//! Parquet files read through DataFusion and converted to a materialized
//! dataset (spec `tabular-sources`).

use std::path::Path;

use ::datafusion::dataframe::DataFrame;
use ::datafusion::prelude::{ParquetReadOptions, SessionContext};
use sciencekit_common::{SKError, SKFloat};

use super::dataframe_conversion::{
    sk_datafusion_runtime, sk_tabular_dataset_from_dataframe,
};
use crate::tabular::{SKTabularDataset, SKTabularSourceConfig};

/// The alias a Parquet file receives inside the temporary DataFusion session.
const SK_PARQUET_TABLE_ALIAS: &str = "sciencekit_source";

/// Register a Parquet file in a fresh session and return its `DataFrame`.
///
/// # Errors
/// Returns [`SKError::Conversion`] when DataFusion cannot register or plan the
/// file.
pub async fn sk_dataframe_from_parquet(path: &Path) -> Result<DataFrame, SKError> {
    let context = SessionContext::new();
    context
        .register_parquet(
            SK_PARQUET_TABLE_ALIAS,
            path.to_string_lossy().as_ref(),
            ParquetReadOptions::default(),
        )
        .await
        .map_err(|error| SKError::Conversion(format!("registering the Parquet file failed: {error}")))?;
    context
        .table(SK_PARQUET_TABLE_ALIAS)
        .await
        .map_err(|error| SKError::Conversion(format!("planning the Parquet file failed: {error}")))
}

/// Read a Parquet file through DataFusion into an owned dataset.
///
/// # Errors
/// Propagates the registration/planning failure and the dataset conversion
/// failures.
pub async fn sk_tabular_dataset_from_parquet<F: SKFloat>(
    path: &Path,
    config: &SKTabularSourceConfig,
) -> Result<SKTabularDataset<F>, SKError> {
    let dataframe = sk_dataframe_from_parquet(path).await?;
    sk_tabular_dataset_from_dataframe(dataframe, config).await
}

/// Block on [`sk_tabular_dataset_from_parquet`] with a fresh runtime.
///
/// # Errors
/// Returns [`SKError::UnsupportedOperation`] inside a Tokio runtime and
/// propagates every conversion failure.
pub fn sk_tabular_dataset_from_parquet_blocking<F: SKFloat>(
    path: &Path,
    config: &SKTabularSourceConfig,
) -> Result<SKTabularDataset<F>, SKError> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(SKError::UnsupportedOperation {
            operation: "datafusion_parquet_blocking",
            reason: "use the async conversion inside a Tokio runtime".to_string(),
        });
    }
    sk_datafusion_runtime()?.block_on(sk_tabular_dataset_from_parquet::<F>(path, config))
}
