//! Regression metric computation for `evaluate` and `run`.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
use sciencekit::metrics::{
    sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score, sk_root_mean_squared_error,
};

use crate::error::SKCliError;

/// Compute the selected metrics from targets and predictions.
pub fn sk_compute_metrics(
    targets: ArrayView1<'_, f64>,
    predictions: ArrayView1<'_, f64>,
    metrics: &[String],
) -> Result<BTreeMap<String, f64>, SKCliError> {
    metrics
        .iter()
        .map(|metric| {
            let value = match metric.as_str() {
                "r2" => sk_r2_score(targets, predictions),
                "rmse" => sk_root_mean_squared_error(targets, predictions),
                "mse" => sk_mean_squared_error(targets, predictions),
                "mae" => sk_mean_absolute_error(targets, predictions),
                other => {
                    return Err(SKCliError::validation(
                        "configuration.metrics",
                        format!("`{other}` is not a supported metric; use r2, rmse, mse or mae"),
                    ));
                }
            }
            .map_err(|error| SKCliError::runtime(format!("metric `{metric}`: {error}")))?;
            Ok((metric.clone(), value))
        })
        .collect()
}
