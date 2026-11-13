//! Workflow execution: dataset, pipeline, search, fit, metrics and artifacts.

use std::path::Path;

use serde_json::{Value, json};

use crate::cli::commands::estimator::{sk_fit_regressor, sk_require_targets};
use crate::cli::commands::metrics::sk_compute_metrics;
use crate::cli::commands::support::sk_output_format;
use crate::config::{SKEffectiveConfiguration, sk_parse_backend};
use crate::error::SKCliError;
use crate::io::{SKModelFormat, sk_load_dataset, sk_save_state, sk_write_predictions};
use crate::workflow::schema::SKWorkflow;
use crate::workflow::steps::{sk_apply_pipeline_steps, sk_grid_search};

/// Execute a validated workflow, returning its reproducible summary document.
pub fn sk_execute_workflow(
    workflow: &SKWorkflow,
    configuration: &SKEffectiveConfiguration,
) -> Result<Value, SKCliError> {
    let backend = sk_parse_backend(&configuration.backend)?;
    let dataset_path = configuration
        .dataset
        .clone()
        .ok_or_else(|| SKCliError::usage("a workflow needs a dataset path (`dataset.path`)"))?;
    let target_name = configuration
        .target
        .clone()
        .ok_or_else(|| SKCliError::usage("a workflow needs a target column (`dataset.target`)"))?;
    let dataset = sk_load_dataset(
        Path::new(&dataset_path),
        configuration.dataset_format.as_deref(),
        Some(&target_name),
    )?;
    let targets = sk_require_targets(dataset.targets.as_ref())?;
    let transformed =
        sk_apply_pipeline_steps(&workflow.pipeline, backend, dataset.features.clone())?;

    let (alpha, search_report) = match &workflow.search {
        Some(search) => {
            let folds = search.folds.unwrap_or(5);
            let outcome =
                sk_grid_search(&search.alphas, folds, backend, transformed.view(), targets)?;
            (
                outcome.best_alpha,
                Some(json!({
                    "kind": search.kind,
                    "folds": folds,
                    "scoring": search.scoring.clone().unwrap_or_else(|| "r2".to_string()),
                    "best_alpha": outcome.best_alpha,
                    "best_score": outcome.best_score,
                    "candidates": outcome.candidates.iter().map(|(alpha, score)| {
                        json!({"alpha": alpha, "mean_score": score})
                    }).collect::<Vec<_>>(),
                })),
            )
        }
        None => (configuration.alpha, None),
    };

    let state = sk_fit_regressor(
        &configuration.estimator,
        alpha,
        backend,
        transformed.view(),
        targets,
    )?;
    let model = sciencekit::interop::native::sk_import_native_state::<f64>(&state)
        .map_err(|error| SKCliError::runtime(format!("native model import: {error}")))?;
    let predictions = model
        .predict_regression(transformed.view())
        .map_err(|error| SKCliError::runtime(format!("prediction: {error}")))?;
    let metrics = sk_compute_metrics(targets, predictions.view(), &configuration.metrics)?;

    let mut written_model = Value::Null;
    if let Some(model_path) = &configuration.model {
        sk_save_state(
            Path::new(model_path),
            &state,
            SKModelFormat::from_path(Path::new(model_path)),
        )?;
        written_model = json!(model_path);
    }
    let mut written_predictions = Value::Null;
    if let Some(predictions_path) = &configuration.predictions {
        let format = sk_output_format(configuration, Path::new(predictions_path))?;
        let values: Vec<f64> = predictions.iter().copied().collect();
        sk_write_predictions(Path::new(predictions_path), format, &target_name, &values)?;
        written_predictions = json!(predictions_path);
    }

    Ok(json!({
        "command": "run",
        "workflow_version": workflow.version,
        "dataset": dataset_path,
        "target": target_name,
        "number_of_rows": dataset.number_of_rows(),
        "number_of_features": transformed.ncols(),
        "estimator": configuration.estimator,
        "alpha": alpha,
        "backend": configuration.backend,
        "search": search_report,
        "metrics": metrics,
        "model": written_model,
        "predictions": written_predictions,
    }))
}
