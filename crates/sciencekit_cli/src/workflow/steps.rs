//! Workflow step execution: feature pipeline and grid search.

use ndarray::{ArrayView1, ArrayView2};
use sciencekit::metrics::SKR2Scorer;
use sciencekit::model_selection::{SKKFold, sk_cross_val_score_regressor};
use sciencekit::{SKBackendKind, SKBuilder, SKDataView, SKTargetView};

use crate::cli::commands::estimator::sk_apply_pipeline_step;
use crate::error::SKCliError;
use crate::workflow::schema::SKWorkflowPipelineStep;

/// The outcome of a grid search over penalty values.
#[derive(Debug, Clone)]
pub struct SKGridSearchOutcome {
    /// The penalty with the best mean cross-validation score.
    pub best_alpha: f64,
    /// The best mean score.
    pub best_score: f64,
    /// Every candidate with its mean score, in declaration order.
    pub candidates: Vec<(f64, f64)>,
}

/// Run plain K-fold cross-validation over the penalty candidates.
pub fn sk_grid_search(
    alphas: &[f64],
    folds: usize,
    backend: SKBackendKind,
    features: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Result<SKGridSearchOutcome, SKCliError> {
    let splitter = SKKFold::build()
        .number_of_splits(folds)
        .shuffle(true)
        .random_seed(0)
        .build()
        .map_err(|error| SKCliError::runtime(format!("grid search splitter: {error}")))?;
    let candidates = alphas
        .iter()
        .map(|alpha| {
            let estimator = sciencekit::linear_model::SKRidge::<f64>::build()
                .alpha(*alpha)
                .backend(backend)
                .build()
                .map_err(|error| SKCliError::runtime(format!("grid candidate: {error}")))?;
            let scores = sk_cross_val_score_regressor::<f64, _, _>(
                &estimator,
                &splitter,
                SKDataView::Dense(features.view()),
                SKTargetView::Continuous(targets),
                &SKR2Scorer,
            )
            .map_err(|error| SKCliError::runtime(format!("grid search score: {error}")))?;
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            Ok((*alpha, mean))
        })
        .collect::<Result<Vec<(f64, f64)>, SKCliError>>()?;
    let (best_alpha, best_score) = candidates
        .iter()
        .copied()
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| {
            SKCliError::validation("workflow.search.alphas", "no candidate was given")
        })?;
    Ok(SKGridSearchOutcome {
        best_alpha,
        best_score,
        candidates,
    })
}

/// Apply the configured feature pipeline over the full matrix.
pub fn sk_apply_pipeline_steps(
    steps: &[SKWorkflowPipelineStep],
    backend: SKBackendKind,
    features: ndarray::Array2<f64>,
) -> Result<ndarray::Array2<f64>, SKCliError> {
    steps.iter().try_fold(features, |current, step| {
        sk_apply_pipeline_step(step, backend, &current)
    })
}
