//! Multi-stage scalar boosting driver (regression and binary classification)
//! with optional held-out validation early stopping and best-stage restoration.

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsMat;

use crate::bootstrap::{sk_gather_dense_rows, sk_gather_sparse_rows};

use super::early_stopping::{
    SKEarlyStoppingConfig, SKEarlyStoppingMonitor, sk_split_classification_validation_rows,
    sk_split_regression_validation_rows,
};
use super::init::sk_initialize_scalar_scores;
use super::loss::SKGradientLoss;
use super::scores::sk_update_raw_scores;
use super::stage::{SKBoostingStageParameters, SKFittedStageTree, sk_fit_scalar_stage};

/// Output of a scalar (regression or binary classification) boosting run.
#[derive(Debug, Clone)]
pub(crate) struct SKFittedScalarStages<F> {
    pub(crate) initial_prediction: F,
    pub(crate) stages: Vec<SKFittedStageTree<F>>,
}

pub(super) enum SKOwnedRowSubset<F> {
    Dense(Array2<F>),
    Sparse(CsMat<F>),
}

impl<F: SKFloat> SKOwnedRowSubset<F> {
    pub(super) fn gather(features: &SKDataView<'_, F>, indices: &[usize]) -> Result<Self, SKError> {
        match features {
            SKDataView::Dense(dense) => Ok(Self::Dense(sk_gather_dense_rows(dense, indices))),
            SKDataView::Sparse(sparse) => Ok(Self::Sparse(sk_gather_sparse_rows(sparse, indices))),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    }

    pub(super) fn as_view(&self) -> SKDataView<'_, F> {
        match self {
            Self::Dense(dense) => SKDataView::Dense(dense.view()),
            Self::Sparse(sparse) => SKDataView::Sparse(sparse.view()),
        }
    }
}

/// Fit up to `number_of_estimators` scalar stages, optionally monitoring
/// validation loss and restoring the best stage count.
pub(crate) fn sk_run_scalar_boosting<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    features: &SKDataView<'_, F>,
    targets: &[F],
    number_of_estimators: usize,
    parameters: &SKBoostingStageParameters<F>,
    early_stopping: Option<SKEarlyStoppingConfig<F>>,
) -> Result<SKFittedScalarStages<F>, SKError> {
    let Some(stopping) = early_stopping else {
        let (initial_prediction, mut raw_scores) = sk_initialize_scalar_scores(loss, targets);
        let stages = (0..number_of_estimators)
            .map(|stage| {
                sk_fit_scalar_stage(loss, features, targets, &mut raw_scores, parameters, stage)
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(SKFittedScalarStages {
            initial_prediction,
            stages,
        });
    };
    let (training_rows, validation_rows) = sk_split_regression_validation_rows(
        targets.len(),
        stopping.validation_fraction,
        stopping.random_seed,
    )?;
    sk_run_scalar_with_split(
        loss,
        features,
        targets,
        &training_rows,
        &validation_rows,
        number_of_estimators,
        parameters,
        stopping,
    )
}

/// Fit up to `number_of_estimators` binary classification stages, using a
/// stratified validation split when early stopping is enabled.
pub(crate) fn sk_run_binary_boosting<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    targets: &[F],
    number_of_estimators: usize,
    parameters: &SKBoostingStageParameters<F>,
    early_stopping: Option<SKEarlyStoppingConfig<F>>,
) -> Result<SKFittedScalarStages<F>, SKError> {
    if let Some(stopping) = early_stopping {
        let (training_rows, validation_rows) = sk_split_classification_validation_rows(
            class_indices,
            2,
            stopping.validation_fraction,
            stopping.random_seed,
        )?;
        sk_run_scalar_with_split(
            loss,
            features,
            targets,
            &training_rows,
            &validation_rows,
            number_of_estimators,
            parameters,
            stopping,
        )
    } else {
        sk_run_scalar_boosting(
            loss,
            features,
            targets,
            number_of_estimators,
            parameters,
            None,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn sk_run_scalar_with_split<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    features: &SKDataView<'_, F>,
    targets: &[F],
    training_rows: &[usize],
    validation_rows: &[usize],
    number_of_estimators: usize,
    parameters: &SKBoostingStageParameters<F>,
    stopping: SKEarlyStoppingConfig<F>,
) -> Result<SKFittedScalarStages<F>, SKError> {
    let training_matrix = SKOwnedRowSubset::gather(features, training_rows)?;
    let validation_matrix = SKOwnedRowSubset::gather(features, validation_rows)?;
    let training_view = training_matrix.as_view();
    let validation_view = validation_matrix.as_view();
    let training_targets: Vec<F> = training_rows.iter().map(|&row| targets[row]).collect();
    let validation_targets: Vec<F> = validation_rows.iter().map(|&row| targets[row]).collect();
    let (initial_prediction, mut training_scores) =
        sk_initialize_scalar_scores(loss, &training_targets);
    let mut validation_scores = vec![initial_prediction; validation_targets.len()];
    let mut monitor = SKEarlyStoppingMonitor::new(stopping.patience, stopping.tolerance);
    let mut stages = Vec::with_capacity(number_of_estimators);
    let validation_divisor =
        F::from(validation_targets.len()).expect("validation count fits in scalar");

    let loop_outcome =
        (0..number_of_estimators).try_for_each(|stage_index| -> Result<(), LoopExit> {
            let fitted = sk_fit_scalar_stage(
                loss,
                &training_view,
                &training_targets,
                &mut training_scores,
                parameters,
                stage_index,
            )
            .map_err(LoopExit::Failure)?;
            sk_update_raw_scores(
                &validation_view,
                &fitted.tree,
                parameters.learning_rate,
                &mut validation_scores,
            );
            stages.push(fitted);
            let validation_loss = validation_scores
                .iter()
                .zip(validation_targets.iter())
                .fold(F::zero(), |sum, (&score, &target)| {
                    sum + loss.loss(score, target)
                })
                / validation_divisor;
            if monitor.record(stages.len(), validation_loss) {
                return Err(LoopExit::Converged);
            }
            Ok(())
        });
    if let Err(LoopExit::Failure(error)) = loop_outcome {
        return Err(error);
    }
    stages.truncate(monitor.best_stage_count());
    Ok(SKFittedScalarStages {
        initial_prediction,
        stages,
    })
}

pub(super) enum LoopExit {
    Converged,
    Failure(SKError),
}
