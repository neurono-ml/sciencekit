//! Multi-stage multiclass boosting driver with optional held-out validation
//! early stopping and best-stage restoration.

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::early_stopping::{
    SKEarlyStoppingConfig, SKEarlyStoppingMonitor, sk_split_classification_validation_rows,
};
use super::init::sk_initialize_multiclass_scores;
use super::loss::SKMulticlassLogLoss;
use super::scalar_loop::LoopExit;
use super::scores::sk_update_raw_scores;
use super::stage::{SKBoostingStageParameters, SKFittedStageTree, sk_fit_multiclass_stage};
use super::subsample::SKOwnedRowSubset;

/// Output of a multiclass boosting run (`C` trees per stage).
#[derive(Debug, Clone)]
pub(crate) struct SKFittedMulticlassStages<F> {
    pub(crate) initial_predictions: Vec<F>,
    pub(crate) stages: Vec<Vec<SKFittedStageTree<F>>>,
}

/// Fit up to `number_of_estimators` multiclass stages, optionally monitoring
/// held-out cross-entropy loss and restoring the best stage count.
pub(crate) fn sk_run_multiclass_boosting<F: SKFloat>(
    loss: &SKMulticlassLogLoss,
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    number_of_estimators: usize,
    parameters: &SKBoostingStageParameters<F>,
    early_stopping: Option<SKEarlyStoppingConfig<F>>,
) -> Result<SKFittedMulticlassStages<F>, SKError> {
    let Some(stopping) = early_stopping else {
        let (initial_predictions, mut raw_scores) =
            sk_initialize_multiclass_scores(loss, class_indices);
        let stages = (0..number_of_estimators)
            .map(|stage_index| {
                sk_fit_multiclass_stage(
                    loss,
                    features,
                    class_indices,
                    &mut raw_scores,
                    parameters,
                    stage_index,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(SKFittedMulticlassStages {
            initial_predictions,
            stages,
        });
    };
    let (training_rows, validation_rows) = sk_split_classification_validation_rows(
        class_indices,
        loss.number_of_classes(),
        stopping.validation_fraction,
        stopping.random_seed,
    )?;
    let training_matrix = SKOwnedRowSubset::gather(features, &training_rows)?;
    let validation_matrix = SKOwnedRowSubset::gather(features, &validation_rows)?;
    let training_view = training_matrix.as_view();
    let validation_view = validation_matrix.as_view();
    let training_classes: Vec<usize> = training_rows
        .iter()
        .map(|&row| class_indices[row])
        .collect();
    let validation_classes: Vec<usize> = validation_rows
        .iter()
        .map(|&row| class_indices[row])
        .collect();
    let (initial_predictions, mut training_scores) =
        sk_initialize_multiclass_scores(loss, &training_classes);
    let mut validation_scores = Array2::from_shape_fn(
        (validation_classes.len(), loss.number_of_classes()),
        |(_, class_index)| initial_predictions[class_index],
    );
    let mut monitor = SKEarlyStoppingMonitor::new(stopping.patience, stopping.tolerance);
    let mut stages = Vec::with_capacity(number_of_estimators);
    let validation_divisor =
        F::from(validation_classes.len()).expect("validation count fits in scalar");

    let loop_outcome = (0..number_of_estimators).try_for_each(|stage_index| {
        let stage_trees = sk_fit_multiclass_stage(
            loss,
            &training_view,
            &training_classes,
            &mut training_scores,
            parameters,
            stage_index,
        )
        .map_err(LoopExit::Failure)?;
        stage_trees
            .iter()
            .enumerate()
            .for_each(|(class_index, fitted)| {
                let mut column_scores: Vec<F> = validation_scores.column(class_index).to_vec();
                sk_update_raw_scores(
                    &validation_view,
                    &fitted.tree,
                    parameters.learning_rate,
                    &mut column_scores,
                );
                validation_scores
                    .column_mut(class_index)
                    .iter_mut()
                    .zip(column_scores.iter())
                    .for_each(|(destination, &updated)| *destination = updated);
            });
        stages.push(stage_trees);
        let validation_loss = validation_scores
            .rows()
            .into_iter()
            .zip(validation_classes.iter())
            .fold(F::zero(), |sum, (score_row, &true_class)| {
                let row_slice: Vec<F> = score_row.iter().copied().collect();
                sum + loss.loss_for_row(&row_slice, true_class)
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
    Ok(SKFittedMulticlassStages {
        initial_predictions,
        stages,
    })
}
