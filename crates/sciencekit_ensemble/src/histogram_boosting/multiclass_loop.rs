//! Multiclass histogram boosting driver: one histogram tree per class per
//! stage, with optional early stopping.

use ndarray::Array2;
use sciencekit_common::{SKError, SKFloat};

use crate::boosting::early_stopping::{SKEarlyStoppingConfig, SKEarlyStoppingMonitor};
use crate::boosting::init::sk_initialize_multiclass_scores;
use crate::boosting::loss::SKMulticlassLogLoss;
use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::growth::{SKHistogramTree, sk_grow_histogram_tree};
use crate::histogram_boosting::stage::SKHistogramStageParameters;

/// Output of a multiclass histogram boosting run (`C` trees per stage).
#[derive(Debug, Clone)]
pub(crate) struct SKHistogramMulticlassFit<F> {
    pub(crate) initial_predictions: Vec<F>,
    pub(crate) stages: Vec<Vec<SKHistogramTree<F>>>,
    pub(crate) stage_importances: Vec<Vec<Vec<F>>>,
}

/// Fit up to `number_of_estimators` multiclass histogram stages.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_histogram_multiclass<F: SKFloat>(
    loss: &SKMulticlassLogLoss,
    dataset: &SKBinnedDataset<F>,
    class_indices: &[usize],
    training_rows: &[usize],
    validation_rows: Option<&[usize]>,
    number_of_estimators: usize,
    parameters: &SKHistogramStageParameters<F>,
    early_stopping: Option<SKEarlyStoppingConfig<F>>,
) -> Result<SKHistogramMulticlassFit<F>, SKError> {
    let training_classes: Vec<usize> = training_rows
        .iter()
        .map(|&row| class_indices[row])
        .collect();
    let (initial_predictions, raw_scores) =
        sk_initialize_multiclass_scores(loss, &training_classes);
    let mut full_scores = Array2::from_shape_fn(
        (class_indices.len(), loss.number_of_classes()),
        |(_, class_index)| initial_predictions[class_index],
    );
    let mut stages = Vec::with_capacity(number_of_estimators);
    let mut stage_importances = Vec::with_capacity(number_of_estimators);
    let mut monitor = early_stopping
        .map(|stopping| SKEarlyStoppingMonitor::new(stopping.patience, stopping.tolerance));
    for _stage_index in 0..number_of_estimators {
        let (negative_gradients, hessians) =
            loss.stage_gradients_and_hessians(class_indices, &raw_scores);
        let mut trees = Vec::with_capacity(loss.number_of_classes());
        let mut importances = Vec::with_capacity(loss.number_of_classes());
        (0..loss.number_of_classes()).for_each(|class_index| {
            let result = sk_grow_histogram_tree(
                dataset,
                &negative_gradients[class_index],
                &hessians[class_index],
                training_rows,
                &parameters.growth,
            );
            let mut column_scores: Vec<F> = full_scores.column(class_index).to_vec();
            column_scores
                .iter_mut()
                .enumerate()
                .for_each(|(row, score)| {
                    *score = *score
                        + parameters.learning_rate
                            * result
                                .tree
                                .leaf_value_for(|feature| dataset.bin(feature, row));
                });
            full_scores
                .column_mut(class_index)
                .iter_mut()
                .zip(column_scores.iter())
                .for_each(|(destination, &updated)| *destination = updated);
            trees.push(result.tree);
            importances.push(result.feature_importances);
        });
        stages.push(trees);
        stage_importances.push(importances);
        if let (Some(monitor), Some(validation_rows)) = (monitor.as_mut(), validation_rows) {
            if !validation_rows.is_empty() {
                let divisor = F::from(validation_rows.len()).expect("count fits");
                let validation_loss = validation_rows.iter().fold(F::zero(), |sum, &row| {
                    let row_scores: Vec<F> = full_scores.row(row).iter().copied().collect();
                    sum + loss.loss_for_row(&row_scores, class_indices[row])
                }) / divisor;
                if monitor.record(stages.len(), validation_loss) {
                    break;
                }
            }
        }
    }
    if let Some(monitor) = monitor {
        stages.truncate(monitor.best_stage_count());
        stage_importances.truncate(monitor.best_stage_count());
    }
    Ok(SKHistogramMulticlassFit {
        initial_predictions,
        stages,
        stage_importances,
    })
}
