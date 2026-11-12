//! Scalar histogram boosting driver with optional early stopping.

use sciencekit_common::{SKError, SKFloat};

use crate::boosting::early_stopping::{SKEarlyStoppingConfig, SKEarlyStoppingMonitor};
use crate::boosting::init::sk_initialize_scalar_scores;
use crate::boosting::loss::SKGradientLoss;
use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::growth::SKHistogramTree;
use crate::histogram_boosting::stage::{
    SKHistogramStageParameters, sk_fit_histogram_scalar_stage,
};

/// Output of a scalar histogram boosting run.
#[derive(Debug, Clone)]
pub(crate) struct SKHistogramScalarFit<F> {
    pub(crate) initial_prediction: F,
    pub(crate) stages: Vec<SKHistogramTree<F>>,
    pub(crate) stage_importances: Vec<Vec<F>>,
}

/// Fit up to `number_of_estimators` scalar histogram stages, truncating to the
/// best stage count when early stopping is enabled.
pub(crate) fn sk_run_histogram_scalar<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    dataset: &SKBinnedDataset<F>,
    targets: &[F],
    training_rows: &[usize],
    validation_rows: Option<&[usize]>,
    number_of_estimators: usize,
    parameters: &SKHistogramStageParameters<F>,
    early_stopping: Option<SKEarlyStoppingConfig<F>>,
) -> Result<SKHistogramScalarFit<F>, SKError> {
    let training_targets: Vec<F> = training_rows.iter().map(|&row| targets[row]).collect();
    let (initial_prediction, _) = sk_initialize_scalar_scores(loss, &training_targets);
    let mut full_scores = vec![initial_prediction; targets.len()];
    let mut stages = Vec::with_capacity(number_of_estimators);
    let mut stage_importances = Vec::with_capacity(number_of_estimators);
    let mut monitor = early_stopping
        .map(|stopping| SKEarlyStoppingMonitor::new(stopping.patience, stopping.tolerance));
    for stage_index in 0..number_of_estimators {
        let result = sk_fit_histogram_scalar_stage(
            loss,
            dataset,
            targets,
            &mut full_scores,
            parameters,
            stage_index,
            training_rows,
        )?;
        stages.push(result.tree);
        stage_importances.push(result.feature_importances);
        if let (Some(monitor), Some(validation_rows)) = (monitor.as_mut(), validation_rows) {
            if !validation_rows.is_empty() {
                let divisor = F::from(validation_rows.len()).expect("count fits");
                let validation_loss = validation_rows
                    .iter()
                    .fold(F::zero(), |sum, &row| {
                        sum + loss.loss(full_scores[row], targets[row])
                    })
                    / divisor;
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
    Ok(SKHistogramScalarFit {
        initial_prediction,
        stages,
        stage_importances,
    })
}
