//! Regressor cross-validation scoring.

use ndarray::Array1;
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKRegressorPredictor, SKSupervisedFit, SKSupervisedScorer,
    SKTargetView,
};

use super::fitting::sk_fit_on_partition;
use crate::{SKSplitter, sk_gather_rows};

/// Runs K-fold cross-validation for a regressor and returns one score per fold.
///
/// The estimator is fitted on each fold's train indices and scored on the test
/// indices with `scorer`. Folds run in parallel and the scores are collected in
/// fold order, so the result never depends on the thread count.
pub fn sk_cross_val_score_regressor<F, E, S>(
    estimator: &E,
    splitter: &dyn SKSplitter<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
    scorer: &S,
) -> Result<Array1<F>, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F> + Sync,
    E::Model: SKRegressorPredictor<F>,
    S: SKSupervisedScorer<F, E::Model> + Sync,
    SKError: From<E::Error>
        + From<<E::Model as SKRegressorPredictor<F>>::Error>
        + From<S::Error>,
{
    let splits = splitter.split(features, Some(targets), None)?;
    if splits.is_empty() {
        return Err(SKError::EmptyInput {
            operation: "sk_cross_val_score_regressor",
        });
    }

    let scores: Vec<F> = splits
        .par_iter()
        .map(|split| {
            let (train_features, train_targets) =
                sk_gather_rows(features, Some(targets), split.train())?;
            let (test_features, test_targets) =
                sk_gather_rows(features, Some(targets), split.test())?;
            let train_partition = train_targets.expect("targets were requested");
            let test_partition = test_targets.expect("targets were requested");

            let model = sk_fit_on_partition(estimator, &train_features, &train_partition)?;
            test_partition.with_view(|test_view| {
                scorer
                    .score(&model, test_features.view(), test_view)
                    .map_err(SKError::from)
            })
        })
        .collect::<Result<Vec<F>, SKError>>()?;

    Ok(Array1::from_vec(scores))
}
