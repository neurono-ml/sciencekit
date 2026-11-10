//! Candidate/fold evaluation (spec `grid-search`, requirement `Candidate
//! evaluation across folds`).

use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat, SKSupervisedFit, SKTargetView};

use super::super::error_score::SKErrorScore;
use super::super::results::SKGridSearchResults;
use super::SKFoldScorer;
use crate::SKIndexSplit;
use crate::cross_validation::sk_fit_on_partition;
use crate::{SKSplitter, sk_gather_rows};

/// Evaluates every candidate on every fold and aggregates the scores.
///
/// Candidates and folds are independent; evaluation runs in parallel and the
/// results are written by candidate and fold index, so the matrix never depends
/// on thread count or completion order. The first failing fold (in candidate,
/// then fold order) is either propagated or replaced by the configured error
/// score.
pub(crate) fn sk_evaluate_candidates<F, E, S>(
    candidates: &[E],
    splitter: &dyn SKSplitter<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
    groups: Option<ArrayView1<'_, i64>>,
    error_score: SKErrorScore<F>,
    scoring: &S,
) -> Result<SKGridSearchResults<F>, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F> + Sync,
    S: SKFoldScorer<F, E::Model> + Sync,
    SKError: From<E::Error>,
{
    let splits = splitter.split(features, Some(targets), groups)?;
    if splits.is_empty() {
        return Err(SKError::EmptyInput {
            operation: "grid_search",
        });
    }

    let number_of_candidates = candidates.len();
    let number_of_folds = splits.len();
    let pairs: Vec<(usize, usize)> = (0..number_of_candidates)
        .flat_map(|candidate| (0..number_of_folds).map(move |fold| (candidate, fold)))
        .collect();

    let evaluated: Vec<Result<F, SKError>> = pairs
        .par_iter()
        .map(|&(candidate, fold)| {
            sk_evaluate_fold(
                candidates,
                &splits[fold],
                features,
                targets,
                scoring,
                candidate,
                fold,
            )
        })
        .collect();

    let mut scores = Array2::zeros((number_of_candidates, number_of_folds));
    for (&(candidate, fold), result) in pairs.iter().zip(evaluated) {
        match result {
            Ok(score) => scores[[candidate, fold]] = score,
            Err(error) => match error_score {
                SKErrorScore::Raise => return Err(error),
                SKErrorScore::Value(value) => scores[[candidate, fold]] = value,
            },
        }
    }

    SKGridSearchResults::from_scores(scores)
}

fn sk_evaluate_fold<F, E, S>(
    candidates: &[E],
    split: &SKIndexSplit,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
    scoring: &S,
    candidate: usize,
    fold: usize,
) -> Result<F, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    S: SKFoldScorer<F, E::Model>,
    SKError: From<E::Error>,
{
    let span = tracing::info_span!(
        "sciencekit.grid_search.fold",
        candidate_index = candidate,
        fold_index = fold,
    );
    let _entered = span.enter();

    let (train_features, train_targets) = sk_gather_rows(features, Some(targets), split.train())?;
    let (test_features, test_targets) = sk_gather_rows(features, Some(targets), split.test())?;
    let train_partition = train_targets.expect("targets were requested");
    let test_partition = test_targets.expect("targets were requested");

    let model = sk_fit_on_partition(&candidates[candidate], &train_features, &train_partition)?;
    test_partition.with_view(|test_view| scoring.score(&model, test_features.view(), test_view))
}
