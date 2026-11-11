//! Successive-halving round evaluation over subsampled folds
//! (spec `halving-grid-search`, requirements `Successive halving loop` and
//! `Resource subsampling`).

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat, SKSupervisedFit, SKTargetView};

use crate::SKSplitter;
use crate::grid_search::SKErrorScore;
use crate::grid_search::evaluation::{SKFoldScorer, sk_evaluate_candidates};
use crate::grid_search::observation::sk_view_dimensions;

use super::scheduler::{sk_halving_resource_schedule, sk_surviving_candidate_count};
use super::subsample::sk_gather_round_subsample;
use super::trace::sk_build_round_trace;
use super::{SKHalvingResource, SKHalvingRound};

/// Input bundle for executing a successive-halving search.
pub(crate) struct SKHalvingEvaluationRequest<'s, 'd, F: SKFloat, E> {
    pub(crate) candidates: &'s [E],
    pub(crate) splitter: &'s dyn SKSplitter<F>,
    pub(crate) features: SKDataView<'d, F>,
    pub(crate) targets: SKTargetView<'d, F>,
    pub(crate) groups: Option<ArrayView1<'s, i64>>,
    pub(crate) factor: usize,
    pub(crate) minimum_resource: usize,
    pub(crate) maximum_resource: Option<usize>,
    #[allow(dead_code)]
    pub(crate) resource: SKHalvingResource,
    pub(crate) aggressive_elimination: bool,
    pub(crate) random_seed: u64,
    pub(crate) error_score: SKErrorScore<F>,
}

/// Complete trace and winning candidate of a successive-halving search.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SKHalvingSearchOutcome<F: SKFloat> {
    pub(crate) rounds: Vec<SKHalvingRound<F>>,
    pub(crate) best_candidate_index: usize,
    pub(crate) best_score: F,
}

struct SKCandidateReference<'a, E>(&'a E);

impl<F: SKFloat, E: SKSupervisedFit<F>> SKSupervisedFit<F> for SKCandidateReference<'_, E> {
    type Model = E::Model;
    type Error = E::Error;

    fn fit<'b, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'b, F>, Error = SKError>,
        T: TryInto<SKTargetView<'b, F>, Error = SKError>,
    {
        self.0.fit(features, targets)
    }
}

/// Runs the successive-halving tournament across subsampled rounds.
#[allow(dead_code)]
pub(crate) fn sk_run_halving_search<F, E, S>(
    request: SKHalvingEvaluationRequest<'_, '_, F, E>,
    scoring: &S,
) -> Result<SKHalvingSearchOutcome<F>, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F> + Sync,
    S: SKFoldScorer<F, E::Model> + Sync,
    SKError: From<E::Error>,
{
    let effective_maximum_resource = sk_validate_fit_inputs(&request)?;
    let resource_schedule = sk_halving_resource_schedule(
        request.minimum_resource,
        effective_maximum_resource,
        request.factor,
    );

    let mut alive: Vec<usize> = (0..request.candidates.len()).collect();
    let mut rounds: Vec<SKHalvingRound<F>> = Vec::with_capacity(resource_schedule.len());

    for (round_index, &current_resource) in resource_schedule.iter().enumerate() {
        if round_index > 0 && alive.len() <= 1 {
            break;
        }
        let round_trace = sk_evaluate_single_round(
            &request,
            scoring,
            round_index,
            current_resource,
            effective_maximum_resource,
            alive,
        )?;
        let mut next_alive = round_trace.surviving_candidate_indices().to_vec();
        next_alive.sort_unstable();
        alive = next_alive;
        rounds.push(round_trace);
    }

    let last_round = rounds.last().ok_or(SKError::EmptyInput {
        operation: "halving_grid_search",
    })?;
    let best_candidate_index = last_round.surviving_candidate_indices()[0];
    let best_position = last_round
        .evaluated_candidate_indices()
        .iter()
        .position(|&index| index == best_candidate_index)
        .expect("the winner was evaluated in the final round");
    let best_score = last_round.mean_scores()[best_position];

    Ok(SKHalvingSearchOutcome {
        rounds,
        best_candidate_index,
        best_score,
    })
}

fn sk_validate_fit_inputs<F: SKFloat, E>(
    request: &SKHalvingEvaluationRequest<'_, '_, F, E>,
) -> Result<usize, SKError> {
    if request.candidates.is_empty() {
        return Err(SKError::InvalidHyperparameter {
            name: "candidates",
            reason: "at least one configured candidate is required".to_owned(),
        });
    }
    let (total_rows, _) = sk_view_dimensions(request.features);
    if total_rows == 0 {
        return Err(SKError::EmptyInput {
            operation: "halving_grid_search",
        });
    }
    if request.targets.len() != total_rows {
        return Err(SKError::ShapeMismatch {
            expected: vec![total_rows],
            found: vec![request.targets.len()],
        });
    }
    if let Some(group_values) = request.groups {
        if group_values.len() != total_rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![total_rows],
                found: vec![group_values.len()],
            });
        }
    }

    let effective_maximum_resource = match request.maximum_resource {
        Some(bound) if bound == 0 || bound > total_rows => {
            return Err(SKError::InvalidHyperparameter {
                name: "maximum_resource",
                reason: format!(
                    "maximum_resource ({bound}) must be within 1..={total_rows} for this dataset"
                ),
            });
        }
        Some(bound) => bound,
        None => total_rows,
    };

    if request.minimum_resource == 0 || request.minimum_resource > effective_maximum_resource {
        return Err(SKError::InvalidHyperparameter {
            name: "minimum_resource",
            reason: format!(
                "minimum_resource ({}) must lie within 1..={effective_maximum_resource}",
                request.minimum_resource
            ),
        });
    }
    Ok(effective_maximum_resource)
}

fn sk_evaluate_single_round<F, E, S>(
    request: &SKHalvingEvaluationRequest<'_, '_, F, E>,
    scoring: &S,
    round_index: usize,
    current_resource: usize,
    effective_maximum_resource: usize,
    alive: Vec<usize>,
) -> Result<SKHalvingRound<F>, SKError>
where
    F: SKFloat,
    E: SKSupervisedFit<F> + Sync,
    S: SKFoldScorer<F, E::Model> + Sync,
    SKError: From<E::Error>,
{
    let span = tracing::info_span!(
        "sciencekit.halving_grid_search.round",
        round = round_index,
        resource = current_resource,
        candidates_count = alive.len(),
    );
    let _entered = span.enter();

    let subsample = sk_gather_round_subsample(
        request.features,
        request.targets,
        request.groups,
        current_resource,
        request.random_seed,
        round_index,
        request.splitter.requires_stratification(),
    )?;
    let round_candidates: Vec<SKCandidateReference<'_, E>> = alive
        .iter()
        .map(|&index| SKCandidateReference(&request.candidates[index]))
        .collect();

    let results = subsample.targets.with_view(|sub_targets| {
        sk_evaluate_candidates(
            &round_candidates,
            request.splitter,
            subsample.features.view(),
            sub_targets,
            subsample.groups.as_ref().map(|groups| groups.view()),
            request.error_score,
            scoring,
        )
    })?;

    let standard_survivors = sk_surviving_candidate_count(
        alive.len(),
        request.factor,
        current_resource,
        effective_maximum_resource,
        false,
    );
    let surviving = sk_surviving_candidate_count(
        alive.len(),
        request.factor,
        current_resource,
        effective_maximum_resource,
        request.aggressive_elimination,
    );
    Ok(sk_build_round_trace(
        round_index,
        current_resource,
        alive,
        results,
        standard_survivors,
        surviving,
    ))
}
