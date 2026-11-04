//! The dense epoch driver (spec `stochastic-gradient-solver`, decision 9).

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use super::super::config::SKSGDConfig;
use super::super::loss::SKSGDLoss;
use super::super::objective::SKEpochObjective;
use super::super::state::SKLearningState;
use super::indexed::sk_run_indexed_epoch;

/// Run one epoch over a borrowed dense design.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_dense_epoch<F: SKFloat, L: SKSGDLoss<F>>(
    state: &mut SKLearningState<F>,
    features: ArrayView2<F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: &L,
    config: &SKSGDConfig<F>,
    order: &[usize],
) -> Result<SKEpochObjective<F>, SKError> {
    if features.nrows() != targets.len() {
        return Err(SKError::shape_mismatch_2d(
            features.nrows(),
            features.ncols(),
            targets.len(),
            features.ncols(),
        ));
    }
    sk_run_indexed_epoch(
        state,
        order,
        targets,
        sample_weights,
        loss,
        config,
        |index| features.row(index),
    )
}
