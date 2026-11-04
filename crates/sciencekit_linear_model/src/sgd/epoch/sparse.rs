//! The zero-copy sparse epoch driver (spec `stochastic-gradient-solver`,
//! decision 9).
//!
//! Rows are borrowed from the CSR outer iterator and never converted to a dense
//! `n × p` matrix.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};
use sprs::CsMatView;

use super::super::config::SKSGDConfig;
use super::super::loss::SKSGDLoss;
use super::super::objective::SKEpochObjective;
use super::super::state::SKLearningState;
use super::indexed::sk_run_indexed_epoch;

/// Run one epoch over a borrowed CSR design.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_sparse_epoch<F: SKFloat, L: SKSGDLoss<F>>(
    state: &mut SKLearningState<F>,
    features: CsMatView<'_, F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: &L,
    config: &SKSGDConfig<F>,
    order: &[usize],
) -> Result<SKEpochObjective<F>, SKError> {
    if features.rows() != targets.len() {
        return Err(SKError::shape_mismatch_2d(
            features.rows(),
            features.cols(),
            targets.len(),
            features.cols(),
        ));
    }
    // The closure borrows each CSR outer row; no dense copy is ever built.
    sk_run_indexed_epoch(
        state,
        order,
        targets,
        sample_weights,
        loss,
        config,
        |index| features.outer_view(index).expect("row within bounds"),
    )
}
