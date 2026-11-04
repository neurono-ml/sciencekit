//! The representation-generic epoch driver (spec `stochastic-gradient-solver`,
//! decision 9).
//!
//! A dense view takes the dense row driver; a CSR view takes the zero-copy
//! sparse row driver. The visiting order, shape checks and objective
//! accumulation stay shared.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::super::config::SKSGDConfig;
use super::super::loss::SKSGDLoss;
use super::super::objective::SKEpochObjective;
use super::super::state::SKLearningState;
use super::dense::sk_run_dense_epoch;
use super::sparse::sk_run_sparse_epoch;

/// Run one epoch over a feature view of either representation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_block_epoch<F: SKFloat, L: SKSGDLoss<F>>(
    state: &mut SKLearningState<F>,
    features: SKDataView<'_, F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: &L,
    config: &SKSGDConfig<F>,
    order: &[usize],
) -> Result<SKEpochObjective<F>, SKError> {
    match features {
        SKDataView::Dense(dense) => {
            sk_run_dense_epoch(state, dense, targets, sample_weights, loss, config, order)
        }
        SKDataView::Sparse(sparse) => {
            sk_run_sparse_epoch(state, sparse, targets, sample_weights, loss, config, order)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}
