//! The streaming-batch epoch driver (spec `stochastic-gradient-solver`,
//! decision 9).

use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch};

use super::super::config::SKSGDConfig;
use super::super::loss::SKSGDLoss;
use super::super::objective::SKEpochObjective;
use super::super::state::SKLearningState;
use super::indexed::sk_run_indexed_epoch;

/// Run one epoch over a single supervised streaming batch.
///
/// The batch owns its row block; the driver visits exactly its rows in `order`
/// and stops at the end of the block, so a final block is never over-read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_batch_epoch<F: SKFloat, L: SKSGDLoss<F>>(
    state: &mut SKLearningState<F>,
    batch: &SKSupervisedBatch<F>,
    sample_weights: &[F],
    loss: &L,
    config: &SKSGDConfig<F>,
    order: &[usize],
) -> Result<SKEpochObjective<F>, SKError> {
    let features = batch.features();
    let targets = batch.targets();
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
