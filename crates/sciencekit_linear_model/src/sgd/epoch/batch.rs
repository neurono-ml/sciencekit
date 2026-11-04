//! The streaming-batch epoch driver (spec `stochastic-gradient-solver`,
//! decision 9).

use sciencekit_common::{SKError, SKFloat, SKSupervisedBatch};

use super::super::config::SKSGDConfig;
use super::super::loss::SKSGDLoss;
use super::super::objective::SKEpochObjective;
use super::super::state::SKLearningState;

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
    let view = batch.view();
    let targets = batch.targets();
    if batch.number_of_rows() != targets.len() {
        return Err(SKError::shape_mismatch_2d(
            batch.number_of_rows(),
            batch.number_of_columns(),
            targets.len(),
            batch.number_of_columns(),
        ));
    }
    super::block::sk_run_block_epoch(state, view, targets, sample_weights, loss, config, order)
}
