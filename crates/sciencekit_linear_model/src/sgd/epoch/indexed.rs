//! The shared indexed epoch loop (spec `stochastic-gradient-solver`,
//! decision 9).
//!
//! Every driver differs only in how it yields a row; the visiting order, the
//! shape checks and the objective accumulation are shared here.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};

use super::super::config::SKSGDConfig;
use super::super::loss::SKSGDLoss;
use super::super::objective::SKEpochObjective;
use super::super::row::SKSGDFeatureRow;
use super::super::state::SKLearningState;
use super::super::update::sk_apply_single_sample_update;

/// Run one epoch over the rows selected by `order`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_indexed_epoch<F, L, R, GetRow>(
    state: &mut SKLearningState<F>,
    order: &[usize],
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: &L,
    config: &SKSGDConfig<F>,
    get_row: GetRow,
) -> Result<SKEpochObjective<F>, SKError>
where
    F: SKFloat,
    L: SKSGDLoss<F>,
    R: SKSGDFeatureRow<F>,
    GetRow: Fn(usize) -> R,
{
    let number_of_rows = targets.len();
    if sample_weights.len() != number_of_rows {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_rows],
            found: vec![sample_weights.len()],
        });
    }
    if order.len() != number_of_rows {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_rows],
            found: vec![order.len()],
        });
    }
    let mut objective = SKEpochObjective::new();
    for &index in order {
        if index >= number_of_rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_rows],
                found: vec![index],
            });
        }
        let weighted_loss = sk_apply_single_sample_update(
            state,
            get_row(index),
            targets[index],
            sample_weights[index],
            loss,
            config,
        );
        objective.record(weighted_loss);
    }
    Ok(objective)
}
