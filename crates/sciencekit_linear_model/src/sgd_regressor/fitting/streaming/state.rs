//! The mutable state shared by every pass of the regressor stream.

use sciencekit_common::execution::SKStreamDecision;
use sciencekit_common::{SKError, SKFloat, SKRandomGenerator, SKSupervisedBatch};

use super::super::super::coefficients::SKSGDRegressorCoefficients;
use super::super::state::sk_create_state;
use crate::sgd::config::SKSGDConfig;
use crate::sgd::objective::SKEpochObjective;
use crate::sgd::{
    SKLearningState, SKSGDRegressorLoss, SKSGDRegressorLossAdapter, sk_run_batch_epoch,
};

/// The mutable state shared by every pass of the fit.
pub(super) struct SKRegressorStreamState<'a, F> {
    pub(super) config: &'a SKSGDConfig<F>,
    pub(super) loss: SKSGDRegressorLoss,
    pub(super) epsilon: F,
    pub(super) warm_start: Option<&'a SKSGDRegressorCoefficients<F>>,
    pub(super) generator: SKRandomGenerator,
    pub(super) state: Option<SKLearningState<F>>,
    pub(super) adapter: SKSGDRegressorLossAdapter<F>,
    pub(super) columns: usize,
    pub(super) batches_this_pass: usize,
    pub(super) objective: SKEpochObjective<F>,
    pub(super) error: Option<SKError>,
}

impl<F: SKFloat> SKRegressorStreamState<'_, F> {
    /// Advance one pass by one batch, folding its objective.
    pub(super) fn update(&mut self, batch: SKSupervisedBatch<F>) -> SKStreamDecision {
        match self.apply_batch(batch) {
            Ok(()) => SKStreamDecision::Continue,
            Err(error) => {
                self.error = Some(error);
                SKStreamDecision::Stop
            }
        }
    }

    fn apply_batch(&mut self, batch: SKSupervisedBatch<F>) -> Result<(), SKError> {
        let rows_count = batch.number_of_rows();
        let columns_count = batch.number_of_columns();
        if rows_count == 0 {
            return Ok(());
        }
        if self.columns != 0 && self.columns != columns_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.columns],
                found: vec![columns_count],
            });
        }
        self.columns = columns_count;
        self.batches_this_pass += 1;
        let targets = batch.targets();
        if targets.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![targets.len()],
            });
        }
        if self.state.is_none() {
            self.state = Some(sk_create_state(
                self.config,
                self.loss,
                self.epsilon,
                self.warm_start,
                self.columns,
                targets.as_slice().expect("a batch target is contiguous"),
            )?);
        }
        let live = self
            .state
            .as_mut()
            .expect("the state exists after the first batch");
        let weights = vec![F::one(); rows_count];
        let order = sk_batch_order(self.config, rows_count, &mut self.generator);
        let batch_objective =
            sk_run_batch_epoch(live, &batch, &weights, &self.adapter, self.config, &order)?;
        self.objective.merge(&batch_objective);
        Ok(())
    }
}

/// Shuffle within a batch, with the seeded generator continued across passes.
fn sk_batch_order<F: SKFloat>(
    config: &SKSGDConfig<F>,
    rows: usize,
    generator: &mut SKRandomGenerator,
) -> Vec<usize> {
    if config.shuffle {
        generator.shuffled_indices(rows)
    } else {
        (0..rows).collect()
    }
}
