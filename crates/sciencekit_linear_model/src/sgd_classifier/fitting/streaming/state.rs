//! The mutable state shared by every pass of the classifier stream.

use ndarray::Array1;
use sciencekit_common::execution::SKStreamDecision;
use sciencekit_common::{SKError, SKFloat, SKLabelTable, SKSupervisedBatch};

use super::super::super::coefficients::SKSGDClassifierCoefficients;
use super::super::super::targets::{
    sk_head_targets, sk_integer_label, sk_label_table_from_integers, sk_positive_classes,
    sk_validate_number_of_classes,
};
use super::super::state::sk_create_states;
use crate::class_weight::{SKClassWeight, sk_expand_class_weights};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::objective::SKEpochObjective;
use crate::sgd::{SKLearningState, SKRandomGenerator, SKSGDClassifierLoss, sk_run_dense_epoch};

/// The mutable state shared by every pass of the fit.
pub(super) struct SKClassifierStreamState<'a, F> {
    pub(super) config: &'a SKSGDConfig<F>,
    pub(super) loss: SKSGDClassifierLoss,
    pub(super) class_weight: &'a SKClassWeight<F>,
    pub(super) warm_start: Option<&'a SKSGDClassifierCoefficients<F>>,
    pub(super) generator: SKRandomGenerator,
    pub(super) classes: Option<SKLabelTable>,
    pub(super) states: Option<Vec<SKLearningState<F>>>,
    pub(super) columns: usize,
    pub(super) batches_this_pass: usize,
    pub(super) objective: SKEpochObjective<F>,
    pub(super) error: Option<SKError>,
}

impl<F: SKFloat> SKClassifierStreamState<'_, F> {
    /// Advance one pass by one batch, folding its per-head objectives.
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
        let features = batch.view().as_dense()?;
        if features.nrows() == 0 {
            return Ok(());
        }
        if self.columns != 0 && self.columns != features.ncols() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.columns],
                found: vec![features.ncols()],
            });
        }
        self.columns = features.ncols();
        self.batches_this_pass += 1;
        let labels: Vec<i64> = batch
            .targets()
            .iter()
            .map(|value| sk_integer_label(*value))
            .collect();
        let table = sk_first_batch_classes(&labels, &mut self.classes)?;
        let class_indices = sk_stream_class_indices(&labels, table)?;
        if self.states.is_none() {
            self.states = Some(sk_create_states(
                self.config,
                self.loss,
                self.warm_start,
                table,
                self.columns,
                &class_indices,
            )?);
        }
        let live = self
            .states
            .as_mut()
            .expect("states exist after the first batch");
        let number_of_classes = table.number_of_classes();
        let weights =
            sk_expand_class_weights(self.class_weight, &class_indices, number_of_classes)?;
        let positive_classes = sk_positive_classes(number_of_classes);
        let order = sk_batch_order(self.config, features.nrows(), &mut self.generator);
        for head in 0..live.len() {
            let targets = Array1::from_vec(sk_head_targets(&class_indices, positive_classes[head]));
            let batch_objective = sk_run_dense_epoch(
                &mut live[head],
                features,
                targets.view(),
                &weights,
                &self.loss,
                self.config,
                &order,
            )?;
            self.objective.merge(&batch_objective);
        }
        Ok(())
    }
}

/// The class table of a stream: built from the first batch, then reused.
fn sk_first_batch_classes<'a>(
    labels: &[i64],
    classes: &'a mut Option<SKLabelTable>,
) -> Result<&'a SKLabelTable, SKError> {
    if classes.is_none() {
        let table = sk_label_table_from_integers(labels);
        sk_validate_number_of_classes(&table)?;
        *classes = Some(table);
    }
    Ok(classes.as_ref().expect("the table was just built"))
}

/// Map a batch's integer labels onto canonical indices, rejecting a class that
/// first appears after the first batch.
fn sk_stream_class_indices(labels: &[i64], classes: &SKLabelTable) -> Result<Vec<usize>, SKError> {
    labels
        .iter()
        .map(|&value| {
            classes
                .index_of(&value.to_string())
                .ok_or_else(|| SKError::InvalidHyperparameter {
                    name: "targets",
                    reason: format!(
                        "class `{value}` first appears after the first batch; a streaming fit requires every class in the first batch"
                    ),
                })
        })
        .collect()
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
