//! The single-pass out-of-core fit (spec `stochastic-gradient-classifier`,
//! design decision 8).

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLabelTable, SKOperationAttributes,
    SKOperationKind, SKSupervisedLazySource, sk_run_operation,
};

use super::super::coefficients::SKSGDClassifierCoefficients;
use super::super::model::SKSGDClassifierModel;
use super::super::targets::{
    sk_head_targets, sk_integer_label, sk_label_table_from_integers, sk_positive_classes,
    sk_validate_number_of_classes,
};
use super::state::{sk_create_states, sk_reported_parts};
use crate::class_weight::{SKClassWeight, sk_expand_class_weights};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{
    SKLearningState, SKRandomGenerator, SKSGDClassifierLoss, sk_finish_epochs, sk_run_dense_epoch,
};

/// Fit on a one-shot supervised source in a single pass.
///
/// Every class must appear in the first non-empty batch: the head structure is
/// fixed there and a later first appearance would require replaying the rows
/// already consumed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_fit_streaming<F, S>(
    source: &mut S,
    config: &SKSGDConfig<F>,
    loss: SKSGDClassifierLoss,
    class_weight: &SKClassWeight<F>,
    warm_start: Option<&SKSGDClassifierCoefficients<F>>,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
) -> Result<SKSGDClassifierModel<F>, SKError>
where
    F: SKFloat,
    S: SKSupervisedLazySource<F> + Send,
    S::Error: From<SKError> + Into<SKError>,
{
    if config.early_stopping {
        return Err(SKError::UnsupportedOperation {
            operation: "early_stopping",
            reason: "a one-shot stream has no epochs to compare; use an in-memory fit".to_string(),
        });
    }
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let mut generator = SKRandomGenerator::new(config.random_seed);
        let mut classes: Option<SKLabelTable> = None;
        let mut states: Option<Vec<SKLearningState<F>>> = None;
        let mut columns = 0usize;
        let mut batches_seen = 0usize;
        while let Some(batch) = source.next_supervised_batch().map_err(Into::into)? {
            let features = batch.features();
            if features.nrows() == 0 {
                continue;
            }
            if columns != 0 && columns != features.ncols() {
                return Err(SKError::ShapeMismatch {
                    expected: vec![columns],
                    found: vec![features.ncols()],
                });
            }
            columns = features.ncols();
            batches_seen += 1;
            let labels: Vec<i64> = batch
                .targets()
                .iter()
                .map(|value| sk_integer_label(*value))
                .collect();
            let table = sk_first_batch_classes(&labels, &mut classes)?;
            let class_indices = sk_stream_class_indices(&labels, table)?;
            if states.is_none() {
                let created =
                    sk_create_states(config, loss, warm_start, table, columns, &class_indices)?;
                states = Some(created);
            }
            let live = states.as_mut().expect("states exist after the first batch");
            let number_of_classes = table.number_of_classes();
            let weights = sk_expand_class_weights(class_weight, &class_indices, number_of_classes)?;
            let positive_classes = sk_positive_classes(number_of_classes);
            for head in 0..live.len() {
                let targets =
                    Array1::from_vec(sk_head_targets(&class_indices, positive_classes[head]));
                let order = sk_batch_order(config, features.nrows(), &mut generator);
                sk_run_dense_epoch(
                    &mut live[head],
                    features,
                    targets.view(),
                    &weights,
                    &loss,
                    config,
                    &order,
                )?;
            }
        }
        if batches_seen == 0 {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        sk_finish_epochs(
            config.convergence_policy,
            config.maximum_number_of_iterations,
            1,
            false,
            true,
        )?;
        let classes = classes.expect("a non-empty stream produced a class table");
        let states = states.expect("a non-empty stream produced its heads");
        let (coefficients, intercepts) = sk_reported_parts(&states, columns);
        Ok(SKSGDClassifierModel::from_parts(
            coefficients,
            intercepts,
            classes,
            states,
            config,
            loss,
            class_weight,
            backend,
            execution_intent,
            false,
            1,
        ))
    })
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
                        "class `{value}` first appears after the first batch; a one-pass streaming fit requires every class in the first batch"
                    ),
                })
        })
        .collect()
}

/// Shuffle within a batch only: a one-shot source cannot be rewound.
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
