//! Out-of-core fit orchestration for the logistic estimators (spec
//! `logistic-regression`, Decision 7).

use ndarray::Array2;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLabelTable, SKOperationAttributes,
    SKOperationKind, SKSupervisedLazySource, sk_canonicalize_labels, sk_run_operation,
};

use crate::logistic::SKLogisticState;
use crate::logistic::config::SKLogisticConfig;
use crate::logistic::resolution::sk_resolve_multi_class;
use crate::logistic_regression::model::SKLogisticRegressionModel;

/// Fit on a sequential supervised source, retaining each batch's design and
/// solving the whole accumulated design once at the end.
///
/// The result equals the dense `fit` on the concatenated data within floating
/// point tolerance, regardless of batch order or size. Batch targets are
/// integer-coded class labels. An empty source is rejected with a structured
/// error.
pub(crate) fn sk_fit_streaming<F, S>(
    source: &mut S,
    config: &SKLogisticConfig<F>,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
) -> Result<SKLogisticRegressionModel<F>, SKError>
where
    F: SKFloat,
    S: SKSupervisedLazySource<F> + Send,
    S::Error: From<SKError> + Into<SKError>,
{
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let (batches, raw_labels) = collect_batches(source)?;
        if batches.is_empty() {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        let borrowed: Vec<&str> = raw_labels.iter().map(String::as_str).collect();
        let (labels, classes): (Vec<usize>, SKLabelTable) = sk_canonicalize_labels(&borrowed);
        let number_of_classes = classes.number_of_classes();
        let strategy =
            sk_resolve_multi_class(config.multi_class, config.solver, number_of_classes)?;
        let mut state = SKLogisticState::new_dense();
        let mut offset = 0;
        for batch in &batches {
            let rows = batch.nrows();
            state.accumulate_dense(batch.view(), &labels[offset..offset + rows]);
            offset += rows;
        }
        let solution = state.solve(config, strategy, number_of_classes, None)?;
        Ok(SKLogisticRegressionModel::from_parts(
            solution,
            classes,
            strategy,
            number_of_classes,
            config.clone(),
            backend,
            execution_intent,
            state,
        ))
    })
}

/// Retain every batch's design and its integer-coded labels.
fn collect_batches<F, S>(source: &mut S) -> Result<(Vec<Array2<F>>, Vec<String>), SKError>
where
    F: SKFloat,
    S: SKSupervisedLazySource<F> + Send,
    S::Error: From<SKError> + Into<SKError>,
{
    let mut batches = Vec::new();
    let mut labels = Vec::new();
    let mut columns_count = 0usize;
    while let Some(batch) = source.next_supervised_batch().map_err(Into::into)? {
        let features = batch.features();
        if features.nrows() == 0 {
            continue;
        }
        if columns_count != 0 && columns_count != features.ncols() {
            return Err(SKError::ShapeMismatch {
                expected: vec![columns_count],
                found: vec![features.ncols()],
            });
        }
        columns_count = features.ncols();
        let targets = batch.targets();
        if targets.len() != features.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![features.nrows()],
                found: vec![targets.len()],
            });
        }
        targets
            .iter()
            .for_each(|value| labels.push(sk_float_label(*value)));
        batches.push(features.to_owned());
    }
    Ok((batches, labels))
}

/// The decimal representation of a float-encoded class label, rounded to the
/// nearest integer (matching the dense integer canonicalization).
fn sk_float_label<F: SKFloat>(value: F) -> String {
    let rounded = value.round();
    let negative = rounded < F::zero();
    let mut magnitude = rounded.abs();
    let mut integer: i64 = 0;
    while magnitude >= F::one() {
        magnitude = magnitude - F::one();
        integer += 1;
    }
    if negative {
        (-integer).to_string()
    } else {
        integer.to_string()
    }
}
