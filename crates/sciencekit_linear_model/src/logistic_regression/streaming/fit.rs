//! Out-of-core fit orchestration for the logistic estimators (spec
//! `logistic-regression`, Decision 7).

use ndarray::Array2;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
    SKLabelTable, SKOperationAttributes, SKOperationKind, SKSupervisedBatch,
    SKSupervisedSourceProvider,
    execution::{
        SKAccessPattern, SKStreamDecision, SKStreamingPassPolicy, sk_resolve_execution_plan,
        sk_run_supervised_streaming_provider,
    },
    sk_canonicalize_labels, sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

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
pub(crate) fn sk_fit_streaming<F, Provider>(
    mut provider: Provider,
    config: &SKLogisticConfig<F>,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
) -> Result<SKLogisticRegressionModel<F>, SKError>
where
    F: SKFloat,
    Provider: SKSupervisedSourceProvider<F>,
{
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let plan = sk_resolve_streaming_plan()?;
        let mut accumulator = SKLogisticStreamAccumulator::new();
        sk_run_supervised_streaming_provider(
            &mut provider,
            &mut accumulator,
            |batch, parallelism, accumulator| accumulator.update(batch, parallelism),
            &plan,
            SKStreamingPassPolicy::SinglePass,
        )?;
        if let Some(error) = accumulator.error.take() {
            return Err(error);
        }
        let SKLogisticStreamAccumulator {
            batches, labels, ..
        } = accumulator;
        if batches.is_empty() {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        let borrowed: Vec<&str> = labels.iter().map(String::as_str).collect();
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

/// The single-pass accumulator: every batch's design and integer-coded labels.
struct SKLogisticStreamAccumulator<F> {
    batches: Vec<Array2<F>>,
    labels: Vec<String>,
    columns_count: usize,
    error: Option<SKError>,
}

impl<F: SKFloat> SKLogisticStreamAccumulator<F> {
    fn new() -> Self {
        Self {
            batches: Vec::new(),
            labels: Vec::new(),
            columns_count: 0,
            error: None,
        }
    }

    fn update(&mut self, batch: SKSupervisedBatch<F>, _parallelism: usize) -> SKStreamDecision {
        match self.accumulate(batch) {
            Ok(()) => SKStreamDecision::Continue,
            Err(error) => {
                self.error = Some(error);
                SKStreamDecision::Stop
            }
        }
    }

    fn accumulate(&mut self, batch: SKSupervisedBatch<F>) -> Result<(), SKError> {
        let features = batch.features();
        if features.nrows() == 0 {
            return Ok(());
        }
        if self.columns_count != 0 && self.columns_count != features.ncols() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.columns_count],
                found: vec![features.ncols()],
            });
        }
        self.columns_count = features.ncols();
        let targets = batch.targets();
        if targets.len() != features.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![features.nrows()],
                found: vec![targets.len()],
            });
        }
        targets
            .iter()
            .for_each(|value| self.labels.push(sk_float_label(*value)));
        self.batches.push(features.to_owned());
        Ok(())
    }
}

/// Resolve the plan for an unbounded stream (Sequential access pattern).
fn sk_resolve_streaming_plan() -> Result<SKExecutionPlan, SKError> {
    let mut context = SKExecutionContext::real();
    context.dataset_size_bytes = u64::MAX;
    context.scalar_size_bytes = 8;
    context.dataset_elements = None;
    context.access_pattern = SKAccessPattern::Sequential;
    context.grain = SK_AXIS_SUM_GRAIN;
    sk_resolve_execution_plan(SKExecutionMode::Automatic, &context)
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
