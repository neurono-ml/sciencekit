//! Out-of-core fit orchestration for the logistic estimators (spec
//! `logistic-regression`, Decision 7).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
    SKLabelTable, SKOperationAttributes, SKOperationKind, SKSupervisedSourceProvider,
    execution::{
        SKAccessPattern, SKStreamingPassPolicy, sk_resolve_execution_plan,
        sk_run_supervised_streaming_provider,
    },
    sk_canonicalize_labels, sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::accumulator::SKLogisticStreamAccumulator;
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
            |batch, _, accumulator| accumulator.update(batch),
            &plan,
            SKStreamingPassPolicy::SinglePass,
        )?;
        if let Some(error) = accumulator.error.take() {
            return Err(error);
        }
        if accumulator.batches.is_empty() {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        let labels = &accumulator.labels;
        let borrowed: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (labels, classes): (Vec<usize>, SKLabelTable) = sk_canonicalize_labels(&borrowed);
        let number_of_classes = classes.number_of_classes();
        let strategy =
            sk_resolve_multi_class(config.multi_class, config.solver, number_of_classes)?;
        let state = accumulator.into_state(&labels);
        debug_assert_eq!(state.row_count() as usize, labels.len());
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

/// The decimal representation of a float-encoded class label, rounded to the
/// nearest integer (matching the dense integer canonicalization).
pub(super) fn sk_float_label<F: SKFloat>(value: F) -> String {
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
