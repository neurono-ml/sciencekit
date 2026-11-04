//! Out-of-core fit orchestration shared by `SKLasso` and `SKElasticNet`
//! (spec `coordinate-descent-solver`, Decision 7).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
    SKOperationAttributes, SKOperationKind, SKSupervisedBatch, SKSupervisedSourceProvider,
    execution::{
        SKAccessPattern, SKStreamDecision, SKStreamingPassPolicy, sk_resolve_execution_plan,
        sk_run_supervised_streaming_provider,
    },
    sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::accumulator::SKCoordinateDescentState;
use super::config::SKCoordinateDescentConfig;
use super::core::SKCoordinateDescentModelCore;

/// Fit on a sequential supervised source, retaining each batch's design and
/// solving the whole accumulated design once at the end.
///
/// The result equals the dense `fit` on the concatenated data within floating
/// point tolerance, regardless of batch order or size. An empty source is
/// rejected with a structured error.
pub(crate) fn sk_fit_streaming<F, Provider>(
    mut provider: Provider,
    alphas: &[F],
    l1_ratio: F,
    include_intercept: bool,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
    config: SKCoordinateDescentConfig<F>,
) -> Result<SKCoordinateDescentModelCore<F>, SKError>
where
    F: SKFloat,
    Provider: SKSupervisedSourceProvider<F>,
{
    // Single-pass source: totals are unknown upfront, so the outer span carries
    // zero dimensions.
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let plan = sk_resolve_streaming_plan()?;
        let mut accumulator = SKCoordinateDescentStreamAccumulator::new();
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
        let state = accumulator.state;
        if state.is_empty() {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        let solution = state.solve(alphas, l1_ratio, include_intercept, &config)?;
        Ok(SKCoordinateDescentModelCore::from_solution(
            alphas.to_vec(),
            l1_ratio,
            solution,
            include_intercept,
            backend,
            execution_intent,
            config,
            state,
        ))
    })
}

/// The single-pass accumulator: one retained-design state plus a read error.
struct SKCoordinateDescentStreamAccumulator<F> {
    state: SKCoordinateDescentState<F>,
    error: Option<SKError>,
}

impl<F: SKFloat> SKCoordinateDescentStreamAccumulator<F> {
    fn new() -> Self {
        Self {
            state: SKCoordinateDescentState::new_dense(),
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
        let features = batch.view().as_dense()?;
        let targets = batch.targets();
        if features.nrows() == 0 {
            return Ok(());
        }
        if targets.len() != features.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![features.nrows()],
                found: vec![targets.len()],
            });
        }
        let columns_count = self.state.columns_count();
        if columns_count != 0 && columns_count != features.ncols() {
            return Err(SKError::ShapeMismatch {
                expected: vec![columns_count],
                found: vec![features.ncols()],
            });
        }
        self.state.accumulate_dense(features, targets);
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
