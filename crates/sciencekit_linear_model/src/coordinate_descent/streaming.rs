//! Out-of-core fit orchestration shared by `SKLasso` and `SKElasticNet`
//! (spec `coordinate-descent-solver`, Decision 7).

use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedBatch, SKSupervisedSourceProvider,
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
        let view = batch.view();
        let (rows_count, columns_count) = match view {
            SKDataView::Dense(features) => features.dim(),
            SKDataView::Sparse(features) => (features.rows(), features.cols()),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        let targets = batch.targets();
        if rows_count == 0 {
            return Ok(());
        }
        if targets.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![targets.len()],
            });
        }
        let accumulated_columns = self.state.columns_count();
        if accumulated_columns != 0 && accumulated_columns != columns_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![accumulated_columns],
                found: vec![columns_count],
            });
        }
        match view {
            SKDataView::Dense(features) => {
                if matches!(self.state, SKCoordinateDescentState::Sparse { .. }) {
                    self.state.accumulate_dense_into_sparse(features, targets);
                } else {
                    self.state.accumulate_dense(features, targets);
                }
            }
            SKDataView::Sparse(features) => {
                self.state.ensure_sparse();
                self.state.accumulate_sparse(features, targets);
            }
            _ => unreachable!("the shape check rejected every other representation"),
        }
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
