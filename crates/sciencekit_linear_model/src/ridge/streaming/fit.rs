//! The streaming entry point: plan dispatch, batch collection and the driver run
//! that finalizes the dense-equivalent model (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{
    SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKOperationAttributes,
    SKOperationKind, SKSupervisedLazySource, execution::SKAccessPattern,
    execution::sk_run_supervised_streaming_driver, sk_run_operation,
};
use sciencekit_math::SKBackendSelection;
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::batches::update_gram;
use super::state::StreamingCenteredGramState;
use crate::ridge::core_implementation::SKRidge;
use crate::ridge::model::SKRidgeModel;
use crate::ridge::solving::solve_from_gram;

impl<F: SKBackendSelection> SKRidge<F> {
    /// Fit on a sequential supervised source, returning the same model the
    /// dense `fit` would produce for the concatenated data.
    ///
    /// Under an in-memory execution intent the batches are collected and run
    /// through the dense path; otherwise the supervised streaming driver
    /// accumulates centered cross products per batch and solves once at the
    /// end, recording the `Cholesky` Gram solver actually used. Every batch is
    /// instrumented as `PartialFit`.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKRidgeModel<F>, SKError>
    where
        S: SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        // Single-pass source: totals are unknown upfront, so the outer span
        // carries zero dimensions while each batch reports its real ones.
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: 0,
            columns_count: 0,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let plan = self.resolve_streaming_plan()?;
            if plan.mode == SKExecutionMode::InProcessSynchronous
                || plan.mode == SKExecutionMode::InProcessAsynchronous
            {
                self.fit_collected(source)
            } else {
                self.fit_via_driver(source, &plan)
            }
        })
    }

    /// Resolve the plan for an unbounded stream: declare a dataset larger than
    /// memory so `Automatic` takes the streaming branch (`Sequential` pattern).
    fn resolve_streaming_plan(&self) -> Result<SKExecutionPlan, SKError> {
        let mut context = SKExecutionContext::real();
        context.dataset_size_bytes = u64::MAX;
        context.scalar_size_bytes = size_of::<F>() as u64;
        context.dataset_elements = None;
        context.access_pattern = SKAccessPattern::Sequential;
        context.grain = SK_AXIS_SUM_GRAIN;
        self.resolve_plan(&context)
    }

    /// Collect every batch and reuse the dense in-memory path.
    fn fit_collected<S>(&self, source: &mut S) -> Result<SKRidgeModel<F>, SKError>
    where
        S: SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        use sciencekit_common::SKSupervisedFit;
        let mut feature_batches: Vec<Array2<F>> = Vec::new();
        let mut target_batches: Vec<Array1<F>> = Vec::new();
        let mut columns_count = None;
        while let Some(batch) = source.next_supervised_batch().map_err(Into::into)? {
            let features = batch.features();
            match columns_count {
                None => columns_count = Some(features.ncols()),
                Some(expected) if expected != features.ncols() => {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![expected],
                        found: vec![features.ncols()],
                    });
                }
                _ => {}
            }
            feature_batches.push(features.to_owned());
            target_batches.push(batch.targets().to_owned());
        }
        if columns_count.is_none() {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        let feature_views: Vec<ArrayView2<F>> =
            feature_batches.iter().map(|batch| batch.view()).collect();
        let concatenated_features = ndarray::concatenate(Axis(0), &feature_views)
            .map_err(|_| SKError::Conversion("batch concatenation failed".into()))?;
        let target_views: Vec<ArrayView1<F>> =
            target_batches.iter().map(|batch| batch.view()).collect();
        let concatenated_targets = ndarray::concatenate(Axis(0), &target_views)
            .map_err(|_| SKError::Conversion("target concatenation failed".into()))?;
        self.fit(concatenated_features.view(), concatenated_targets.view())
    }

    /// Drive the centered Gram state through the prefetch pipeline, then solve
    /// the accumulated normal equations once.
    fn fit_via_driver<S>(
        &self,
        source: &mut S,
        plan: &SKExecutionPlan,
    ) -> Result<SKRidgeModel<F>, SKError>
    where
        S: SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let machine = SKExecutionContext::real();
        let mut state =
            StreamingCenteredGramState::new(machine.available_memory_bytes, machine.cpu_cores);
        sk_run_supervised_streaming_driver(
            source,
            &mut state,
            |batch, _, state| update_gram(self, batch, state),
            plan,
        )?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.gram.row_count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        let solution = solve_from_gram(
            &state.gram,
            self.include_intercept,
            &self.alphas,
            self.backend,
        )?;
        Ok(SKRidgeModel::from_solve(
            self.alphas.clone(),
            solution.coefficients,
            solution.intercepts,
            solution.rank,
            solution.singular_values,
            solution.resolved,
            self.include_intercept,
            self.backend,
            self.execution_intent(),
            state.gram,
        ))
    }
}
