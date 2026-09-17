//! The streaming entry point: plan dispatch, batch collection and the driver
//! run that finalizes the dense-equivalent model.

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{
    SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat, SKLazySource,
    SKOperationAttributes, SKOperationKind,
    execution::{SKAccessPattern, sk_run_streaming_driver},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::super::core_implementation::{SKStandardScaler, standardize};
use super::super::model::SKStandardScalerModel;
use super::batches::update_moments;
use super::state::StreamingMoments;

impl SKStandardScaler {
    /// Fit on a sequential batch source, returning the same model the dense
    /// `fit` would produce for the concatenated data.
    ///
    /// Under an in-memory execution intent the batches are collected and run
    /// through the dense path; otherwise the streaming driver updates a
    /// Welford state per batch. Every batch is instrumented as `PartialFit`.
    pub fn fit_streaming<F, S>(&self, source: &mut S) -> Result<SKStandardScalerModel<F>, SKError>
    where
        F: SKFloat,
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        // Single-pass source: totals are unknown upfront, so the outer span
        // carries zero dimensions while each batch reports its real ones.
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows: 0,
            columns: 0,
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
        context.scalar_size_bytes = 8;
        context.dataset_elements = None;
        context.access_pattern = SKAccessPattern::Sequential;
        context.grain = SK_AXIS_SUM_GRAIN;
        self.resolve_plan(&context)
    }

    /// Collect every batch and reuse the dense in-memory path.
    fn fit_collected<F, S>(&self, source: &mut S) -> Result<SKStandardScalerModel<F>, SKError>
    where
        F: SKFloat,
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let mut batches: Vec<Array2<F>> = Vec::new();
        let mut columns_count = None;
        while let Some(batch) = source.next_batch().map_err(Into::into)? {
            let data = batch.data();
            match columns_count {
                None => columns_count = Some(data.ncols()),
                Some(expected) if expected != data.ncols() => {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![expected],
                        found: vec![data.ncols()],
                    });
                }
                _ => {}
            }
            batches.push(data.to_owned());
        }
        let Some(columns) = columns_count else {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        };
        let views: Vec<ArrayView2<F>> = batches.iter().map(|batch| batch.view()).collect();
        let concatenated = ndarray::concatenate(Axis(0), &views)
            .map_err(|_| SKError::Conversion("batch concatenation failed".into()))?;
        debug_assert_eq!(concatenated.ncols(), columns);
        let (raw_mean, raw_deviations) = standardize(self, &concatenated.view())?;
        Ok(SKStandardScalerModel::from_moments(
            concatenated.nrows() as u64,
            raw_mean,
            raw_deviations,
            self.with_mean,
            self.with_std,
            self.backend,
            self.execution_intent(),
        ))
    }

    /// Drive the Welford state through the prefetch pipeline, then finalize
    /// the same mean/scale arrays the dense path produces.
    fn fit_via_driver<F, S>(
        &self,
        source: &mut S,
        plan: &SKExecutionPlan,
    ) -> Result<SKStandardScalerModel<F>, SKError>
    where
        F: SKFloat,
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let machine = SKExecutionContext::real();
        let mut state = StreamingMoments::empty(machine.available_memory_bytes, machine.cpu_cores);
        sk_run_streaming_driver(
            source,
            &mut state,
            |batch, _, state| update_moments(self, batch, state),
            plan,
        )?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        Ok(SKStandardScalerModel::from_moments(
            state.count,
            state.mean.clone(),
            state.sum_squared_deviations.clone(),
            self.with_mean,
            self.with_std,
            self.backend,
            self.execution_intent(),
        ))
    }
}
