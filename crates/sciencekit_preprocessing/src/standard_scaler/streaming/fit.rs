//! The streaming entry point: plan dispatch, batch collection and the driver
//! run that finalizes the dense-equivalent model.

use ndarray::{ArrayView2, Axis};
use sciencekit_common::{
    SKDataBatch, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
    SKLazySource, SKOperationAttributes, SKOperationKind, SKSourceProvider,
    execution::{SKAccessPattern, SKStreamingPassPolicy, sk_run_streaming_provider},
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
    pub fn fit_streaming<F, Provider>(
        &self,
        mut provider: Provider,
    ) -> Result<SKStandardScalerModel<F>, SKError>
    where
        F: SKFloat,
        Provider: SKSourceProvider<F>,
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
                let mut source = provider.open()?;
                self.fit_collected(&mut source)
            } else {
                self.fit_via_driver(&mut provider, &plan)
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
        S::Error: Into<SKError>,
    {
        let mut batches: Vec<SKDataBatch<F>> = Vec::new();
        let mut columns_count = None;
        let mut any_sparse = false;
        while let Some(batch) = source.next_batch().map_err(Into::into)? {
            match columns_count {
                None => columns_count = Some(batch.number_of_columns()),
                Some(expected) if expected != batch.number_of_columns() => {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![expected],
                        found: vec![batch.number_of_columns()],
                    });
                }
                _ => {}
            }
            any_sparse |= batch.is_sparse();
            batches.push(batch);
        }
        let Some(columns) = columns_count else {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        };
        if any_sparse {
            return self.fit_collected_sparse(batches);
        }
        let dense_batches: Vec<ndarray::Array2<F>> = batches
            .iter()
            .map(|batch| {
                batch
                    .view()
                    .as_dense()
                    .expect("the stream had no sparse block")
                    .to_owned()
            })
            .collect();
        let views: Vec<ArrayView2<F>> = dense_batches.iter().map(|batch| batch.view()).collect();
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

    /// Collect sparse batches through the same moment state the driver uses:
    /// the concatenated `fit` is dense-only, so the sparse in-memory branch
    /// accumulates the identical per-batch moments instead of densifying.
    fn fit_collected_sparse<F>(
        &self,
        batches: Vec<SKDataBatch<F>>,
    ) -> Result<SKStandardScalerModel<F>, SKError>
    where
        F: SKFloat,
    {
        let mut state = StreamingMoments::empty(u64::MAX, 1);
        for batch in batches {
            update_moments(self, batch, &mut state);
        }
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

    /// Drive the Welford state through the prefetch pipeline, then finalize
    /// the same mean/scale arrays the dense path produces.
    fn fit_via_driver<F, Provider>(
        &self,
        provider: &mut Provider,
        plan: &SKExecutionPlan,
    ) -> Result<SKStandardScalerModel<F>, SKError>
    where
        F: SKFloat,
        Provider: SKSourceProvider<F>,
    {
        let machine = SKExecutionContext::real();
        let mut state = StreamingMoments::empty(machine.available_memory_bytes, machine.cpu_cores);
        sk_run_streaming_provider(
            provider,
            &mut state,
            |batch, _, state| update_moments(self, batch, state),
            plan,
            SKStreamingPassPolicy::SinglePass,
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
