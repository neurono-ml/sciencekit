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

use super::super::core_implementation::{SKMinMaxScaler, column_extrema};
use super::super::model::SKMinMaxScalerModel;
use super::batches::update_extrema;
use super::state::StreamingExtrema;

impl<F: SKFloat> SKMinMaxScaler<F> {
    /// Fit on a sequential batch source, returning the same model the dense
    /// `fit` would produce for the concatenated data.
    ///
    /// Under an in-memory execution intent the batches are collected and run
    /// through the dense path; otherwise the streaming driver updates running
    /// extrema per batch. Every batch is instrumented as `PartialFit`.
    pub fn fit_streaming<Provider>(
        &self,
        mut provider: Provider,
    ) -> Result<SKMinMaxScalerModel<F>, SKError>
    where
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
    fn fit_collected<S>(&self, source: &mut S) -> Result<SKMinMaxScalerModel<F>, SKError>
    where
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
        let (minimum, maximum) = column_extrema(self, &concatenated.view())?;
        Ok(SKMinMaxScalerModel::from_extrema(
            concatenated.nrows() as u64,
            minimum,
            maximum,
            self.feature_range(),
            self.clip,
            self.backend,
            self.execution_intent(),
        ))
    }

    /// Accumulate sparse batches through the same extrema state the driver
    /// uses; the concatenated `fit` is dense-only, so the sparse in-memory
    /// branch never densifies a block.
    fn fit_collected_sparse(
        &self,
        batches: Vec<SKDataBatch<F>>,
    ) -> Result<SKMinMaxScalerModel<F>, SKError> {
        let mut state = StreamingExtrema::empty(u64::MAX, 1);
        for batch in batches {
            update_extrema(self, batch, &mut state);
        }
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        Ok(SKMinMaxScalerModel::from_extrema(
            state.count,
            state.minimum.clone(),
            state.maximum.clone(),
            self.feature_range(),
            self.clip,
            self.backend,
            self.execution_intent(),
        ))
    }

    /// Drive the running extrema through the prefetch pipeline, then finalize
    /// the same model the dense path produces.
    fn fit_via_driver<Provider>(
        &self,
        provider: &mut Provider,
        plan: &SKExecutionPlan,
    ) -> Result<SKMinMaxScalerModel<F>, SKError>
    where
        Provider: SKSourceProvider<F>,
    {
        let machine = SKExecutionContext::real();
        let mut state = StreamingExtrema::empty(machine.available_memory_bytes, machine.cpu_cores);
        sk_run_streaming_provider(
            provider,
            &mut state,
            |batch, _, state| update_extrema(self, batch, state),
            plan,
            SKStreamingPassPolicy::SinglePass,
        )?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        Ok(SKMinMaxScalerModel::from_extrema(
            state.count,
            state.minimum.clone(),
            state.maximum.clone(),
            self.feature_range(),
            self.clip,
            self.backend,
            self.execution_intent(),
        ))
    }
}
