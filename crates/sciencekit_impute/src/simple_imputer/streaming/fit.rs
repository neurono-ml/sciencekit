//! The streaming entry point: plan dispatch, batch collection and the driver
//! run that assembles the model (spec `simple-imputer`).

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{
    SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat, SKLazySource,
    SKOperationAttributes, SKOperationKind, SKUnsupervisedFit,
    execution::{SKAccessPattern, sk_run_streaming_driver},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::super::core_implementation::{SKSimpleImputer, empty_columns};
use super::super::model::SKSimpleImputerModel;
use super::batches::update_imputation;
use super::state::StreamingImputation;
use crate::missing_indicator::SKMissingIndicatorModel;

impl<F: SKFloat> SKSimpleImputer<F> {
    /// Fit on a sequential batch source.
    ///
    /// Under an in-memory execution intent the batches are collected and run
    /// through the dense path, so the result is exactly the concatenated fit;
    /// otherwise the streaming driver feeds each batch into the incremental
    /// state, which is exact for `Mean`, `MostFrequent` and `Constant` and
    /// within the median sketch's documented tolerance. Every batch is
    /// instrumented as `PartialFit`.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKSimpleImputerModel<F>, SKError>
    where
        S: SKLazySource<F> + Send,
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
    fn fit_collected<S>(&self, source: &mut S) -> Result<SKSimpleImputerModel<F>, SKError>
    where
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
        let Some(columns_count) = columns_count else {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        };
        if batches.iter().all(|batch| batch.nrows() == 0) {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        let views: Vec<ArrayView2<F>> = batches.iter().map(|batch| batch.view()).collect();
        let concatenated = ndarray::concatenate(Axis(0), &views)
            .map_err(|_| SKError::Conversion("batch concatenation failed".into()))?;
        debug_assert_eq!(concatenated.ncols(), columns_count);
        self.fit(concatenated.view())
    }

    /// Drive the incremental state through the prefetch pipeline, then
    /// assemble the model from the accumulated statistics.
    fn fit_via_driver<S>(
        &self,
        source: &mut S,
        plan: &SKExecutionPlan,
    ) -> Result<SKSimpleImputerModel<F>, SKError>
    where
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let mut state = StreamingImputation::empty();
        sk_run_streaming_driver(
            source,
            &mut state,
            |batch, _, state| update_imputation(self, batch, state),
            plan,
        )?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.rows_count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        let Some(imputation) = state.imputation.take() else {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        };
        let statistics = imputation.statistics(self.strategy, self.fill_value);
        let dropped_columns = empty_columns(&statistics, self.keep_empty_features);
        let indicator = if self.add_indicator {
            let scope: Vec<usize> = state
                .missing_columns
                .iter()
                .enumerate()
                .filter_map(|(column, &missing)| missing.then_some(column))
                .collect();
            Some(SKMissingIndicatorModel::from_scope(
                scope,
                state.columns_count,
                &self.indicator_estimator()?,
            ))
        } else {
            None
        };
        Ok(SKSimpleImputerModel::from_state(
            statistics,
            dropped_columns,
            state.columns_count,
            imputation,
            indicator,
            self,
        ))
    }
}
