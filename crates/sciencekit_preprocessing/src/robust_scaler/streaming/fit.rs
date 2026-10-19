//! The streaming entry point: plan dispatch, batch collection and the driver
//! run that finalizes the sketch-derived model.

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{
    SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat, SKLazySource,
    SKOperationAttributes, SKOperationKind,
    execution::{SKAccessPattern, sk_run_streaming_driver},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::super::configuration::finalize_statistics;
use super::super::core_implementation::{SKRobustScaler, robust_statistics};
use super::super::model::SKRobustScalerModel;
use super::super::sketch::sketch_statistics;
use super::batches::update_sketches;
use super::state::StreamingSketches;

impl<F: SKFloat> SKRobustScaler<F> {
    /// Fit on a sequential batch source, returning a model whose center and
    /// scale approximate the dense fit within the documented tolerance.
    ///
    /// Under an in-memory execution intent the batches are collected and run
    /// through the dense path; otherwise the streaming driver feeds each batch
    /// into the online sketch. Every batch is instrumented as `PartialFit`.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKRobustScalerModel<F>, SKError>
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
        context.scalar_size_bytes = 8;
        context.dataset_elements = None;
        context.access_pattern = SKAccessPattern::Sequential;
        context.grain = SK_AXIS_SUM_GRAIN;
        self.resolve_plan(&context)
    }

    /// Collect every batch and reuse the dense in-memory path.
    fn fit_collected<S>(&self, source: &mut S) -> Result<SKRobustScalerModel<F>, SKError>
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
        let Some(columns) = columns_count else {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        };
        let views: Vec<ArrayView2<F>> = batches.iter().map(|batch| batch.view()).collect();
        let concatenated = ndarray::concatenate(Axis(0), &views)
            .map_err(|_| SKError::Conversion("batch concatenation failed".into()))?;
        debug_assert_eq!(concatenated.ncols(), columns);
        let (center, scale, sketches) = robust_statistics(self, &concatenated.view())?;
        Ok(SKRobustScalerModel::from_sketches(
            concatenated.nrows() as u64,
            center,
            scale,
            sketches,
            self.configuration(),
            self.backend,
            self.execution_intent(),
        ))
    }

    /// Drive the online sketches through the prefetch pipeline, then finalize
    /// the sketch-derived model.
    fn fit_via_driver<S>(
        &self,
        source: &mut S,
        plan: &SKExecutionPlan,
    ) -> Result<SKRobustScalerModel<F>, SKError>
    where
        S: SKLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let mut state = StreamingSketches::empty();
        sk_run_streaming_driver(
            source,
            &mut state,
            |batch, _, state| update_sketches(self, batch, state),
            plan,
        )?;
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if state.count == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        let (raw_center, raw_scale) = sketch_statistics(&state.sketches, &self.configuration());
        let (center, scale) = finalize_statistics(&self.configuration(), &raw_center, &raw_scale);
        Ok(SKRobustScalerModel::from_sketches(
            state.count,
            center,
            scale,
            state.sketches,
            self.configuration(),
            self.backend,
            self.execution_intent(),
        ))
    }
}
