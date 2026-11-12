//! Multi-epoch out-of-core streaming fit over [`SKSourceProvider`] through the
//! shared prefetch executor (spec `mini-batch-k-means`, decisions 5 and 6).

use sciencekit_common::{
    SKDataBatch, SKError, SKExecutionContext, SKExecutionPlan, SKFloat, SKOperationAttributes,
    SKOperationKind, SKSourceProvider,
    execution::{
        SKAccessPattern, SKStreamDecision, SKStreamingPassPolicy, sk_resolve_execution_plan,
        sk_run_streaming_provider,
    },
    sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::config::SKMiniBatchKMeansConfiguration;
use super::core_implementation::SKMiniBatchKMeans;
use super::init::sk_initialize_minibatch_centers;
use super::model::SKMiniBatchKMeansModel;
use super::state::SKMiniBatchState;
use super::validation::{sk_validate_continuation_batch, sk_validate_initial_batch};

struct SKMiniBatchStreamDriver<F: SKFloat> {
    configuration: SKMiniBatchKMeansConfiguration<F>,
    inner: Option<SKMiniBatchState<F>>,
    feature_count: usize,
    error: Option<SKError>,
}

impl<F: SKFloat> SKMiniBatchStreamDriver<F> {
    fn new(configuration: SKMiniBatchKMeansConfiguration<F>) -> Self {
        Self {
            configuration,
            inner: None,
            feature_count: 0,
            error: None,
        }
    }

    fn process_batch(&mut self, batch: SKDataBatch<F>) -> SKStreamDecision {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count: batch.number_of_rows(),
            columns_count: batch.number_of_columns(),
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        let step_result = sk_run_operation(attributes, || -> Result<bool, SKError> {
            let view = batch.view();
            match self.inner.as_mut() {
                None => {
                    sk_validate_initial_batch(
                        view,
                        self.configuration.number_of_clusters,
                        "fit_streaming",
                    )?;
                    let (centers, _) = sk_initialize_minibatch_centers(view, &self.configuration)?;
                    let mut state = SKMiniBatchState::new(centers, self.configuration);
                    let converged = state.apply_batch(view)?;
                    self.feature_count = batch.number_of_columns();
                    self.inner = Some(state);
                    Ok(converged)
                }
                Some(state) => {
                    sk_validate_continuation_batch(
                        view,
                        self.feature_count,
                        self.configuration.number_of_clusters,
                        false,
                        "fit_streaming",
                    )?;
                    state.apply_batch(view)
                }
            }
        });

        match step_result {
            Ok(true) => SKStreamDecision::Stop,
            Ok(false) => SKStreamDecision::Continue,
            Err(error) => {
                self.error = Some(error);
                SKStreamDecision::Stop
            }
        }
    }
}

impl<F: SKFloat> SKMiniBatchKMeans<F> {
    /// Fits the estimator over a sequential or reopenable batch provider
    /// through the shared prefetch executor.
    pub fn fit_streaming<Provider>(
        &self,
        mut provider: Provider,
    ) -> Result<SKMiniBatchKMeansModel<F>, SKError>
    where
        Provider: SKSourceProvider<F>,
    {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: 0,
            columns_count: 0,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            let plan = resolve_streaming_plan::<F>(self.configuration.execution_mode)?;
            let mut driver = SKMiniBatchStreamDriver::new(self.configuration);
            let epochs_consumed = sk_run_streaming_provider(
                &mut provider,
                &mut driver,
                |batch, _parallelism, state| state.process_batch(batch),
                &plan,
                SKStreamingPassPolicy::Iterative {
                    maximum_number_of_passes: self.configuration.maximum_number_of_epochs,
                },
            )?;
            if let Some(error) = driver.error.take() {
                return Err(error);
            }
            let Some(state) = driver.inner.take() else {
                return Err(SKError::EmptyInput {
                    operation: "fit_streaming",
                });
            };
            let inertia = state.last_batch_inertia;
            Ok(SKMiniBatchKMeansModel::from_state(
                state,
                None,
                inertia,
                epochs_consumed,
            ))
        })
    }
}

fn resolve_streaming_plan<F: SKFloat>(
    execution_mode: sciencekit_common::SKExecutionMode,
) -> Result<SKExecutionPlan, SKError> {
    let mut context = SKExecutionContext::real();
    context.dataset_size_bytes = u64::MAX;
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = None;
    context.access_pattern = SKAccessPattern::Iterative;
    context.grain = SK_AXIS_SUM_GRAIN;
    sk_resolve_execution_plan(execution_mode, &context)
}
