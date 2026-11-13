//! Out-of-core streaming fitting over [`SKSourceProvider`] through the shared
//! prefetch executor (spec `incremental-pca`, decision 4).

use std::mem::size_of;

use sciencekit_common::{
    SKDataBatch, SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKOperationAttributes,
    SKOperationKind, SKSourceProvider,
    execution::{
        SKAccessPattern, SKStreamDecision, SKStreamingPassPolicy, sk_run_streaming_provider,
    },
    sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, SKMathBackend, sk_math_backend_for_kind};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::accumulator::SKCovarianceAccumulator;
use super::core_implementation::SKIncrementalPCA;
use super::finalize::sk_finalize_accumulator;
use super::merge::sk_update_dense;
use super::model::SKIncrementalPCAModel;
use super::sparse::sk_update_sparse;
use crate::pca::{sk_validate_finiteness, sk_view_dimensions};

struct StreamingCovarianceState<F: sciencekit_common::SKFloat> {
    accumulator: Option<SKCovarianceAccumulator<F>>,
    error: Option<SKError>,
}

impl<F: SKBackendSelection> SKIncrementalPCA<F> {
    /// Drives `provider` through the shared prefetch executor without
    /// retaining dataset rows, running one pass for one-shot providers and up
    /// to `maximum_number_of_epochs` passes for reopenable providers.
    pub fn fit_streaming<Provider>(
        &self,
        mut provider: Provider,
    ) -> Result<SKIncrementalPCAModel<F>, SKError>
    where
        Provider: SKSourceProvider<F>,
    {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: 0,
            columns_count: 0,
            execution_mode: self.configuration.execution_mode(),
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            let plan = self.resolve_streaming_plan()?;
            let backend = sk_math_backend_for_kind::<F>(self.configuration.backend);
            let mut state = StreamingCovarianceState::<F> {
                accumulator: None,
                error: None,
            };
            let policy = SKStreamingPassPolicy::Iterative {
                maximum_number_of_passes: self.configuration.maximum_number_of_epochs,
            };
            let epochs_consumed = sk_run_streaming_provider(
                &mut provider,
                &mut state,
                |batch, parallelism, state| {
                    self.update_stream_batch(batch, parallelism, state, backend.as_ref())
                },
                &plan,
                policy,
            )?;
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            let Some(accumulator) = state.accumulator else {
                return Err(SKError::EmptyInput {
                    operation: "fit_streaming",
                });
            };
            let samples_seen = accumulator.samples_seen();
            let features_count = accumulator.number_of_features();
            let maximum_components = samples_seen.min(features_count);
            let resolved_components = self
                .configuration
                .number_of_components
                .unwrap_or(maximum_components);
            if resolved_components == 0 || resolved_components > maximum_components {
                return Err(SKError::InvalidHyperparameter {
                    name: "number_of_components",
                    reason: format!(
                        "requested {resolved_components} components, which must lie in 1..={maximum_components} for streamed totals ({samples_seen}, {features_count})"
                    ),
                });
            }
            if self.configuration.batch_size < resolved_components {
                return Err(SKError::InvalidHyperparameter {
                    name: "batch_size",
                    reason: format!(
                        "batch_size ({}) must be at least number_of_components ({resolved_components})",
                        self.configuration.batch_size
                    ),
                });
            }
            let decomposition =
                sk_finalize_accumulator(&accumulator, backend.as_ref(), resolved_components)?;
            Ok(SKIncrementalPCAModel::from_accumulator_and_decomposition(
                accumulator,
                decomposition,
                epochs_consumed,
                self.configuration,
            ))
        })
    }

    fn resolve_streaming_plan(&self) -> Result<SKExecutionPlan, SKError> {
        let mut context = SKExecutionContext::real();
        context.cpu_cores = context.cpu_cores.min(rayon::current_num_threads().max(1));
        context.dataset_size_bytes = u64::MAX;
        context.scalar_size_bytes = size_of::<F>() as u64;
        context.dataset_elements = None;
        context.access_pattern = SKAccessPattern::Sequential;
        context.grain = SK_AXIS_SUM_GRAIN;
        self.resolve_plan(&context)
    }

    fn update_stream_batch(
        &self,
        batch: SKDataBatch<F>,
        parallelism: usize,
        state: &mut StreamingCovarianceState<F>,
        backend: &dyn SKMathBackend<F>,
    ) -> SKStreamDecision {
        let view = batch.view();
        let (rows_count, columns_count) = match sk_view_dimensions(&view) {
            Ok(dimensions) => dimensions,
            Err(error) => {
                state.error = Some(error);
                return SKStreamDecision::Stop;
            }
        };
        if rows_count == 0 || columns_count == 0 {
            state.error = Some(SKError::EmptyInput {
                operation: "fit_streaming",
            });
            return SKStreamDecision::Stop;
        }
        match state.accumulator.as_ref() {
            None => {
                if let Some(requested) = self.configuration.number_of_components {
                    if requested > columns_count {
                        state.error = Some(SKError::InvalidHyperparameter {
                            name: "number_of_components",
                            reason: format!(
                                "requested {requested} components, which exceeds feature count ({columns_count})"
                            ),
                        });
                        return SKStreamDecision::Stop;
                    }
                }
                state.accumulator = Some(SKCovarianceAccumulator::new(columns_count));
            }
            Some(existing) if existing.number_of_features() != columns_count => {
                state.error = Some(SKError::shape_mismatch_2d(
                    rows_count,
                    existing.number_of_features(),
                    rows_count,
                    columns_count,
                ));
                return SKStreamDecision::Stop;
            }
            Some(_) => {}
        }
        if let Err(error) = sk_validate_finiteness(view) {
            state.error = Some(error);
            return SKStreamDecision::Stop;
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode(),
            backend: self.configuration.backend,
        };
        let accumulator = state
            .accumulator
            .as_mut()
            .expect("accumulator initialized above");
        let _ = sk_run_operation(attributes, || -> Result<(), SKError> {
            match view {
                SKDataView::Dense(dense) => {
                    sk_update_dense(accumulator, backend, dense, parallelism);
                }
                SKDataView::Sparse(sparse) => {
                    sk_update_sparse(accumulator, sparse);
                }
                _ => unreachable!("sk_view_dimensions rejected unsupported representations"),
            }
            Ok(())
        });
        SKStreamDecision::Continue
    }
}
