//! Batch-chunked fitting for [`SKIncrementalPCA`] (spec `incremental-pca`,
//! decision 4).

use std::mem::size_of;

use ndarray::{Axis, Slice};
use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKFloat, SKOperationAttributes,
    SKOperationKind, SKUnsupervisedFit, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::accumulator::SKCovarianceAccumulator;
use super::core_implementation::SKIncrementalPCA;
use super::finalize::sk_finalize_accumulator;
use super::merge::sk_update_dense;
use super::model::SKIncrementalPCAModel;
use super::sparse::sk_update_sparse;
use super::validation::{sk_chunk_ranges, sk_validate_fit_chunking};
use crate::pca::sk_view_dimensions;

impl<F: SKBackendSelection> SKUnsupervisedFit<F> for SKIncrementalPCA<F> {
    type Model = SKIncrementalPCAModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&view).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode(),
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_incremental_pca(self, view, "fit"))
    }
}

pub(super) fn sk_fit_incremental_pca<F: SKBackendSelection>(
    estimator: &SKIncrementalPCA<F>,
    features: SKDataView<'_, F>,
    operation: &'static str,
) -> Result<SKIncrementalPCAModel<F>, SKError> {
    let configuration = estimator.configuration;
    let (rows_count, columns_count, resolved_components) = sk_validate_fit_chunking(
        features,
        configuration.number_of_components,
        configuration.batch_size,
        operation,
    )?;
    let plan = sk_resolve_batch_plan(
        &configuration.builder_state,
        &features,
        rows_count,
        columns_count,
    )?;
    let backend = sk_math_backend_for_kind::<F>(configuration.backend);
    let ranges = sk_chunk_ranges(rows_count, configuration.batch_size, resolved_components);

    let mut accumulator = SKCovarianceAccumulator::<F>::new(columns_count);
    match features {
        SKDataView::Dense(dense) => {
            ranges.into_iter().for_each(|(start, end)| {
                let chunk = dense.slice_axis(Axis(0), Slice::from(start..end));
                sk_update_dense(&mut accumulator, backend.as_ref(), chunk, plan.parallelism);
            });
        }
        SKDataView::Sparse(sparse) => {
            ranges.into_iter().for_each(|(start, end)| {
                let chunk = sparse.slice_outer(start..end);
                sk_update_sparse(&mut accumulator, chunk);
            });
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    }

    let decomposition =
        sk_finalize_accumulator(&accumulator, backend.as_ref(), resolved_components)?;
    Ok(SKIncrementalPCAModel::from_accumulator_and_decomposition(
        accumulator,
        decomposition,
        1,
        configuration,
    ))
}

pub(super) fn sk_resolve_batch_plan<F: SKFloat>(
    builder_state: &sciencekit_common::SKBuilderState,
    features: &SKDataView<'_, F>,
    rows_count: usize,
    columns_count: usize,
) -> Result<SKExecutionPlan, SKError> {
    let elements = match features {
        SKDataView::Dense(_) => (rows_count as u64).saturating_mul(columns_count as u64),
        SKDataView::Sparse(sparse) => sparse.nnz() as u64,
        _ => 0,
    };
    let mut context = SKExecutionContext::real();
    context.cpu_cores = context.cpu_cores.min(rayon::current_num_threads().max(1));
    context.dataset_size_bytes = elements.saturating_mul(size_of::<F>() as u64);
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(elements);
    context.access_pattern = SKAccessPattern::Sequential;
    context.grain = SK_AXIS_SUM_GRAIN;
    builder_state.resolve_plan(&context)
}
