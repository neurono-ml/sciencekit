//! Inverse projection (`inverse_transform`) and execution-plan resolution for
//! [`SKPCAModel`] (spec `principal-component-analysis`, decision 7).

use std::mem::size_of;

use ndarray::{Array2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKFloat, SKOperationAttributes,
    SKOperationKind, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
use sciencekit_math::sparse_ops::sk_csr_dense_product;

use crate::pca::centering::sk_view_dimensions;
use crate::pca::model::SKPCAModel;
use crate::pca::validation::sk_validate_matrix_input;

const DENSE_RECONSTRUCT_ROW_CHUNK: usize = 256;

impl<F: SKBackendSelection> SKPCAModel<F> {
    /// Maps projected component coordinates back into the original feature
    /// space (`Z · components + mean`, reversing whitening when enabled).
    pub fn inverse_transform<'a, X>(&self, projected: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = projected.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&view).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || sk_reconstruct_features(self, view))
    }
}

fn sk_reconstruct_features<F: SKBackendSelection>(
    model: &SKPCAModel<F>,
    projected: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (rows_count, columns_count) =
        sk_validate_matrix_input(projected, model.number_of_components(), "inverse_transform")?;
    let plan = sk_resolve_model_plan(model, &projected, rows_count, columns_count)?;
    let effective_components = if model.whiten {
        let mut scaled = model.components.clone();
        for (mut row, &scale) in scaled.rows_mut().into_iter().zip(model.whiten_scale.iter()) {
            azip!((value in &mut row) {
                *value = *value * scale;
            });
        }
        scaled
    } else {
        model.components.clone()
    };

    let mut reconstructed = match projected {
        SKDataView::Dense(dense) => {
            let backend = sk_math_backend_for_kind::<F>(model.backend);
            let mut output = Array2::<F>::zeros((rows_count, model.number_of_features()));
            if plan.parallelism > 1 && rows_count > DENSE_RECONSTRUCT_ROW_CHUNK {
                dense
                    .axis_chunks_iter(Axis(0), DENSE_RECONSTRUCT_ROW_CHUNK)
                    .into_par_iter()
                    .zip(
                        output
                            .axis_chunks_iter_mut(Axis(0), DENSE_RECONSTRUCT_ROW_CHUNK)
                            .into_par_iter(),
                    )
                    .for_each(|(input_chunk, mut output_chunk)| {
                        let block =
                            backend.gemm(input_chunk, effective_components.view(), F::one(), 1);
                        output_chunk.assign(&block);
                    });
            } else {
                for (input_chunk, mut output_chunk) in dense
                    .axis_chunks_iter(Axis(0), DENSE_RECONSTRUCT_ROW_CHUNK)
                    .zip(output.axis_chunks_iter_mut(Axis(0), DENSE_RECONSTRUCT_ROW_CHUNK))
                {
                    let block = backend.gemm(input_chunk, effective_components.view(), F::one(), 1);
                    output_chunk.assign(&block);
                }
            }
            output
        }
        SKDataView::Sparse(sparse) => sk_csr_dense_product(&sparse, &effective_components.view()),
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: projected.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    };

    for mut row in reconstructed.rows_mut() {
        azip!((value in &mut row, &column_mean in &model.mean) {
            *value = *value + column_mean;
        });
    }
    Ok(reconstructed)
}

pub(super) fn sk_resolve_model_plan<F: SKFloat>(
    model: &SKPCAModel<F>,
    features: &SKDataView<'_, F>,
    rows_count: usize,
    columns_count: usize,
) -> Result<SKExecutionPlan, SKError> {
    let effective_columns = columns_count.max(model.number_of_features());
    let elements = match features {
        SKDataView::Dense(_) => (rows_count as u64).saturating_mul(effective_columns as u64),
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
    model.builder_state.resolve_plan(&context)
}
