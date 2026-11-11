//! Projection (`transform`), reconstruction (`inverse_transform`) and combined
//! `fit_transform` for [`SKPCA`] and [`SKPCAModel`] (spec
//! `principal-component-analysis`, decision 7).

use std::mem::size_of;

use ndarray::{Array2, Axis, azip};
use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKFeatureTransformer, SKFloat,
    SKOperationAttributes, SKOperationKind, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
use sciencekit_math::sparse_ops::sk_csr_dense_product;

use super::centering::{sk_center_dense, sk_view_dimensions};
use super::core_implementation::SKPCA;
use super::fit::sk_fit_pca;
use super::model::SKPCAModel;
use super::validation::sk_validate_matrix_input;

const DENSE_TRANSFORM_ROW_CHUNK: usize = 2048;

impl<F: SKBackendSelection> SKPCA<F> {
    /// Fits the estimator on `features` and projects the same input onto the
    /// learned principal components under a single `FitTransform` span.
    pub fn fit_transform<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&view).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::FitTransform,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode(),
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            let model = sk_fit_pca(self, view)?;
            sk_project_features(&model, view)
        })
    }
}

impl<F: SKBackendSelection> SKPCAModel<F> {
    /// Projects `features` onto the retained principal components:
    /// `(X − mean) · componentsᵀ`, optionally scaled to unit variance when
    /// `whiten` is enabled. Sparse CSR inputs are projected without
    /// materializing a dense centered matrix.
    pub fn transform<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&view).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || sk_project_features(self, view))
    }

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

impl<F: SKBackendSelection> SKFeatureTransformer<F> for SKPCAModel<F> {
    type Output = Array2<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        SKPCAModel::transform(self, features)
    }
}

pub(super) fn sk_project_features<F: SKBackendSelection>(
    model: &SKPCAModel<F>,
    features: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (rows_count, columns_count) =
        sk_validate_matrix_input(features, model.number_of_features(), "transform")?;
    let plan = sk_resolve_model_plan(model, &features, rows_count, columns_count)?;
    let components_transposed = model.components.t().to_owned();

    let mut projected = match features {
        SKDataView::Dense(dense) => {
            let backend = sk_math_backend_for_kind::<F>(model.backend);
            let mut output = Array2::<F>::zeros((rows_count, model.number_of_components()));
            for (input_chunk, mut output_chunk) in dense
                .axis_chunks_iter(Axis(0), DENSE_TRANSFORM_ROW_CHUNK)
                .zip(output.axis_chunks_iter_mut(Axis(0), DENSE_TRANSFORM_ROW_CHUNK))
            {
                let centered_chunk = sk_center_dense(input_chunk, model.mean.view());
                let block = backend.gemm(
                    centered_chunk.view(),
                    components_transposed.view(),
                    F::one(),
                    plan.parallelism,
                );
                output_chunk.assign(&block);
            }
            output
        }
        SKDataView::Sparse(sparse) => {
            let mut output = sk_csr_dense_product(&sparse, &components_transposed.view());
            let mean_shift = model.components.dot(&model.mean);
            for mut row in output.rows_mut() {
                azip!((value in &mut row, &shift in &mean_shift) {
                    *value = *value - shift;
                });
            }
            output
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    };

    if model.whiten {
        for mut row in projected.rows_mut() {
            azip!((value in &mut row, &scale in &model.whiten_scale) {
                *value = *value / scale;
            });
        }
    }
    Ok(projected)
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
            backend.gemm(
                dense,
                effective_components.view(),
                F::one(),
                plan.parallelism,
            )
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

fn sk_resolve_model_plan<F: SKFloat>(
    model: &SKPCAModel<F>,
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
    context.dataset_size_bytes = elements.saturating_mul(size_of::<F>() as u64);
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(elements);
    context.access_pattern = SKAccessPattern::Sequential;
    context.grain = SK_AXIS_SUM_GRAIN;
    model.builder_state.resolve_plan(&context)
}
