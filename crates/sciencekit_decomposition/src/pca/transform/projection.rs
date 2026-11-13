//! Forward projection (`transform` and `fit_transform`) onto the principal
//! components (spec `principal-component-analysis`, decision 7).

use ndarray::{Array2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFeatureTransformer, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::sparse_ops::sk_csr_dense_product;

use super::reconstruction::sk_resolve_model_plan;
use crate::pca::centering::sk_center_dense;
use crate::pca::core_implementation::SKPCA;
use crate::pca::fit::sk_fit_pca;
use crate::pca::model::SKPCAModel;
use crate::pca::validation::{sk_validate_matrix_input, sk_view_dimensions};

const DENSE_TRANSFORM_ROW_CHUNK: usize = 256;

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
            let model = sk_fit_pca(self, view, "fit_transform")?;
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
            if plan.parallelism > 1 && rows_count > DENSE_TRANSFORM_ROW_CHUNK {
                dense
                    .axis_chunks_iter(Axis(0), DENSE_TRANSFORM_ROW_CHUNK)
                    .into_par_iter()
                    .zip(
                        output
                            .axis_chunks_iter_mut(Axis(0), DENSE_TRANSFORM_ROW_CHUNK)
                            .into_par_iter(),
                    )
                    .for_each(|(input_chunk, mut output_chunk)| {
                        let centered_chunk = sk_center_dense(input_chunk, model.mean.view());
                        let block = backend.gemm(
                            centered_chunk.view(),
                            components_transposed.view(),
                            F::one(),
                            1,
                        );
                        output_chunk.assign(&block);
                    });
            } else {
                for (input_chunk, mut output_chunk) in dense
                    .axis_chunks_iter(Axis(0), DENSE_TRANSFORM_ROW_CHUNK)
                    .zip(output.axis_chunks_iter_mut(Axis(0), DENSE_TRANSFORM_ROW_CHUNK))
                {
                    let centered_chunk = sk_center_dense(input_chunk, model.mean.view());
                    let block = backend.gemm(
                        centered_chunk.view(),
                        components_transposed.view(),
                        F::one(),
                        1,
                    );
                    output_chunk.assign(&block);
                }
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
