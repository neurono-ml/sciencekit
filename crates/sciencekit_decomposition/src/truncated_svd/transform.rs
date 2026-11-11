//! Forward projection (`transform`) and reconstruction (`inverse_transform`)
//! for [`SKTruncatedSVDModel`] (`spec` `truncated-svd`, decision 2).

use ndarray::Array2;
use sciencekit_common::{
    SKDataView, SKError, SKFeatureTransformer, SKOperationAttributes, SKOperationKind,
    execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::backend::SKBackendSelection;

use super::fit::sk_resolve_plan;
use super::model::SKTruncatedSVDModel;
use super::product::sk_multiply_view_by_dense;
use super::validation::{sk_validate_matrix_input, sk_view_dimensions};

impl<F: SKBackendSelection> SKTruncatedSVDModel<F> {
    /// Projects `features` onto the retained right singular vectors
    /// (`X · componentsᵀ`) without centering or densifying sparse CSR inputs.
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

    /// Reconstructs feature-space approximations from projected coordinates
    /// (`Z · components`) without adding any centering vector or densifying
    /// sparse CSR inputs.
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

impl<F: SKBackendSelection> SKFeatureTransformer<F> for SKTruncatedSVDModel<F> {
    type Output = Array2<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        SKTruncatedSVDModel::transform(self, features)
    }
}

fn sk_project_features<F: SKBackendSelection>(
    model: &SKTruncatedSVDModel<F>,
    features: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (rows_count, columns_count) =
        sk_validate_matrix_input(features, model.number_of_features(), "transform")?;
    let plan = sk_resolve_plan(
        &model.builder_state,
        &features,
        rows_count,
        columns_count,
        SKAccessPattern::Sequential,
    )?;
    let components_transposed = model.components.t().to_owned();
    sk_multiply_view_by_dense(
        features,
        components_transposed.view(),
        model.backend,
        plan.parallelism,
    )
}

fn sk_reconstruct_features<F: SKBackendSelection>(
    model: &SKTruncatedSVDModel<F>,
    projected: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (rows_count, columns_count) =
        sk_validate_matrix_input(projected, model.number_of_components(), "inverse_transform")?;
    let effective_columns = columns_count.max(model.number_of_features());
    let plan = sk_resolve_plan(
        &model.builder_state,
        &projected,
        rows_count,
        effective_columns,
        SKAccessPattern::Sequential,
    )?;
    sk_multiply_view_by_dense(
        projected,
        model.components.view(),
        model.backend,
        plan.parallelism,
    )
}
