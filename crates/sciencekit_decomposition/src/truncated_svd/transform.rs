//! Forward projection (`transform`) and reconstruction (`inverse_transform`)
//! for [`SKTruncatedSVDModel`] (`spec` `truncated-svd`, decision 2).

use ndarray::Array2;
use sciencekit_common::{
    SKDataView, SKError, SKFeatureTransformer, SKOperationAttributes, SKOperationKind,
    execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::backend::SKBackendSelection;

use super::core_implementation::SKTruncatedSVD;
use super::fit::{sk_fit_with_projection, sk_resolve_plan};
use super::model::SKTruncatedSVDModel;
use super::product::sk_multiply_view_by_dense;
use super::validation::{SKTruncatedInput, sk_validate_matrix_input};

impl<F: SKBackendSelection> SKTruncatedSVD<F> {
    /// Fits the estimator on `features` and projects the same input onto the
    /// retained right singular vectors under a single `FitTransform` span,
    /// without centering or densifying sparse CSR inputs.
    pub fn fit_transform<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let prepared = SKTruncatedInput::try_from_view(view);
        let (rows_count, columns_count) = prepared
            .as_ref()
            .map(SKTruncatedInput::dimensions)
            .unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::FitTransform,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode(),
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            sk_fit_with_projection(&self.configuration, prepared?, "fit_transform")
                .map(|(_, projected)| projected)
        })
    }
}

impl<F: SKBackendSelection> SKTruncatedSVDModel<F> {
    /// Projects `features` onto the retained right singular vectors
    /// (`X · componentsᵀ`) without centering or densifying sparse CSR inputs.
    pub fn transform<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let prepared = SKTruncatedInput::try_from_view(view);
        let (rows_count, columns_count) = prepared
            .as_ref()
            .map(SKTruncatedInput::dimensions)
            .unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || sk_project_features(self, prepared?))
    }

    /// Reconstructs feature-space approximations from projected coordinates
    /// (`Z · components`) without adding any centering vector or densifying
    /// sparse CSR inputs.
    pub fn inverse_transform<'a, X>(&self, projected: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = projected.try_into()?;
        let prepared = SKTruncatedInput::try_from_view(view);
        let (rows_count, columns_count) = prepared
            .as_ref()
            .map(SKTruncatedInput::dimensions)
            .unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || sk_reconstruct_features(self, prepared?))
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
    input: SKTruncatedInput<'_, F>,
) -> Result<Array2<F>, SKError> {
    sk_validate_matrix_input(input, model.number_of_features(), "transform")?;
    let plan = sk_resolve_plan::<F>(
        &model.builder_state,
        input.stored_elements(),
        SKAccessPattern::Sequential,
    )?;
    let components_transposed = model.components.t().to_owned();
    Ok(sk_multiply_view_by_dense(
        input,
        components_transposed.view(),
        model.backend,
        plan.parallelism,
    ))
}

fn sk_reconstruct_features<F: SKBackendSelection>(
    model: &SKTruncatedSVDModel<F>,
    input: SKTruncatedInput<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (rows_count, columns_count) =
        sk_validate_matrix_input(input, model.number_of_components(), "inverse_transform")?;
    let effective_elements = match input {
        SKTruncatedInput::Dense(_) => {
            let effective_columns = columns_count.max(model.number_of_features());
            (rows_count as u64).saturating_mul(effective_columns as u64)
        }
        SKTruncatedInput::SparseCsr(_) => input.stored_elements(),
    };
    let plan = sk_resolve_plan::<F>(
        &model.builder_state,
        effective_elements,
        SKAccessPattern::Sequential,
    )?;
    Ok(sk_multiply_view_by_dense(
        input,
        model.components.view(),
        model.backend,
        plan.parallelism,
    ))
}
