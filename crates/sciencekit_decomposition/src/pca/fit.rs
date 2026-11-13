//! Unsupervised fitting for [`SKPCA`] (spec `principal-component-analysis`).

use std::mem::size_of;

use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKFloat, SKOperationAttributes,
    SKOperationKind, SKUnsupervisedFit, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::centering::{sk_centered_dense_or_reject, sk_feature_means, sk_total_centered_variance};
use super::config::SKPCASolver;
use super::core_implementation::SKPCA;
use super::model::SKPCAModel;
use super::solvers::{
    sk_resolve_pca_solver, sk_solve_covariance_eigh, sk_solve_full_svd, sk_solve_randomized_svd,
};
use super::validation::{sk_validate_finiteness, sk_validate_fit_shape, sk_view_dimensions};

impl<F: SKBackendSelection> SKUnsupervisedFit<F> for SKPCA<F> {
    type Model = SKPCAModel<F>;
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
        sk_run_operation(attributes, || sk_fit_pca(self, view, "fit"))
    }
}

/// Shared fit implementation used by both `fit` and `fit_transform`.
pub(super) fn sk_fit_pca<F: SKBackendSelection>(
    estimator: &SKPCA<F>,
    features: SKDataView<'_, F>,
    operation: &'static str,
) -> Result<SKPCAModel<F>, SKError> {
    let configuration = estimator.configuration;
    let (rows_count, columns_count, resolved_components) =
        sk_validate_fit_shape(&features, configuration.number_of_components, operation)?;

    let is_sparse = matches!(features, SKDataView::Sparse(_));
    let resolved_solver = sk_resolve_pca_solver(
        configuration.svd_solver,
        rows_count,
        columns_count,
        resolved_components,
        is_sparse,
        configuration.allow_densification,
    );

    // Reject incompatible sparse representations before scanning any element.
    if is_sparse
        && !configuration.allow_densification
        && matches!(
            resolved_solver,
            SKPCASolver::Full | SKPCASolver::CovarianceEigh
        )
    {
        return Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: super::centering::SPARSE_SOLVER_SUGGESTION,
        });
    }

    sk_validate_finiteness(features)?;
    let plan = sk_resolve_fit_plan(estimator, &features, rows_count, columns_count)?;
    let mean = sk_feature_means(features, plan.parallelism)?;
    let total_variance = sk_total_centered_variance(features, mean.view())?;

    let decomposition = match resolved_solver {
        SKPCASolver::Full => {
            let centered = sk_centered_dense_or_reject(
                features,
                mean.view(),
                configuration.allow_densification,
            )?;
            let backend = sk_math_backend_for_kind::<F>(configuration.backend);
            sk_solve_full_svd(
                backend.as_ref(),
                centered.view(),
                resolved_components,
                total_variance,
            )?
        }
        SKPCASolver::CovarianceEigh => {
            let centered = sk_centered_dense_or_reject(
                features,
                mean.view(),
                configuration.allow_densification,
            )?;
            let backend = sk_math_backend_for_kind::<F>(configuration.backend);
            sk_solve_covariance_eigh(
                backend.as_ref(),
                centered.view(),
                resolved_components,
                total_variance,
                plan.parallelism,
            )?
        }
        SKPCASolver::Randomized => sk_solve_randomized_svd(
            features,
            mean.view(),
            resolved_components,
            configuration.number_of_power_iterations,
            configuration.random_seed,
            configuration.backend,
            total_variance,
        )?,
        SKPCASolver::Auto => unreachable!("sk_resolve_pca_solver never returns Auto"),
    };

    Ok(SKPCAModel::from_decomposition(
        decomposition,
        mean,
        configuration.whiten,
        resolved_solver,
        configuration.backend,
        configuration.builder_state,
    ))
}

fn sk_resolve_fit_plan<F: SKFloat>(
    estimator: &SKPCA<F>,
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
    context.access_pattern = SKAccessPattern::RandomAccess;
    context.grain = SK_AXIS_SUM_GRAIN;
    estimator.resolve_plan(&context)
}
