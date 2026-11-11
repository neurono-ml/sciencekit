//! Unsupervised randomized-SVD fitting for `SKTruncatedSVD` (spec
//! `truncated-svd`).

use std::mem::size_of;

use ndarray::{Array1, Array2, azip};
use sciencekit_common::{
    SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKFloat,
    execution::SKAccessPattern,
};
use sciencekit_math::backend::SKBackendSelection;
use sciencekit_math::decomposition::SKRandomizedSvdBuilder;
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::config::SKTruncatedSVDConfiguration;
use super::model::SKTruncatedSVDModel;
use super::validation::sk_validate_fit_input;
use super::variance::{sk_multiply_view_by_dense, sk_variance_and_ratio};

/// Fits an uncentered truncated SVD model from `configuration` and returns both
/// the fitted model and the projected training rows (`X · componentsᵀ`).
#[allow(dead_code)]
pub(super) fn sk_fit_with_projection<F: SKBackendSelection>(
    configuration: &SKTruncatedSVDConfiguration<F>,
    features: SKDataView<'_, F>,
    operation: &'static str,
) -> Result<(SKTruncatedSVDModel<F>, Array2<F>), SKError> {
    let (rows_count, columns_count) = sk_validate_fit_input(features, configuration, operation)?;
    let plan = sk_resolve_plan(
        &configuration.builder_state,
        &features,
        rows_count,
        columns_count,
        SKAccessPattern::RandomAccess,
    )?;

    let randomized = SKRandomizedSvdBuilder::new()
        .number_of_components(configuration.number_of_components)
        .number_of_power_iterations(configuration.number_of_iterations)
        .random_seed(configuration.random_seed)
        .backend(configuration.backend)
        .compute(features, None)?;

    let mut components = Array2::from_shape_fn(
        (configuration.number_of_components, columns_count),
        |(component, feature)| randomized.v[(feature, component)],
    );
    sk_complete_zero_norm_components(&mut components);

    let singular_values: Array1<F> = randomized
        .singular_values
        .into_iter()
        .take(configuration.number_of_components)
        .map(|value| value.max(F::zero()))
        .collect();

    let components_transposed = components.t().to_owned();
    let projected = sk_multiply_view_by_dense(
        features,
        components_transposed.view(),
        configuration.backend,
        plan.parallelism,
    )?;
    let (explained_variance, explained_variance_ratio) =
        sk_variance_and_ratio(projected.view(), features, plan.parallelism)?;

    let model = SKTruncatedSVDModel::from_parts(
        components,
        singular_values,
        explained_variance,
        explained_variance_ratio,
        configuration.algorithm,
        configuration.backend,
        configuration.builder_state,
    );
    Ok((model, projected))
}

/// Resolves an execution plan from `builder_state` without materializing sparse
/// matrices.
pub(super) fn sk_resolve_plan<F: SKFloat>(
    builder_state: &SKBuilderState,
    features: &SKDataView<'_, F>,
    rows_count: usize,
    columns_count: usize,
    access_pattern: SKAccessPattern,
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
    context.access_pattern = access_pattern;
    context.grain = SK_AXIS_SUM_GRAIN;
    builder_state.resolve_plan(&context)
}

/// Orthonormalizes the rows of `components` via modified Gram-Schmidt while
/// preserving the kernel-assigned sign of every active singular vector, and
/// completes any rank-deficient row (whose norm collapses once projections onto
/// preceding singular vectors are removed) with a deterministic canonical unit
/// vector.
fn sk_complete_zero_norm_components<F: SKFloat>(components: &mut Array2<F>) {
    let (number_of_components, number_of_features) = components.dim();
    let mut completed: Vec<Array1<F>> = Vec::with_capacity(number_of_components);
    let collapsed_threshold = F::from(1e-6).expect("threshold fits in scalar");

    for mut row in components.rows_mut() {
        for previous in &completed {
            let dot = row.dot(previous);
            azip!((entry in &mut row, &basis_value in previous) {
                *entry = *entry - dot * basis_value;
            });
        }
        let norm = row
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value * value)
            .sqrt();
        if norm > collapsed_threshold {
            azip!((entry in &mut row) {
                *entry = *entry / norm;
            });
        } else if let Some(basis_vector) =
            sk_orthogonal_unit_direction(&completed, number_of_features)
        {
            row.assign(&basis_vector);
        }
        completed.push(row.to_owned());
    }
}

fn sk_orthogonal_unit_direction<F: SKFloat>(
    existing_rows: &[Array1<F>],
    number_of_features: usize,
) -> Option<Array1<F>> {
    (0..number_of_features).find_map(|basis_index| {
        let mut candidate = Array1::from_shape_fn(number_of_features, |column| {
            if column == basis_index {
                F::one()
            } else {
                F::zero()
            }
        });
        for previous in existing_rows {
            let dot = candidate.dot(previous);
            azip!((entry in &mut candidate, &basis_value in previous) {
                *entry = *entry - dot * basis_value;
            });
        }
        let norm = candidate
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value * value)
            .sqrt();
        (norm > F::epsilon()).then(|| candidate.mapv(|value| value / norm))
    })
}
