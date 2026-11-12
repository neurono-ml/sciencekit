//! Randomized singular-value decomposition solver for PCA (spec
//! `principal-component-analysis`, decision 4).

use ndarray::{Array1, Array2, ArrayView1};
use sciencekit_common::{SKBackendKind, SKDataView, SKError};
use sciencekit_math::backend::SKBackendSelection;
use sciencekit_math::decomposition::SKRandomizedSvdBuilder;

use super::state::{SKPCADecomposition, sk_from_singular_values};
use crate::pca::validation::sk_view_dimensions;

/// Fits PCA via the deterministic randomized SVD kernel (`SKRandomizedSvdBuilder`
/// / `sk_randomized_svd`) using implicit centering so sparse CSR inputs never
/// materialize a dense centered matrix.
pub(in crate::pca) fn sk_solve_randomized_svd<F: SKBackendSelection>(
    features: SKDataView<'_, F>,
    mean: ArrayView1<'_, F>,
    number_of_components: usize,
    number_of_power_iterations: usize,
    random_seed: u64,
    backend_kind: SKBackendKind,
    total_variance: F,
) -> Result<SKPCADecomposition<F>, SKError> {
    let (number_of_samples, number_of_features) = sk_view_dimensions(&features)?;
    let randomized = SKRandomizedSvdBuilder::new()
        .number_of_components(number_of_components)
        .number_of_power_iterations(number_of_power_iterations)
        .random_seed(random_seed)
        .backend(backend_kind)
        .compute(features, Some(mean))?;
    let components = Array2::from_shape_fn(
        (number_of_components, number_of_features),
        |(component, feature)| randomized.v[(feature, component)],
    );
    let singular_values: Array1<F> = randomized
        .singular_values
        .into_iter()
        .take(number_of_components)
        .map(|value| value.max(F::zero()))
        .collect();
    Ok(sk_from_singular_values(
        components,
        singular_values,
        number_of_samples,
        total_variance,
    ))
}
