//! Full singular-value decomposition solver for PCA (spec
//! `principal-component-analysis`, decision 3).

use ndarray::{Array1, Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};
use sciencekit_math::backend::SKMathBackend;

use super::state::{SKPCADecomposition, sk_from_singular_values};

/// Fits PCA via the backend's full singular-value decomposition of the centered
/// dense design matrix `Xc` (`n × d`).
pub(in crate::pca) fn sk_solve_full_svd<F: SKFloat>(
    backend: &dyn SKMathBackend<F>,
    centered: ArrayView2<'_, F>,
    number_of_components: usize,
    total_variance: F,
) -> Result<SKPCADecomposition<F>, SKError> {
    let (number_of_samples, number_of_features) = centered.dim();
    let decomposition = backend.svd(centered)?;
    let components = Array2::from_shape_fn(
        (number_of_components, number_of_features),
        |(component, feature)| decomposition.v[(feature, component)],
    );
    let singular_values: Array1<F> = decomposition
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
