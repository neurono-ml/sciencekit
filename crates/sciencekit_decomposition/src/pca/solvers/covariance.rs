//! Symmetric covariance eigendecomposition solver for PCA (spec
//! `principal-component-analysis`, decision 5).

use std::cmp::Ordering;

use ndarray::{Array1, Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};
use sciencekit_math::backend::SKMathBackend;

use super::state::{SKPCADecomposition, sk_compute_variance_ratio, sk_orient_components};

/// Fits PCA by forming the unbiased covariance matrix `C = Xcᵀ Xc / (n − 1)`
/// and running a symmetric eigendecomposition on `C`.
pub(in crate::pca) fn sk_solve_covariance_eigh<F: SKFloat>(
    backend: &dyn SKMathBackend<F>,
    centered: ArrayView2<'_, F>,
    number_of_components: usize,
    total_variance: F,
    parallelism: usize,
) -> Result<SKPCADecomposition<F>, SKError> {
    let (number_of_samples, number_of_features) = centered.dim();
    let (covariance, degrees_of_freedom) = if number_of_samples <= 1 {
        (
            Array2::<F>::zeros((number_of_features, number_of_features)),
            F::zero(),
        )
    } else {
        let dof = F::from(number_of_samples - 1).expect("degrees of freedom fit in scalar");
        let matrix = backend.gemm(centered.t(), centered, F::one() / dof, parallelism);
        (matrix, dof)
    };

    let (eigenvalues, eigenvectors) = backend.eigh(covariance.view())?;
    let mut sorted_indices: Vec<usize> = (0..eigenvalues.len()).collect();
    sorted_indices.sort_by(|&left, &right| {
        eigenvalues[right]
            .partial_cmp(&eigenvalues[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    sorted_indices.truncate(number_of_components);

    let mut components = Array2::from_shape_fn(
        (number_of_components, number_of_features),
        |(component, feature)| eigenvectors[(feature, sorted_indices[component])],
    );
    sk_orient_components(&mut components);

    let explained_variance: Array1<F> = sorted_indices
        .iter()
        .map(|&index| eigenvalues[index].max(F::zero()))
        .collect();
    let singular_values =
        explained_variance.mapv(|variance| (variance * degrees_of_freedom).sqrt());
    let explained_variance_ratio = sk_compute_variance_ratio(&explained_variance, total_variance);

    Ok(SKPCADecomposition {
        components,
        explained_variance,
        explained_variance_ratio,
        singular_values,
    })
}
