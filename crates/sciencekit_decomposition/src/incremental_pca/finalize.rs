//! Symmetric covariance eigendecomposition finalization for incremental PCA
//! (spec `incremental-pca`, decision 3).

use std::cmp::Ordering;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKError, SKFloat};
use sciencekit_math::backend::SKMathBackend;

use super::accumulator::SKCovarianceAccumulator;
use crate::pca::{SKPCADecomposition, sk_compute_variance_ratio, sk_orient_components};

/// Derives the sample covariance `scatter / (n − 1)`, runs the symmetric
/// eigensolver, sorts eigenvalues non-increasingly and extracts the leading
/// `number_of_components` principal axes oriented with the PCA sign rule.
pub(super) fn sk_finalize_accumulator<F: SKFloat>(
    accumulator: &SKCovarianceAccumulator<F>,
    backend: &dyn SKMathBackend<F>,
    number_of_components: usize,
) -> Result<SKPCADecomposition<F>, SKError> {
    let number_of_samples = accumulator.samples_seen();
    let number_of_features = accumulator.number_of_features();

    let (covariance, degrees_of_freedom, total_variance) = if number_of_samples <= 1 {
        (
            Array2::<F>::zeros((number_of_features, number_of_features)),
            F::zero(),
            F::zero(),
        )
    } else {
        let degrees_of_freedom =
            F::from(number_of_samples - 1).expect("degrees of freedom fit in scalar");
        let covariance = accumulator
            .scatter()
            .mapv(|value| value / degrees_of_freedom);
        let total_variance = covariance
            .diag()
            .iter()
            .fold(F::zero(), |sum, &value| sum + value.max(F::zero()));
        (covariance, degrees_of_freedom, total_variance)
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
