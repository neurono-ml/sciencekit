//! Shared solver output container, component sign convention and variance
//! derivation from singular values.

use ndarray::{Array1, Array2, azip};
use sciencekit_common::SKFloat;

/// Raw decomposition state shared by every PCA solver before wrapping into the
/// public fitted model.
#[derive(Debug, Clone)]
pub(in crate::pca) struct SKPCADecomposition<F: SKFloat> {
    /// Principal axes in feature space, with shape `k × d`.
    pub(in crate::pca) components: Array2<F>,
    /// Unbiased variance explained by each retained component.
    pub(in crate::pca) explained_variance: Array1<F>,
    /// Fraction of total centered variance explained by each component.
    pub(in crate::pca) explained_variance_ratio: Array1<F>,
    /// Retained non-increasing singular values of the centered design.
    pub(in crate::pca) singular_values: Array1<F>,
}

/// Applies the deterministic PCA sign convention: the largest-magnitude
/// coordinate of every principal component row is made non-negative, breaking
/// exact ties by the smallest feature index.
pub(super) fn sk_orient_components<F: SKFloat>(components: &mut Array2<F>) {
    for mut row in components.rows_mut() {
        let mut pivot_column = 0usize;
        let mut pivot_magnitude = F::zero();
        for (column, &value) in row.iter().enumerate() {
            let magnitude = value.abs();
            if column == 0 || magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_column = column;
            }
        }
        if row[pivot_column] < F::zero() {
            azip!((value in &mut row) {
                *value = -*value;
            });
        }
    }
}

/// Derives explained variance and explained-variance ratio from singular values
/// and the total centered variance.
pub(super) fn sk_from_singular_values<F: SKFloat>(
    mut components: Array2<F>,
    singular_values: Array1<F>,
    number_of_samples: usize,
    total_variance: F,
) -> SKPCADecomposition<F> {
    sk_orient_components(&mut components);
    let explained_variance = if number_of_samples <= 1 {
        Array1::<F>::zeros(singular_values.len())
    } else {
        let degrees_of_freedom =
            F::from(number_of_samples - 1).expect("degrees of freedom fit in scalar");
        singular_values.mapv(|singular| (singular * singular) / degrees_of_freedom)
    };
    let explained_variance_ratio = sk_compute_variance_ratio(&explained_variance, total_variance);
    SKPCADecomposition {
        components,
        explained_variance,
        explained_variance_ratio,
        singular_values,
    }
}

/// Divides each component's variance by `total_variance`, returning zeros when
/// the input design is constant (`total_variance <= 0`).
pub(super) fn sk_compute_variance_ratio<F: SKFloat>(
    explained_variance: &Array1<F>,
    total_variance: F,
) -> Array1<F> {
    if total_variance <= F::zero() {
        Array1::<F>::zeros(explained_variance.len())
    } else {
        explained_variance.mapv(|variance| variance / total_variance)
    }
}
