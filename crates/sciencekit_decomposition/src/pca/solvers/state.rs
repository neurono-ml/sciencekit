//! Shared solver output container, orthonormal completion, component sign
//! convention and variance derivation from singular values.

use ndarray::{Array1, Array2, azip};
use sciencekit_common::SKFloat;

/// Raw decomposition state shared by every PCA solver before wrapping into the
/// public fitted model.
#[derive(Debug, Clone)]
pub(crate) struct SKPCADecomposition<F: SKFloat> {
    /// Principal axes in feature space, with shape `k × d`.
    pub(crate) components: Array2<F>,
    /// Unbiased variance explained by each retained component.
    pub(crate) explained_variance: Array1<F>,
    /// Fraction of total centered variance explained by each component.
    pub(crate) explained_variance_ratio: Array1<F>,
    /// Retained non-increasing singular values of the centered design.
    pub(crate) singular_values: Array1<F>,
}

/// Ensures every component row is a unit vector orthogonal to preceding rows
/// (completing zero-norm directions deterministically on rank-deficient inputs)
/// and applies the PCA sign convention so the largest-magnitude coordinate of
/// every row is non-negative (ties broken by smallest feature index).
pub(crate) fn sk_orient_components<F: SKFloat>(components: &mut Array2<F>) {
    let (number_of_components, number_of_features) = components.dim();
    let mut completed: Vec<Array1<F>> = Vec::with_capacity(number_of_components);

    for mut row in components.rows_mut() {
        let squared_norm = row
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value * value);
        if squared_norm <= F::epsilon() * F::epsilon() {
            if let Some(basis_vector) =
                sk_orthogonal_basis_completion(&completed, number_of_features)
            {
                row.assign(&basis_vector);
            }
        }
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
        completed.push(row.to_owned());
    }
}

fn sk_orthogonal_basis_completion<F: SKFloat>(
    existing_rows: &[Array1<F>],
    number_of_features: usize,
) -> Option<Array1<F>> {
    (0..number_of_features).find_map(|basis_index| {
        let mut candidate = Array1::<F>::zeros(number_of_features);
        candidate[basis_index] = F::one();
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
pub(crate) fn sk_compute_variance_ratio<F: SKFloat>(
    explained_variance: &Array1<F>,
    total_variance: F,
) -> Array1<F> {
    if total_variance <= F::zero() {
        Array1::<F>::zeros(explained_variance.len())
    } else {
        explained_variance.mapv(|variance| variance / total_variance)
    }
}
