//! Deterministic `Auto` solver resolution (spec `principal-component-analysis`,
//! decision 6).

use crate::pca::config::SKPCASolver;

/// Resolves a configured [`SKPCASolver`] against the observed input dimensions,
/// retained component count, input sparsity and densification flag.
///
/// The returned solver is always concrete (`Full`, `Randomized` or
/// `CovarianceEigh`), never `Auto`.
pub(in crate::pca) fn sk_resolve_pca_solver(
    requested: SKPCASolver,
    number_of_samples: usize,
    number_of_features: usize,
    number_of_components: usize,
    is_sparse: bool,
    allow_densification: bool,
) -> SKPCASolver {
    match requested {
        SKPCASolver::Auto => {
            if is_sparse && !allow_densification {
                return SKPCASolver::Randomized;
            }
            let minimum_dimension = number_of_samples.min(number_of_features);
            let maximum_dimension = number_of_samples.max(number_of_features);
            if number_of_components.saturating_mul(5) >= minimum_dimension.saturating_mul(4) {
                SKPCASolver::Full
            } else if number_of_samples > number_of_features.saturating_mul(10)
                && number_of_features <= 1000
            {
                SKPCASolver::CovarianceEigh
            } else if maximum_dimension <= 500 {
                SKPCASolver::Full
            } else {
                SKPCASolver::Randomized
            }
        }
        concrete => concrete,
    }
}
