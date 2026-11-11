//! PCA solver vocabulary (spec `principal-component-analysis`).

/// Solver portfolio for [`super::SKPCA`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKPCASolver {
    /// Deterministic heuristic: `Randomized` for sparse inputs without explicit
    /// densification, `Full` when at least 80% of `min(n_samples, n_features)`
    /// components are requested or on small dense designs (`max(n, d) <= 500`
    /// outside the tall-skinny covariance regime), `CovarianceEigh` when
    /// `n_samples > 10 * n_features` and `n_features <= 1000`, and `Randomized`
    /// on larger truncated designs.
    #[default]
    Auto,
    /// Exact singular-value decomposition of the centered design `Xc`.
    Full,
    /// Seeded randomized range-finder SVD (`sk_randomized_svd`), supporting
    /// implicit centering on both dense and sparse CSR views.
    Randomized,
    /// Symmetric eigendecomposition of the sample covariance `Xcᵀ Xc / (n − 1)`.
    CovarianceEigh,
}
