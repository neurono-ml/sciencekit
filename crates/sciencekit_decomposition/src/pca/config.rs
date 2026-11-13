//! PCA solver vocabulary and validated configuration (spec
//! `principal-component-analysis`).

use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode, SKFloat};

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

/// Validated configuration carried by a built [`super::SKPCA`] estimator.
#[derive(Debug, Clone, Copy)]
pub(super) struct SKPCAConfiguration<F: SKFloat> {
    /// Optional component count (`None` retains `min(n_samples, n_features)`).
    pub(super) number_of_components: Option<usize>,
    /// Requested SVD/eigensolver strategy.
    pub(super) svd_solver: SKPCASolver,
    /// Whether transformed projections are divided by `sqrt(explained_variance)`.
    pub(super) whiten: bool,
    /// Solver singular-value tolerance (finite and non-negative).
    pub(super) tolerance: F,
    /// Power iterations for the randomized range finder.
    pub(super) number_of_power_iterations: usize,
    /// Whether dense-only solvers may explicitly materialize a sparse matrix.
    pub(super) allow_densification: bool,
    /// Deterministic seed for the randomized projection.
    pub(super) random_seed: u64,
    /// Math backend recorded in spans and dispatched per operation.
    pub(super) backend: SKBackendKind,
    /// Shared builder state holding the execution intent.
    pub(super) builder_state: SKBuilderState,
}

impl<F: SKFloat> SKPCAConfiguration<F> {
    /// Returns the configured execution mode.
    pub(super) fn execution_mode(&self) -> SKExecutionMode {
        self.builder_state.execution_intent()
    }
}
