//! Truncated SVD algorithm vocabulary and validated configuration (spec
//! `truncated-svd`).

use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode, SKFloat};

/// Documented dense/sparse equivalence and rank-`k` reconstruction tolerance
/// (`1e-6` in `f64`) inherited from the randomized range-finder kernel.
pub const SK_TRUNCATED_SVD_PARITY_TOLERANCE: f64 = 1e-6;

/// Solver algorithm for `SKTruncatedSVD`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SKTruncatedSVDAlgorithm {
    /// Seeded randomized range-finder SVD (`sk_randomized_svd`) operating
    /// directly on dense or sparse CSR views without centering.
    #[default]
    Randomized,
    /// Implicitly restarted Lanczos / ARPACK solver (reserved; explicit
    /// non-goal of the initial `truncated-svd` change).
    Arpack,
}

/// Validated configuration carried by a built `SKTruncatedSVD` estimator.
#[derive(Debug, Clone, Copy)]
pub(super) struct SKTruncatedSVDConfiguration<F: SKFloat> {
    /// Number of retained singular components `k` (default `2`).
    pub(super) number_of_components: usize,
    /// Requested low-rank SVD solver (`Randomized` by default).
    pub(super) algorithm: SKTruncatedSVDAlgorithm,
    /// Number of power iterations for the randomized range finder (default `5`).
    pub(super) number_of_iterations: usize,
    /// Solver singular-value tolerance (default `0.0`, finite and non-negative).
    pub(super) tolerance: F,
    /// Deterministic seed for the randomized projection (default `0`).
    pub(super) random_seed: u64,
    /// Math backend recorded in spans and dispatched per operation.
    pub(super) backend: SKBackendKind,
    /// Shared builder state holding the execution intent.
    pub(super) builder_state: SKBuilderState,
}

impl<F: SKFloat> SKTruncatedSVDConfiguration<F> {
    /// Returns the configured execution mode.
    pub(super) fn execution_mode(&self) -> SKExecutionMode {
        self.builder_state.execution_intent()
    }
}
