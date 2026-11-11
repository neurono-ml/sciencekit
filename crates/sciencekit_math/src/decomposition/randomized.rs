//! The public surface of the randomized singular-value decomposition.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKDataView, SKError};

use super::kernel::sk_randomized_svd_kernel;
use super::result::SKRandomizedSvdResult;
use crate::backend::{SKBackendSelection, sk_math_backend_for_kind};

/// The builder of the randomized singular-value decomposition.
///
/// The builder carries the configuration of the decomposition: the requested
/// rank, the number of power iterations, the oversampling, the seed of the
/// random projection and the backend kind used for the dense algebra. The
/// backend kind is a runtime value, dispatched per operation.
#[derive(Debug, Clone, Copy)]
pub struct SKRandomizedSvdBuilder {
    number_of_components: usize,
    number_of_power_iterations: usize,
    oversampling: usize,
    random_seed: u64,
    backend: SKBackendKind,
}

impl Default for SKRandomizedSvdBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKRandomizedSvdBuilder {
    /// Creates a builder with the default configuration: one component, one
    /// power iteration, an oversampling of 10, the seed `0` and the `Faer`
    /// backend.
    pub fn new() -> Self {
        Self {
            number_of_components: 1,
            number_of_power_iterations: 1,
            oversampling: 10,
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// Sets the number of retained components (the rank of the truncation).
    pub fn number_of_components(mut self, number_of_components: usize) -> Self {
        self.number_of_components = number_of_components;
        self
    }

    /// Sets the number of power iterations applied to the projected basis.
    pub fn number_of_power_iterations(mut self, number_of_power_iterations: usize) -> Self {
        self.number_of_power_iterations = number_of_power_iterations;
        self
    }

    /// Sets the oversampling: the number of extra random directions kept
    /// during the range search.
    pub fn oversampling(mut self, oversampling: usize) -> Self {
        self.oversampling = oversampling;
        self
    }

    /// Sets the seed of the random projection.
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// Sets the backend kind used for the dense algebra of the kernel.
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Computes the truncated singular-value decomposition of `matrix`.
    ///
    /// The optional `centering` vector holds one mean per feature; when it is
    /// supplied the decomposition is computed on the implicitly centered
    /// matrix, without materializing a centered copy.
    pub fn compute<F: SKBackendSelection>(
        self,
        matrix: SKDataView<'_, F>,
        centering: Option<ArrayView1<'_, F>>,
    ) -> Result<SKRandomizedSvdResult<F>, SKError> {
        let backend = sk_math_backend_for_kind::<F>(self.backend);
        sk_randomized_svd_kernel(
            backend.as_ref(),
            matrix,
            self.number_of_components,
            self.number_of_power_iterations,
            self.oversampling,
            self.random_seed,
            centering,
        )
    }
}

/// Computes a deterministic truncated singular-value decomposition of `matrix`.
///
/// The kernel is a randomized range finder: a seeded Gaussian test basis is
/// projected onto the matrix, the projection is orthonormalized with a QR
/// decomposition and refined by `number_of_power_iterations` power
/// iterations. The small projected matrix is then decomposed with a small
/// dense SVD of its `(k + oversampling) × (k + oversampling)` Gram matrix and
/// the result is truncated to `number_of_components`. The backend's full QR
/// and SVD are never used, so the intermediates stay thin: at most
/// `(k + oversampling) · max(rows, columns)` values are materialized.
///
/// The same `random_seed` always yields the same basis, therefore the same
/// factorization, independently of the thread count. An optional `centering`
/// vector centers the matrix implicitly, without materializing a centered
/// copy, for dense and sparse inputs alike.
///
/// **Accuracy.** The returned singular values approximate the leading
/// singular values of the (centered) matrix: for a matrix whose spectrum
/// decays, the relative error of the `k`-th singular value stays below
/// `1e-6` once the range search is given enough power iterations and
/// oversampling. The reconstruction error of the truncation never exceeds the
/// error of the optimal rank-`k` truncation by more than `1e-6` (relative).
/// The dense and the sparse paths agree within the same tolerance.
///
/// # Errors
///
/// Returns [`SKError::EmptyInput`] for an empty matrix and
/// [`SKError::InvalidHyperparameter`] for a zero or excessive component
/// count, for zero power iterations and for a zero oversampling.
pub fn sk_randomized_svd<F: SKBackendSelection>(
    matrix: SKDataView<'_, F>,
    number_of_components: usize,
    number_of_power_iterations: usize,
    oversampling: usize,
    random_seed: u64,
    centering: Option<ArrayView1<'_, F>>,
) -> Result<SKRandomizedSvdResult<F>, SKError> {
    SKRandomizedSvdBuilder::new()
        .number_of_components(number_of_components)
        .number_of_power_iterations(number_of_power_iterations)
        .oversampling(oversampling)
        .random_seed(random_seed)
        .compute(matrix, centering)
}
