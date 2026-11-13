//! Row-wise access contract shared by every linear SVC design
//! (spec `linear-support-vector-classifier`).

use sciencekit_common::SKFloat;

/// Row-wise access the dual coordinate-descent solver needs from a design.
///
/// Coefficients are stored in the augmented layout: `number_of_columns()`
/// entries followed by one intercept entry when `has_intercept()` is true.
pub(crate) trait SKLinearSvcDesignAccess<F: SKFloat>: Sync {
    /// Number of design rows (`n`).
    fn number_of_rows(&self) -> usize;

    /// Number of stored feature columns (`p`).
    fn number_of_columns(&self) -> usize;

    /// Whether the design carries a constant (intercept) column.
    fn has_intercept(&self) -> bool;

    /// Dot product of the augmented row with the augmented coefficient slice.
    fn row_dot(&self, row_index: usize, coefficients: &[F]) -> F;

    /// Accumulate `scale · augmented_row` into `target`.
    fn add_scaled_row(&self, row_index: usize, scale: F, target: &mut [F]);

    /// Squared Euclidean norm of the augmented row (`Q_ii` diagonal).
    fn row_norm_squared(&self, row_index: usize) -> F;
}
