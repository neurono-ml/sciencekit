//! Convenience constructors for the central error taxonomy.

use super::taxonomy::SKError;

/// Convenience constructors for the most common variants.
impl SKError {
    /// Build a shape-mismatch error for a 2-D input (rows, columns).
    pub fn shape_mismatch_2d(
        expected_rows: usize,
        expected_cols: usize,
        found_rows: usize,
        found_cols: usize,
    ) -> Self {
        SKError::ShapeMismatch {
            expected: vec![expected_rows, expected_cols],
            found: vec![found_rows, found_cols],
        }
    }

    /// Build a not-converged error reporting the effort spent.
    pub fn not_converged(iterations: usize) -> Self {
        SKError::NotConverged { iterations }
    }
}
