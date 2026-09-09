//! Reduction kernels: reduce a 2-D array along one axis into a 1-D result.

use ndarray::{Array, ArrayView2, azip};
use sciencekit_common::SKFloat;

/// Sum a 2-D array along one axis, returning a 1-D result.
///
/// `axis = 0` collapses rows (per-column sums, length = columns);
/// `axis = 1` collapses columns (per-row sums, length = rows). Reads rows as
/// contiguous runs on row-major inputs, accumulating each row into the output
/// with [`azip!`] — layout-aware and free of manual index loops.
pub fn sk_axis_sum<F: SKFloat>(input: &ArrayView2<F>, axis: usize) -> Array<F, ndarray::Ix1> {
    let (rows, cols) = input.dim();
    match axis {
        0 => {
            // Per-column sums: one accumulator per column, traversed row by row.
            let mut output = Array::zeros(cols);
            for row in input.rows() {
                azip!((acc in &mut output, value in row) { *acc = *acc + *value; });
            }
            output
        }
        1 => {
            // Per-row sums: each row is contiguous on row-major inputs.
            let mut output = Array::zeros(rows);
            for (index, row) in input.rows().into_iter().enumerate() {
                output[index] = row.iter().fold(F::zero(), |acc, &value| acc + value);
            }
            output
        }
        _ => panic!("sk_axis_sum axis must be 0 or 1, found {axis}"),
    }
}
