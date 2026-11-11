//! Deterministic fixtures for PCA acceptance tests.

use ndarray::Array2;
use sprs::{CsMat, TriMat};

/// Generates a deterministic low-rank `rows × columns` dense matrix with `rank`
/// orthogonal-like signal modes plus a small deterministic residual and column
/// offsets so centering is non-trivial.
pub(super) fn planted_rank_matrix(
    rows: usize,
    columns: usize,
    rank: usize,
    noise_amplitude: f64,
) -> Array2<f64> {
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        let row_scalar = row as f64;
        let column_scalar = column as f64;
        let mut signal = 3.0 * (column_scalar + 1.0);
        for mode in 0..rank {
            let weight = 8.0 / ((mode + 1) * (mode + 1)) as f64;
            let sample_wave = (0.07 * (mode as f64 + 1.0) * row_scalar + 0.3 * mode as f64).sin();
            let feature_wave =
                (0.19 * (mode as f64 + 1.0) * column_scalar + 0.5 * mode as f64).cos();
            signal += weight * sample_wave * feature_wave;
        }
        let deterministic_noise =
            noise_amplitude * (((row * 31 + column * 17 + 5) % 29) as f64 - 14.0) / 14.0;
        signal + deterministic_noise
    })
}

/// Generates a sparse CSR matrix with dominant feature modes and a sparse tail
/// column per row without allocating a dense `rows × columns` buffer.
pub(super) fn planted_wide_csr(rows: usize, columns: usize) -> CsMat<f64> {
    let mut triplet = TriMat::with_capacity((rows, columns), rows * 4);
    for row in 0..rows {
        let mode_one = ((row % 11) as f64 - 5.0) * 2.0;
        let mode_two = ((row % 7) as f64 - 3.0) * 0.8;
        let tail_column = 4 + (row * 13) % (columns - 4);
        triplet.add_triplet(row, 0, mode_one * 3.0 + 1.0);
        triplet.add_triplet(row, 1, -mode_one * 2.0 + 0.5);
        triplet.add_triplet(row, 2 + (row % 2), mode_two);
        triplet.add_triplet(row, tail_column, 0.02 * ((row % 5) as f64 + 1.0));
    }
    triplet.to_csr()
}
