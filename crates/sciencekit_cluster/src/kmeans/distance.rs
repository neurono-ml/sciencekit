//! Squared-Euclidean assignment kernels (spec `k-means`, decision 2).
//!
//! Assignment uses `‖x − c‖² = ‖x‖² − 2·x·c + ‖c‖²`. The dense path reuses the
//! SIMD-friendly pairwise kernel.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKDataView, SKError, SKFloat};

/// Squared distances from every dense row to every center.
pub(crate) fn sk_dense_squared_distances_to_centers<F: SKFloat>(
    features: &ArrayView2<F>,
    centers: &ArrayView2<F>,
) -> Array2<F> {
    sciencekit_math::sk_squared_euclidean_distance_matrix(features, centers)
}

/// Squared distances from every row of either representation to every dense
/// center. Representation dispatch happens exactly once, before the kernel.
pub(crate) fn sk_squared_distances_to_centers<F: SKFloat>(
    features: SKDataView<'_, F>,
    centers: &ArrayView2<F>,
) -> Result<Array2<F>, SKError> {
    let (number_of_rows, number_of_features) = sk_view_dimensions(&features);
    if number_of_features != centers.ncols() {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_rows, centers.ncols()],
            found: vec![number_of_rows, number_of_features],
        });
    }
    match features {
        SKDataView::Dense(dense) => Ok(sk_dense_squared_distances_to_centers(&dense, centers)),
        SKDataView::Sparse(_) => Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "convert the sparse matrix to dense with to_dense",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

/// Squared distances from every row to a single center.
pub(crate) fn sk_squared_distances_to_center<F: SKFloat>(
    features: SKDataView<'_, F>,
    center: ArrayView1<'_, F>,
) -> Result<Array1<F>, SKError> {
    let centers = center.insert_axis(Axis(0));
    let matrix = sk_squared_distances_to_centers(features, &centers.view())?;
    Ok(matrix.column(0).to_owned())
}

/// `(rows, columns)` of either representation.
pub(crate) fn sk_view_dimensions<F>(features: &SKDataView<'_, F>) -> (usize, usize) {
    match features {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}
