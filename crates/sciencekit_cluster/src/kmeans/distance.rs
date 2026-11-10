//! Squared-Euclidean assignment kernels (spec `k-means`, decision 2).
//!
//! Assignment uses `‖x − c‖² = ‖x‖² − 2·x·c + ‖c‖²`. The dense path reuses the
//! SIMD-friendly pairwise kernel; the sparse path walks only the stored entries
//! of each CSR row, so a sparse design is never densified.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsMatView;

/// Squared distances from every dense row to every center.
pub(crate) fn sk_dense_squared_distances_to_centers<F: SKFloat>(
    features: &ArrayView2<F>,
    centers: &ArrayView2<F>,
) -> Array2<F> {
    sciencekit_math::sk_squared_euclidean_distance_matrix(features, centers)
}

/// Squared distances from every CSR row to every dense center, by norm
/// expansion over the stored entries.
pub(crate) fn sk_sparse_squared_distances_to_centers<F: SKFloat>(
    features: &CsMatView<F>,
    centers: &ArrayView2<F>,
) -> Array2<F> {
    let number_of_rows = features.rows();
    let number_of_centers = centers.nrows();
    let center_norms: Vec<F> = centers
        .rows()
        .into_iter()
        .map(|center| {
            center.iter().fold(F::zero(), |accumulator, value| {
                accumulator + *value * *value
            })
        })
        .collect();
    let mut distances = Array2::zeros((number_of_rows, number_of_centers));
    for (row_index, row) in features.outer_iterator().enumerate() {
        let row_norm = row.data().iter().fold(F::zero(), |accumulator, value| {
            accumulator + *value * *value
        });
        for (center_index, center) in centers.rows().into_iter().enumerate() {
            let dot_product = row
                .indices()
                .iter()
                .zip(row.data().iter())
                .fold(F::zero(), |accumulator, (&column, &value)| {
                    accumulator + value * center[column]
                });
            distances[[row_index, center_index]] =
                row_norm + center_norms[center_index] - dot_product - dot_product;
        }
    }
    distances
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
    Ok(match features {
        SKDataView::Dense(dense) => sk_dense_squared_distances_to_centers(&dense, centers),
        SKDataView::Sparse(sparse) => sk_sparse_squared_distances_to_centers(&sparse, centers),
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    })
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
