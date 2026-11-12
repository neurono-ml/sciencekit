//! Parallel nearest-center assignment over dense and CSR views without
//! densification (spec `mini-batch-k-means`, decisions 3 and 7).

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsVecView;

use crate::kmeans::distance::sk_view_dimensions;

/// Assigns every row to its nearest center in parallel, returning `(labels,
/// squared_distances)`. Ties keep the smallest center index.
pub(crate) fn sk_assign_batch_rows<F: SKFloat>(
    features: SKDataView<'_, F>,
    centers: &ArrayView2<'_, F>,
) -> Result<(Array1<usize>, Array1<F>), SKError> {
    let (number_of_rows, number_of_features) = sk_view_dimensions(&features);
    if number_of_features != centers.ncols() {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_rows, centers.ncols()],
            found: vec![number_of_rows, number_of_features],
        });
    }
    let pairs: Vec<(usize, F)> = match features {
        SKDataView::Dense(dense) => dense
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| nearest_dense_center(row, centers))
            .collect(),
        SKDataView::Sparse(sparse) => {
            if !sparse.is_csr() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR before calling MiniBatchKMeans",
                });
            }
            let center_norms = compute_center_norms(centers);
            (0..sparse.rows())
                .into_par_iter()
                .map(|row_index| {
                    let stored = sparse
                        .outer_view(row_index)
                        .expect("row index comes from validated CSR bounds");
                    nearest_sparse_center(stored, centers, &center_norms)
                })
                .collect()
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    };
    let mut labels = Vec::with_capacity(pairs.len());
    let mut distances = Vec::with_capacity(pairs.len());
    pairs.into_iter().for_each(|(label, distance)| {
        labels.push(label);
        distances.push(distance);
    });
    Ok((Array1::from_vec(labels), Array1::from_vec(distances)))
}

/// Squared norms of all center rows.
pub(crate) fn compute_center_norms<F: SKFloat>(centers: &ArrayView2<'_, F>) -> Vec<F> {
    centers
        .rows()
        .into_iter()
        .map(|center| {
            center
                .iter()
                .fold(F::zero(), |accumulator, &value| accumulator + value * value)
        })
        .collect()
}

/// Squared distance from one dense row to one dense center.
pub(crate) fn dense_squared_distance<F: SKFloat>(
    row: ArrayView1<'_, F>,
    center: ArrayView1<'_, F>,
) -> F {
    row.iter()
        .zip(center.iter())
        .fold(F::zero(), |accumulator, (&left, &right)| {
            let difference = left - right;
            accumulator + difference * difference
        })
}

/// Squared distance from one CSR row to one dense center using norm expansion
/// over stored entries only.
pub(crate) fn sparse_squared_distance<F: SKFloat>(
    row: CsVecView<'_, F>,
    row_norm: F,
    center: ArrayView1<'_, F>,
    center_norm: F,
) -> F {
    let dot_product = row
        .indices()
        .iter()
        .zip(row.data().iter())
        .fold(F::zero(), |accumulator, (&column, &value)| {
            accumulator + value * center[column]
        });
    let raw = row_norm + center_norm - dot_product - dot_product;
    if raw > F::zero() { raw } else { F::zero() }
}

fn nearest_dense_center<F: SKFloat>(
    row: ArrayView1<'_, F>,
    centers: &ArrayView2<'_, F>,
) -> (usize, F) {
    centers.rows().into_iter().enumerate().fold(
        (0_usize, F::infinity()),
        |(best_index, best_distance), (index, center)| {
            let distance = dense_squared_distance(row, center);
            if distance < best_distance {
                (index, distance)
            } else {
                (best_index, best_distance)
            }
        },
    )
}

fn nearest_sparse_center<F: SKFloat>(
    row: CsVecView<'_, F>,
    centers: &ArrayView2<'_, F>,
    center_norms: &[F],
) -> (usize, F) {
    let row_norm = row
        .data()
        .iter()
        .fold(F::zero(), |accumulator, &value| accumulator + value * value);
    centers
        .rows()
        .into_iter()
        .zip(center_norms.iter())
        .enumerate()
        .fold(
            (0_usize, F::infinity()),
            |(best_index, best_distance), (index, (center, &center_norm))| {
                let distance = sparse_squared_distance(row, row_norm, center, center_norm);
                if distance < best_distance {
                    (index, distance)
                } else {
                    (best_index, best_distance)
                }
            },
        )
}
