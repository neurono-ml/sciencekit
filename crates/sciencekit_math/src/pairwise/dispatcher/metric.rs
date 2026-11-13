//! The metric catalogue and the single public dispatch entry.

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::routing::{
    sk_dispatch_representation, sk_has_sparse_side, sk_sparse_nan_unsupported,
    sk_unsupported_representation,
};
use crate::pairwise::sk_nan_euclidean_distance_matrix;

/// The closed catalogue of pairwise metrics offered by the dispatch surface.
///
/// The enum is exhaustive on purpose: the dispatcher matches it without dynamic
/// dispatch, mirroring the closed-catalogue style of `SKBackendKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKPairwiseMetric {
    /// Squared Euclidean distance, `‖a − b‖²`.
    SquaredEuclidean,
    /// Euclidean distance, `‖a − b‖`.
    Euclidean,
    /// Manhattan distance, `Σ |aᵢ − bᵢ|`.
    Manhattan,
    /// Chebyshev distance, `maxᵢ |aᵢ − bᵢ|`.
    Chebyshev,
    /// Cosine distance, `1 − cos θ`.
    Cosine,
    /// NaN-aware Euclidean distance; dense-only, since a CSR implicit zero is
    /// an observed value and carries no missing marker.
    NanEuclidean,
}

/// Compute a pairwise-distance matrix between the rows of two data views.
///
/// The metric and the two representations are matched once; the chosen kernel
/// then reads its representation directly, so no hot loop branches on the
/// representation and no sparse input is ever densified. Shapes are validated
/// before any distance is computed, and a sparse request for
/// [`SKPairwiseMetric::NanEuclidean`] is rejected up front with
/// [`SKError::UnsupportedRepresentation`].
pub fn sk_pairwise_distance_matrix<F: SKFloat>(
    metric: SKPairwiseMetric,
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (left_rows, left_features) = sk_view_dimensions(&left)?;
    let (right_rows, right_features) = sk_view_dimensions(&right)?;
    if left_features != right_features {
        return Err(SKError::shape_mismatch_2d(
            left_rows,
            left_features,
            right_rows,
            right_features,
        ));
    }
    if metric == SKPairwiseMetric::NanEuclidean && sk_has_sparse_side(&left, &right) {
        return Err(sk_sparse_nan_unsupported());
    }
    match metric {
        SKPairwiseMetric::NanEuclidean => {
            let (left_dense, right_dense) = (left.as_dense()?, right.as_dense()?);
            Ok(sk_nan_euclidean_distance_matrix(&left_dense, &right_dense))
        }
        _ => sk_dispatch_representation(metric, &left, &right),
    }
}

/// Dimensions `(rows, columns)` of either representation.
fn sk_view_dimensions<F>(view: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match view {
        SKDataView::Dense(dense) => Ok(dense.dim()),
        SKDataView::Sparse(sparse) => Ok((sparse.rows(), sparse.cols())),
        _ => Err(sk_unsupported_representation(view)),
    }
}
