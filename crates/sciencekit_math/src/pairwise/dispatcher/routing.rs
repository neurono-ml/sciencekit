//! The pairwise dispatch surface: one metric enum and one entry point that
//! selects the dense or the sparse kernel exactly once per operation
//! (spec `pairwise-distances`, `data-view-boundary`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsMatView;

use crate::pairwise::sparse::{
    sk_dense_sparse_chebyshev_distance_matrix, sk_dense_sparse_cosine_distance_matrix,
    sk_dense_sparse_euclidean_distance_matrix, sk_dense_sparse_manhattan_distance_matrix,
    sk_dense_sparse_squared_euclidean_distance_matrix, sk_sparse_chebyshev_distance_matrix,
    sk_sparse_cosine_distance_matrix, sk_sparse_dense_chebyshev_distance_matrix,
    sk_sparse_dense_cosine_distance_matrix, sk_sparse_dense_euclidean_distance_matrix,
    sk_sparse_dense_manhattan_distance_matrix, sk_sparse_dense_squared_euclidean_distance_matrix,
    sk_sparse_euclidean_distance_matrix, sk_sparse_manhattan_distance_matrix,
    sk_sparse_squared_euclidean_distance_matrix,
};
use crate::pairwise::{
    sk_chebyshev_distance_matrix, sk_cosine_distance_matrix, sk_euclidean_distance_matrix,
    sk_manhattan_distance_matrix, sk_nan_euclidean_distance_matrix,
    sk_squared_euclidean_distance_matrix,
};

use super::metric::SKPairwiseMetric;

/// Rejection for the dense-only NaN-aware metric under a sparse input.
pub(super) fn sk_sparse_nan_unsupported() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "csr",
        suggestion: "convert the sparse matrix to dense with to_dense; CSR implicit zeros are observed values, so no missing marker exists",
    }
}

/// Whether either side is sparse.
pub(super) fn sk_has_sparse_side<F>(left: &SKDataView<'_, F>, right: &SKDataView<'_, F>) -> bool {
    matches!(left, SKDataView::Sparse(_)) || matches!(right, SKDataView::Sparse(_))
}

/// Route a non-NaN metric to the kernel matching the representation pair.
pub(super) fn sk_dispatch_representation<F: SKFloat>(
    metric: SKPairwiseMetric,
    left: &SKDataView<'_, F>,
    right: &SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    match (left, right) {
        (SKDataView::Dense(left_dense), SKDataView::Dense(right_dense)) => {
            Ok(sk_dense_metric(metric, left_dense, right_dense))
        }
        (SKDataView::Sparse(left_sparse), SKDataView::Sparse(right_sparse)) => {
            sk_sparse_sparse_metric(metric, left_sparse, right_sparse)
        }
        (SKDataView::Sparse(left_sparse), SKDataView::Dense(right_dense)) => {
            sk_sparse_dense_metric(metric, left_sparse, right_dense)
        }
        (SKDataView::Dense(left_dense), SKDataView::Sparse(right_sparse)) => {
            sk_dense_sparse_metric(metric, left_dense, right_sparse)
        }
        _ => Err(sk_unsupported_representation(left)),
    }
}

/// Route a metric to its dense kernel.
fn sk_dense_metric<F: SKFloat>(
    metric: SKPairwiseMetric,
    left: &ArrayView2<F>,
    right: &ArrayView2<F>,
) -> Array2<F> {
    match metric {
        SKPairwiseMetric::SquaredEuclidean => sk_squared_euclidean_distance_matrix(left, right),
        SKPairwiseMetric::Euclidean => sk_euclidean_distance_matrix(left, right),
        SKPairwiseMetric::Manhattan => sk_manhattan_distance_matrix(left, right),
        SKPairwiseMetric::Chebyshev => sk_chebyshev_distance_matrix(left, right),
        SKPairwiseMetric::Cosine => sk_cosine_distance_matrix(left, right),
        SKPairwiseMetric::NanEuclidean => sk_nan_euclidean_distance_matrix(left, right),
    }
}

/// Route a metric to its sparse×sparse kernel.
fn sk_sparse_sparse_metric<F: SKFloat>(
    metric: SKPairwiseMetric,
    left: &CsMatView<F>,
    right: &CsMatView<F>,
) -> Result<Array2<F>, SKError> {
    match metric {
        SKPairwiseMetric::SquaredEuclidean => {
            Ok(sk_sparse_squared_euclidean_distance_matrix(left, right))
        }
        SKPairwiseMetric::Euclidean => Ok(sk_sparse_euclidean_distance_matrix(left, right)),
        SKPairwiseMetric::Manhattan => Ok(sk_sparse_manhattan_distance_matrix(left, right)),
        SKPairwiseMetric::Chebyshev => Ok(sk_sparse_chebyshev_distance_matrix(left, right)),
        SKPairwiseMetric::Cosine => Ok(sk_sparse_cosine_distance_matrix(left, right)),
        SKPairwiseMetric::NanEuclidean => Err(sk_sparse_nan_unsupported()),
    }
}

/// Route a metric to its sparse×dense kernel.
fn sk_sparse_dense_metric<F: SKFloat>(
    metric: SKPairwiseMetric,
    left: &CsMatView<F>,
    right: &ArrayView2<F>,
) -> Result<Array2<F>, SKError> {
    match metric {
        SKPairwiseMetric::SquaredEuclidean => Ok(
            sk_sparse_dense_squared_euclidean_distance_matrix(left, right),
        ),
        SKPairwiseMetric::Euclidean => Ok(sk_sparse_dense_euclidean_distance_matrix(left, right)),
        SKPairwiseMetric::Manhattan => Ok(sk_sparse_dense_manhattan_distance_matrix(left, right)),
        SKPairwiseMetric::Chebyshev => Ok(sk_sparse_dense_chebyshev_distance_matrix(left, right)),
        SKPairwiseMetric::Cosine => Ok(sk_sparse_dense_cosine_distance_matrix(left, right)),
        SKPairwiseMetric::NanEuclidean => Err(sk_sparse_nan_unsupported()),
    }
}

/// Route a metric to its dense×sparse kernel.
fn sk_dense_sparse_metric<F: SKFloat>(
    metric: SKPairwiseMetric,
    left: &ArrayView2<F>,
    right: &CsMatView<F>,
) -> Result<Array2<F>, SKError> {
    match metric {
        SKPairwiseMetric::SquaredEuclidean => Ok(
            sk_dense_sparse_squared_euclidean_distance_matrix(left, right),
        ),
        SKPairwiseMetric::Euclidean => Ok(sk_dense_sparse_euclidean_distance_matrix(left, right)),
        SKPairwiseMetric::Manhattan => Ok(sk_dense_sparse_manhattan_distance_matrix(left, right)),
        SKPairwiseMetric::Chebyshev => Ok(sk_dense_sparse_chebyshev_distance_matrix(left, right)),
        SKPairwiseMetric::Cosine => Ok(sk_dense_sparse_cosine_distance_matrix(left, right)),
        SKPairwiseMetric::NanEuclidean => Err(sk_sparse_nan_unsupported()),
    }
}

/// Rejection for a representation this library version does not understand.
pub(super) fn sk_unsupported_representation<F>(view: &SKDataView<'_, F>) -> SKError {
    SKError::UnsupportedRepresentation {
        representation: view.representation(),
        suggestion: "convert the input to a supported dense or CSR view",
    }
}
