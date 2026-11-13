//! Algorithm resolution and dense/sparse dispatch (spec `k-means`, decision 4).

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::SKKMeansAlgorithm;
use super::elkan::sk_elkan_fit;
use super::lloyd::sk_lloyd_fit;
use super::result::SKKMeansFitResult;

/// Resolves the requested algorithm against the input representation.
///
/// `Auto` selects Elkan for dense data and Lloyd for sparse data. An explicit
/// `Elkan` request over sparse data resolves to Lloyd because the bound
/// bookkeeping is not profitable there; the assignments and centers are
/// unchanged, and the resolved algorithm is reported on the model.
pub(crate) fn sk_resolve_algorithm<F>(
    requested: SKKMeansAlgorithm,
    features: &SKDataView<'_, F>,
) -> SKKMeansAlgorithm {
    match (requested, features) {
        (SKKMeansAlgorithm::Auto, SKDataView::Dense(_)) => SKKMeansAlgorithm::Elkan,
        (SKKMeansAlgorithm::Auto, SKDataView::Sparse(_)) => SKKMeansAlgorithm::Lloyd,
        (SKKMeansAlgorithm::Elkan, SKDataView::Dense(_)) => SKKMeansAlgorithm::Elkan,
        (SKKMeansAlgorithm::Elkan, SKDataView::Sparse(_)) => SKKMeansAlgorithm::Lloyd,
        (SKKMeansAlgorithm::Lloyd, _) => SKKMeansAlgorithm::Lloyd,
        (SKKMeansAlgorithm::Auto, _) | (SKKMeansAlgorithm::Elkan, _) => SKKMeansAlgorithm::Lloyd,
    }
}

/// Runs one initialization with the requested algorithm, returning the result
/// and the algorithm that was actually executed.
pub(crate) fn sk_fit_with_algorithm<F: SKFloat>(
    features: SKDataView<'_, F>,
    initial_centers: Array2<F>,
    requested: SKKMeansAlgorithm,
    maximum_number_of_iterations: usize,
    tolerance: F,
) -> Result<(SKKMeansFitResult<F>, SKKMeansAlgorithm), SKError> {
    let resolved = sk_resolve_algorithm(requested, &features);
    let result = match resolved {
        SKKMeansAlgorithm::Elkan => sk_elkan_fit(
            features.as_dense()?,
            initial_centers,
            maximum_number_of_iterations,
            tolerance,
        )?,
        _ => sk_lloyd_fit(
            features,
            initial_centers,
            maximum_number_of_iterations,
            tolerance,
        )?,
    };
    Ok((result, resolved))
}
