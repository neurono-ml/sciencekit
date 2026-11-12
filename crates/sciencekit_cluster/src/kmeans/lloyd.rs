//! Lloyd's assignment/update loop (spec `k-means`, decision 4).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::distance::sk_squared_distances_to_centers;
use super::iteration::{
    sk_has_converged, sk_maximum_squared_center_shift, sk_nearest_labels, sk_update_centers,
};
use super::result::SKKMeansFitResult;

/// Alternates nearest-center assignment and mean updates until the center
/// movement meets the tolerance or the iteration budget is exhausted.
pub(crate) fn sk_lloyd_fit<F: SKFloat>(
    features: SKDataView<'_, F>,
    initial_centers: Array2<F>,
    maximum_number_of_iterations: usize,
    tolerance: F,
) -> Result<SKKMeansFitResult<F>, SKError> {
    let mut centers = initial_centers;
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..maximum_number_of_iterations {
        let distances = sk_squared_distances_to_centers(features, &centers.view())?;
        let labels = sk_nearest_labels(&distances);
        let assigned_distances =
            Array1::from_iter((0..distances.nrows()).map(|row| distances[[row, labels[row]]]));
        let updated = sk_update_centers(features, &labels, &assigned_distances, centers.nrows());
        let maximum_shift = sk_maximum_squared_center_shift(&centers, &updated);
        centers = updated;
        iterations += 1;
        if sk_has_converged(maximum_shift, tolerance, &centers.view()) {
            converged = true;
            break;
        }
    }

    let distances = sk_squared_distances_to_centers(features, &centers.view())?;
    let labels = sk_nearest_labels(&distances);
    let inertia = labels
        .iter()
        .enumerate()
        .fold(F::zero(), |accumulator, (row, cluster)| {
            accumulator + distances[[row, *cluster]]
        });
    Ok(SKKMeansFitResult {
        centers,
        labels,
        inertia,
        iterations,
        converged,
    })
}
