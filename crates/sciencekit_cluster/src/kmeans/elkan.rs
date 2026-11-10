//! Elkan's triangle-inequality acceleration (spec `k-means`, decision 4).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::distance::sk_dense_squared_distances_to_centers;
use super::iteration::sk_nearest_labels;
use super::iteration::{
    sk_euclidean_center_shifts, sk_has_converged, sk_update_centers,
};
use super::result::SKKMeansFitResult;

/// Elkan fit on dense data, returning the model state and the number of exact
/// distance evaluations performed by the pruned assignment passes.
pub(crate) fn sk_elkan_fit<F: SKFloat>(
    features: ArrayView2<'_, F>,
    initial_centers: Array2<F>,
    maximum_number_of_iterations: usize,
    tolerance: F,
) -> Result<SKKMeansFitResult<F>, SKError> {
    Ok(sk_elkan_fit_with_evaluations(
        features,
        initial_centers,
        maximum_number_of_iterations,
        tolerance,
    )?
    .0)
}

/// Elkan fit plus the accumulated exact-evaluation counter (used by tests to
/// prove the bounds actually prune).
pub(crate) fn sk_elkan_fit_with_evaluations<F: SKFloat>(
    features: ArrayView2<'_, F>,
    initial_centers: Array2<F>,
    maximum_number_of_iterations: usize,
    tolerance: F,
) -> Result<(SKKMeansFitResult<F>, usize), SKError> {
    let number_of_rows = features.nrows();
    let number_of_clusters = initial_centers.nrows();
    let mut centers = initial_centers;
    let mut distances = sk_dense_squared_distances_to_centers(&features, &centers.view());
    let mut labels = sk_nearest_labels(&distances);
    let mut lower = distances.clone();
    let mut upper = Array1::from_iter((0..number_of_rows).map(|row| distances[[row, labels[row]]]));
    let mut assigned = upper.clone();
    let mut evaluations = 0;
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..maximum_number_of_iterations {
        let updated = sk_update_centers(
            SKDataView::Dense(features),
            &labels,
            &assigned,
            number_of_clusters,
        );
        let shifts = sk_euclidean_center_shifts(&centers, &updated);
        let maximum_squared_shift = shifts.iter().fold(F::zero(), |maximum, shift| {
            if *shift * *shift > maximum {
                *shift * *shift
            } else {
                maximum
            }
        });
        centers = updated;
        for row in 0..number_of_rows {
            for center in 0..number_of_clusters {
                let adjusted = lower[[row, center]] - shifts[center];
                lower[[row, center]] = if adjusted > F::zero() {
                    adjusted
                } else {
                    F::zero()
                };
            }
        }
        let center_center_squared =
            sk_dense_squared_distances_to_centers(&centers.view(), &centers.view());
        assigned = sk_elkan_assign(
            &features,
            &centers.view(),
            &mut labels,
            &mut lower,
            &mut upper,
            &center_center_squared,
            &mut evaluations,
        );
        iterations += 1;
        if sk_has_converged(maximum_squared_shift, tolerance, &centers.view()) {
            converged = true;
            break;
        }
    }

    let inertia = assigned
        .iter()
        .fold(F::zero(), |accumulator, distance| accumulator + *distance);
    Ok((
        SKKMeansFitResult {
            centers,
            labels,
            inertia,
            iterations,
            converged,
        },
        evaluations,
    ))
}

/// One bound-pruned assignment pass. Every row starts from its current
/// assignment; exact distances are evaluated only when the lower bound and the
/// center-to-center triangle test cannot rule a center out.
pub(crate) fn sk_elkan_assign<F: SKFloat>(
    features: &ArrayView2<'_, F>,
    centers: &ArrayView2<'_, F>,
    labels: &mut Array1<usize>,
    lower: &mut Array2<F>,
    upper: &mut Array1<F>,
    center_center_squared: &Array2<F>,
    evaluations: &mut usize,
) -> Array1<F> {
    let four = F::from(4.0).expect("four fits the scalar");
    let mut assigned = Array1::zeros(features.nrows());
    for row in 0..features.nrows() {
        let mut best_center = labels[row];
        let mut best_upper = upper[row];
        for center in 0..centers.nrows() {
            if center == best_center {
                continue;
            }
            if best_upper <= lower[[row, center]] {
                continue;
            }
            if best_upper * four <= center_center_squared[[best_center, center]] {
                continue;
            }
            let distance = sk_squared_distance_rows(features.row(row), centers.row(center));
            *evaluations += 1;
            lower[[row, center]] = distance;
            if distance < best_upper {
                best_center = center;
                best_upper = distance;
            }
        }
        let exact = sk_squared_distance_rows(features.row(row), centers.row(best_center));
        labels[row] = best_center;
        upper[row] = exact;
        assigned[row] = exact;
    }
    assigned
}

/// Squared Euclidean distance between two dense rows.
fn sk_squared_distance_rows<F: SKFloat>(
    left: ArrayView1<'_, F>,
    right: ArrayView1<'_, F>,
) -> F {
    left.iter()
        .zip(right.iter())
        .fold(F::zero(), |accumulator, (left_value, right_value)| {
            let difference = *left_value - *right_value;
            accumulator + difference * difference
        })
}
