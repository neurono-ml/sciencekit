//! Elkan's triangle-inequality acceleration (spec `k-means`, decision 4).
//!
//! Bounds are Euclidean distances. After every centre move, `upper` grows by the
//! movement of the row's assigned centre and `lower` shrinks by the movement of
//! each centre; a row whose `upper` no longer matches its stored lower bound is
//! tightened with one exact distance. Only then is the centre scan started, and
//! it evaluates a centre exactly when neither the stored lower bound nor the
//! half-centre-distance test can rule it out. Every pruning decision is a proven
//! lower bound on the true distance, so the acceleration never changes the
//! assignment.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::distance::sk_dense_squared_distances_to_centers;
use super::iteration::sk_nearest_labels;
use super::iteration::{sk_euclidean_center_shifts, sk_has_converged, sk_update_centers};
use super::result::SKKMeansFitResult;

/// Elkan fit on dense data, returning the model state.
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

    let squared = sk_dense_squared_distances_to_centers(&features, &centers.view());
    let mut labels = sk_nearest_labels(&squared);
    let mut lower = Array2::from_shape_fn((number_of_rows, number_of_clusters), |(row, center)| {
        squared[[row, center]].sqrt()
    });
    let mut upper = Array1::from_iter((0..number_of_rows).map(|row| lower[[row, labels[row]]]));
    let mut assigned_squared =
        Array1::from_iter((0..number_of_rows).map(|row| squared[[row, labels[row]]]));
    let mut evaluations = 0;
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..maximum_number_of_iterations {
        let updated = sk_update_centers(
            SKDataView::Dense(features),
            &labels,
            &assigned_squared,
            number_of_clusters,
        );
        let shifts = sk_euclidean_center_shifts(&centers, &updated);
        let maximum_squared_shift = shifts.iter().fold(F::zero(), |maximum, shift| {
            let squared_shift = *shift * *shift;
            if squared_shift > maximum {
                squared_shift
            } else {
                maximum
            }
        });
        centers = updated;

        // The assigned centre moved away, so its distance can only grow.
        for row in 0..number_of_rows {
            upper[row] = upper[row] + shifts[labels[row]];
        }
        // Every centre moved, so every lower bound can only shrink.
        for row in 0..number_of_rows {
            for center in 0..number_of_clusters {
                let adjusted = lower[[row, center]] - shifts[center];
                lower[[row, center]] = adjusted.max(F::zero());
            }
        }

        let center_distances = sk_center_distances(&centers.view());
        assigned_squared = sk_elkan_assign(
            &features,
            &centers.view(),
            &mut labels,
            &mut lower,
            &mut upper,
            &center_distances,
            &mut evaluations,
        );
        iterations += 1;
        if sk_has_converged(maximum_squared_shift, tolerance, &centers.view()) {
            converged = true;
            break;
        }
    }

    let inertia = assigned_squared
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
/// assignment; exact distances are evaluated only when the bounds cannot rule a
/// centre out.
pub(crate) fn sk_elkan_assign<F: SKFloat>(
    features: &ArrayView2<'_, F>,
    centers: &ArrayView2<'_, F>,
    labels: &mut Array1<usize>,
    lower: &mut Array2<F>,
    upper: &mut Array1<F>,
    center_distances: &Array2<F>,
    evaluations: &mut usize,
) -> Array1<F> {
    let half = F::from(0.5).expect("one half fits the scalar");
    let mut assigned = Array1::zeros(features.nrows());
    for row in 0..features.nrows() {
        let assigned_center = labels[row];
        // Tighten the current centre when the bound no longer pins it down.
        if upper[row] > lower[[row, assigned_center]] {
            let exact = sk_distance_rows(features.row(row), centers.row(assigned_center));
            *evaluations += 1;
            upper[row] = exact;
            lower[[row, assigned_center]] = exact;
        }
        let mut best_center = assigned_center;
        let mut best_upper = upper[row];
        let mut best_exact = best_upper;
        for center in 0..centers.nrows() {
            if center == assigned_center {
                continue;
            }
            if best_upper <= lower[[row, center]] {
                continue;
            }
            if best_upper <= half * center_distances[[best_center, center]] {
                continue;
            }
            let distance = sk_distance_rows(features.row(row), centers.row(center));
            *evaluations += 1;
            lower[[row, center]] = distance;
            if distance < best_upper {
                best_center = center;
                best_upper = distance;
                best_exact = distance;
            }
        }
        if best_center != assigned_center {
            // The previous centre's exact distance stays a valid lower bound.
            lower[[row, assigned_center]] = upper[row];
        }
        labels[row] = best_center;
        upper[row] = best_exact;
        lower[[row, best_center]] = best_exact;
        assigned[row] = best_exact * best_exact;
    }
    assigned
}

/// Euclidean distances between every pair of centre rows.
fn sk_center_distances<F: SKFloat>(centers: &ArrayView2<'_, F>) -> Array2<F> {
    let number_of_clusters = centers.nrows();
    Array2::from_shape_fn((number_of_clusters, number_of_clusters), |(left, right)| {
        sk_distance_rows(centers.row(left), centers.row(right))
    })
}

/// Euclidean distance between two dense rows.
fn sk_distance_rows<F: SKFloat>(left: ArrayView1<'_, F>, right: ArrayView1<'_, F>) -> F {
    left.iter()
        .zip(right.iter())
        .fold(F::zero(), |accumulator, (left_value, right_value)| {
            let difference = *left_value - *right_value;
            accumulator + difference * difference
        })
        .sqrt()
}
