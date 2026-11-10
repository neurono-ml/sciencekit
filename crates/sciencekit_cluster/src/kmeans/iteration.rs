//! Shared assignment, center-update and convergence helpers (spec `k-means`,
//! decisions 6 and 7).

use ndarray::{Array1, Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKFloat};

use super::distance::sk_view_dimensions;

/// Nearest center per row; exact ties keep the smallest center index.
pub(crate) fn sk_nearest_labels<F: SKFloat>(distances: &Array2<F>) -> Array1<usize> {
    let mut labels = Array1::zeros(distances.nrows());
    for row in 0..distances.nrows() {
        let mut best_center = 0;
        let mut best_distance = distances[[row, 0]];
        for center in 1..distances.ncols() {
            let distance = distances[[row, center]];
            if distance < best_distance {
                best_center = center;
                best_distance = distance;
            }
        }
        labels[row] = best_center;
    }
    labels
}

/// Mean of every cluster, repairing empty clusters deterministically with the
/// farthest training point (largest distance, then smallest row index).
pub(crate) fn sk_update_centers<F: SKFloat>(
    features: SKDataView<'_, F>,
    labels: &Array1<usize>,
    assigned_distances: &Array1<F>,
    number_of_clusters: usize,
) -> Array2<F> {
    let number_of_features = sk_view_dimensions(&features).1;
    let (effective_labels, counts) =
        repaired_labels(labels, assigned_distances, number_of_clusters);
    let mut sums = Array2::<F>::zeros((number_of_clusters, number_of_features));
    match features {
        SKDataView::Dense(dense) => {
            for row in 0..dense.nrows() {
                let cluster = effective_labels[row];
                let mut sum_row = sums.row_mut(cluster);
                for (sum, value) in sum_row.iter_mut().zip(dense.row(row).iter()) {
                    *sum = *sum + *value;
                }
            }
        }
        SKDataView::Sparse(sparse) => {
            for (row, stored) in sparse.outer_iterator().enumerate() {
                let cluster = effective_labels[row];
                for (column, value) in stored.iter() {
                    sums[[cluster, column]] = sums[[cluster, column]] + *value;
                }
            }
        }
        _ => {}
    }

    let mut centers = Array2::<F>::zeros((number_of_clusters, number_of_features));
    for cluster in 0..number_of_clusters {
        if counts[cluster] > 0 {
            let divisor = F::from(counts[cluster]).expect("a row count fits the scalar");
            for feature in 0..number_of_features {
                centers[[cluster, feature]] = sums[[cluster, feature]] / divisor;
            }
        }
    }
    centers
}

/// Deterministic empty-cluster repair: move the farthest still-claimed point
/// into each empty center, ascending by cluster index.
fn repaired_labels<F: SKFloat>(
    labels: &Array1<usize>,
    assigned_distances: &Array1<F>,
    number_of_clusters: usize,
) -> (Array1<usize>, Vec<usize>) {
    let mut effective = labels.clone();
    let mut counts = vec![0_usize; number_of_clusters];
    for cluster in effective.iter() {
        counts[*cluster] += 1;
    }
    let mut used_rows: Vec<usize> = Vec::new();
    for empty_cluster in 0..number_of_clusters {
        if counts[empty_cluster] > 0 {
            continue;
        }
        let candidate = (0..labels.len())
            .filter(|row| !used_rows.contains(row))
            .filter(|row| counts[effective[*row]] > 1)
            .fold(None, |best: Option<(usize, F)>, row| {
                let distance = assigned_distances[row];
                match best {
                    Some((_, best_distance)) if distance <= best_distance => best,
                    _ => Some((row, distance)),
                }
            })
            .or_else(|| {
                (0..labels.len())
                    .filter(|row| !used_rows.contains(row))
                    .filter(|row| counts[effective[*row]] > 0)
                    .fold(None, |best: Option<(usize, F)>, row| {
                        let distance = assigned_distances[row];
                        match best {
                            Some((_, best_distance)) if distance <= best_distance => best,
                            _ => Some((row, distance)),
                        }
                    })
            });
        let Some((row, _)) = candidate else {
            continue;
        };
        used_rows.push(row);
        counts[effective[row]] -= 1;
        effective[row] = empty_cluster;
        counts[empty_cluster] += 1;
    }
    (effective, counts)
}

/// Largest squared movement between the old and new centers.
pub(crate) fn sk_maximum_squared_center_shift<F: SKFloat>(
    previous: &Array2<F>,
    current: &Array2<F>,
) -> F {
    let mut maximum = F::zero();
    for center in 0..previous.nrows() {
        let shift = previous
            .row(center)
            .iter()
            .zip(current.row(center).iter())
            .fold(F::zero(), |accumulator, (old, new)| {
                let difference = *old - *new;
                accumulator + difference * difference
            });
        if shift > maximum {
            maximum = shift;
        }
    }
    maximum
}

/// Euclidean movement of every center, used by the Elkan bound update.
pub(crate) fn sk_euclidean_center_shifts<F: SKFloat>(
    previous: &Array2<F>,
    current: &Array2<F>,
) -> Array1<F> {
    Array1::from_iter((0..previous.nrows()).map(|center| {
        let squared = previous
            .row(center)
            .iter()
            .zip(current.row(center).iter())
            .fold(F::zero(), |accumulator, (old, new)| {
                let difference = *old - *new;
                accumulator + difference * difference
            });
        squared.sqrt()
    }))
}

/// Mean squared norm of the center rows.
pub(crate) fn sk_mean_squared_norm<F: SKFloat>(centers: &ArrayView2<'_, F>) -> F {
    let total = centers
        .rows()
        .into_iter()
        .fold(F::zero(), |accumulator, row| {
            accumulator
                + row
                    .iter()
                    .fold(F::zero(), |norm, value| norm + *value * *value)
        });
    total / F::from(centers.nrows()).expect("a cluster count fits the scalar")
}

/// The documented convergence rule: maximum squared center movement at or
/// below `tolerance × mean squared norm of the centers`.
pub(crate) fn sk_has_converged<F: SKFloat>(
    maximum_squared_shift: F,
    tolerance: F,
    centers: &ArrayView2<'_, F>,
) -> bool {
    maximum_squared_shift <= tolerance * sk_mean_squared_norm(centers)
}
