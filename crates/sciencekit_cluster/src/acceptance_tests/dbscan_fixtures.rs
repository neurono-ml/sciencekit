//! Deterministic fixtures shared by the DBSCAN acceptance tests.

use ndarray::{Array1, Array2};

/// Planted density-connected chains plus isolated noise rows.
///
/// `cluster_rows` rows are distributed evenly across `clusters` well-separated
/// horizontal chains, each split into `stations_per_cluster` stations spaced
/// `0.25` apart so every station is density-reachable from its neighbours under
/// `eps = 0.45` while keeping per-point neighbourhood sizes small (`O(1)`).
/// `noise_rows` additional rows sit far from every chain and from each other,
/// carrying ground-truth label `-1`.
pub(crate) fn planted_density_chains(
    cluster_rows: usize,
    clusters: usize,
    stations_per_cluster: usize,
    noise_rows: usize,
) -> (Array2<f64>, Array1<i64>) {
    let total_rows = cluster_rows + noise_rows;
    let mut features = Array2::zeros((total_rows, 2));
    let mut truth = Array1::zeros(total_rows);

    for row in 0..cluster_rows {
        let cluster = row % clusters;
        let within_cluster = row / clusters;
        let station = within_cluster % stations_per_cluster;
        let jitter_x = ((row * 7) % 11) as f64 * 0.005;
        let jitter_y = ((row * 13) % 11) as f64 * 0.005;
        features[[row, 0]] = (cluster + 1) as f64 * 100.0 + station as f64 * 0.25 + jitter_x;
        features[[row, 1]] = (cluster + 1) as f64 * 10.0 + jitter_y;
        truth[row] = cluster as i64;
    }

    for offset in 0..noise_rows {
        let row = cluster_rows + offset;
        features[[row, 0]] = -500.0 - offset as f64 * 5.0;
        features[[row, 1]] = -500.0 - offset as f64 * 3.0;
        truth[row] = -1;
    }

    (features, truth)
}

/// Two concentric circles (radii `1.0` and `4.0`) plus isolated noise points.
///
/// Non-convex geometry that KMeans splits across a hyperplane, whereas DBSCAN
/// separates cleanly by density reachability along each ring.
pub(crate) fn concentric_rings_with_noise(
    points_per_ring: usize,
    noise_rows: usize,
) -> (Array2<f64>, Array1<i64>) {
    let total_rows = points_per_ring * 2 + noise_rows;
    let mut features = Array2::zeros((total_rows, 2));
    let mut truth = Array1::zeros(total_rows);
    let two_pi = std::f64::consts::TAU;

    for index in 0..points_per_ring {
        let angle = two_pi * (index as f64) / (points_per_ring as f64);
        features[[index, 0]] = angle.cos();
        features[[index, 1]] = angle.sin();
        truth[index] = 0;

        let outer = points_per_ring + index;
        features[[outer, 0]] = 4.0 * angle.cos();
        features[[outer, 1]] = 4.0 * angle.sin();
        truth[outer] = 1;
    }

    for offset in 0..noise_rows {
        let row = points_per_ring * 2 + offset;
        features[[row, 0]] = 20.0 + offset as f64 * 4.0;
        features[[row, 1]] = -20.0 - offset as f64 * 4.0;
        truth[row] = -1;
    }

    (features, truth)
}
