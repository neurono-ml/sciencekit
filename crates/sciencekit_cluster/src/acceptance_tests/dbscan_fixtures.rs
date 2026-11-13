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
    let features = Array2::from_shape_fn((total_rows, 2), |(row, column)| {
        if row < cluster_rows {
            let cluster = row % clusters;
            let within_cluster = row / clusters;
            let station = within_cluster % stations_per_cluster;
            if column == 0 {
                let jitter_x = ((row * 7) % 11) as f64 * 0.005;
                (cluster + 1) as f64 * 100.0 + station as f64 * 0.25 + jitter_x
            } else {
                let jitter_y = ((row * 13) % 11) as f64 * 0.005;
                (cluster + 1) as f64 * 10.0 + jitter_y
            }
        } else {
            let offset = (row - cluster_rows) as f64;
            if column == 0 {
                -500.0 - offset * 5.0
            } else {
                -500.0 - offset * 3.0
            }
        }
    });
    let truth = Array1::from_shape_fn(total_rows, |row| {
        if row < cluster_rows {
            (row % clusters) as i64
        } else {
            -1
        }
    });
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
    let two_pi = std::f64::consts::TAU;
    let features = Array2::from_shape_fn((total_rows, 2), |(row, column)| {
        if row < points_per_ring {
            let angle = two_pi * (row as f64) / (points_per_ring as f64);
            if column == 0 {
                angle.cos()
            } else {
                angle.sin()
            }
        } else if row < points_per_ring * 2 {
            let angle = two_pi * ((row - points_per_ring) as f64) / (points_per_ring as f64);
            if column == 0 {
                4.0 * angle.cos()
            } else {
                4.0 * angle.sin()
            }
        } else {
            let offset = (row - points_per_ring * 2) as f64;
            if column == 0 {
                20.0 + offset * 4.0
            } else {
                -20.0 - offset * 4.0
            }
        }
    });
    let truth = Array1::from_shape_fn(total_rows, |row| {
        if row < points_per_ring {
            0
        } else if row < points_per_ring * 2 {
            1
        } else {
            -1
        }
    });
    (features, truth)
}
