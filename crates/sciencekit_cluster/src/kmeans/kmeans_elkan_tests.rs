//! Companion tests for Elkan's bound-pruned assignment.

use ndarray::{Array1, Array2, array};
use sciencekit_common::SKDataView;

use super::distance::sk_dense_squared_distances_to_centers;
use super::elkan::{sk_elkan_assign, sk_elkan_fit};
use super::iteration::sk_nearest_labels;
use super::iteration::sk_update_centers;
use super::lloyd::sk_lloyd_fit;
use crate::{SKKMeans, SKKMeansAlgorithm};
use sciencekit_common::SKUnsupervisedFit;

fn separated_blobs() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1]
    ]
}

fn initial_centers() -> Array2<f64> {
    array![[0.0_f64, 0.0], [10.0, 10.0]]
}

#[test]
fn kmeans_elkan_equals_lloyd_on_dense_data() {
    let data = separated_blobs();
    let lloyd = sk_lloyd_fit(SKDataView::Dense(data.view()), initial_centers(), 100, 1e-9).unwrap();
    let elkan = sk_elkan_fit(data.view(), initial_centers(), 100, 1e-9).unwrap();

    assert_eq!(elkan.labels, lloyd.labels);
    assert!(elkan.converged);
    assert_eq!(elkan.converged, lloyd.converged);
    assert!(elkan.iterations.abs_diff(lloyd.iterations) <= 1);
    for (elkan_value, lloyd_value) in elkan.centers.iter().zip(lloyd.centers.iter()) {
        assert!((elkan_value - lloyd_value).abs() < 1e-9);
    }
    assert!((elkan.inertia - lloyd.inertia).abs() < 1e-9);
}

#[test]
fn kmeans_elkan_equals_lloyd_from_a_seeded_initialization() {
    let data = separated_blobs();
    let centers = super::init::sk_initialize_centers(
        SKDataView::Dense(data.view()),
        2,
        Default::default(),
        4,
    )
    .unwrap();
    let lloyd = sk_lloyd_fit(SKDataView::Dense(data.view()), centers.clone(), 100, 1e-9).unwrap();
    let elkan = sk_elkan_fit(data.view(), centers, 100, 1e-9).unwrap();

    assert_eq!(elkan.labels, lloyd.labels);
    assert!((elkan.inertia - lloyd.inertia).abs() < 1e-9);
}

#[test]
fn kmeans_elkan_prunes_impossible_distances() {
    let data = separated_blobs();
    let centers = initial_centers();
    let distances = sk_dense_squared_distances_to_centers(&data.view(), &centers.view());
    let mut labels = sk_nearest_labels(&distances);
    let mut lower = distances.clone();
    let mut upper = Array1::from_iter((0..data.nrows()).map(|row| distances[[row, labels[row]]]));
    let center_center_squared =
        sk_dense_squared_distances_to_centers(&centers.view(), &centers.view());
    let mut evaluations = 0;

    let assigned = sk_elkan_assign(
        &data.view(),
        &centers.view(),
        &mut labels,
        &mut lower,
        &mut upper,
        &center_center_squared,
        &mut evaluations,
    );

    assert_eq!(evaluations, 0);
    assert!(evaluations < data.nrows() * centers.nrows());
    for row in 0..data.nrows() {
        assert_eq!(labels[row], if row < 3 { 0 } else { 1 });
        assert!(assigned[row] < 0.05);
    }
}

#[test]
fn kmeans_elkan_repairs_an_empty_cluster() {
    let data = array![[0.0_f64], [0.1], [5.0], [10.0]];
    let initial_centers = array![[0.0_f64], [5.0], [100.0]];
    let result = sk_elkan_fit(data.view(), initial_centers, 20, 1e-6).unwrap();

    assert_eq!(result.centers.nrows(), 3);
    assert!(result.centers.iter().all(|value| value.is_finite()));
}

#[test]
fn kmeans_elkan_empty_cluster_update_uses_exact_distances() {
    let data = array![[0.0_f64], [0.1], [5.0], [10.0]];
    let labels = Array1::from(vec![0_usize, 0, 1, 1]);
    let assigned_distances = Array1::from(vec![0.0_f64, 0.0, 0.0, 25.0]);
    let centers = sk_update_centers(
        SKDataView::Dense(data.view()),
        &labels,
        &assigned_distances,
        3,
    );
    assert_eq!(centers.nrows(), 3);
}

/// A small deterministic pseudo-random source, so the overlap test needs no
/// external crate.
struct FixtureGenerator(u64);

impl FixtureGenerator {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

/// Overlapping, non-separable data: the acceleration must still match Lloyd
/// exactly. Well-separated blobs hide stale-bound bugs, so this fixture draws
/// every coordinate from one wide distribution with no planted clusters.
#[test]
fn kmeans_elkan_matches_lloyd_on_overlapping_data() {
    use sciencekit_common::SKBuilder;

    for seed in [0_u64, 1, 7, 42, 1234] {
        let mut generator = FixtureGenerator(seed.wrapping_add(17));
        let mut values = Vec::with_capacity(120 * 3);
        for _ in 0..120 * 3 {
            values.push(generator.next_unit() * 4.0 - 2.0);
        }
        let features = Array2::from_shape_vec((120, 3), values).expect("shape");
        let lloyd = SKKMeans::build()
            .number_of_clusters(4)
            .algorithm(SKKMeansAlgorithm::Lloyd)
            .random_seed(seed)
            .build()
            .expect("build")
            .fit(&features)
            .expect("fit");
        let elkan = SKKMeans::build()
            .number_of_clusters(4)
            .algorithm(SKKMeansAlgorithm::Elkan)
            .random_seed(seed)
            .build()
            .expect("build")
            .fit(&features)
            .expect("fit");
        assert_eq!(
            lloyd.labels(),
            elkan.labels(),
            "seed {seed}: Elkan must return Lloyd's assignment"
        );
        let relative_gap = (lloyd.inertia() - elkan.inertia()).abs() / lloyd.inertia().max(1.0);
        assert!(
            relative_gap < 1e-9,
            "seed {seed}: inertia gap {relative_gap}"
        );
    }
}
