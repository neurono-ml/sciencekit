//! Determinism across repeated fits and thread counts.

use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::planted_design;
use crate::SKKMeans;

fn estimator() -> SKKMeans<f64> {
    SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .maximum_number_of_iterations(40)
        .random_seed(23)
        .build()
        .unwrap()
}

#[test]
fn acceptance_determinism_repeated_fits_are_identical() {
    let (data, _) = planted_design(2_000, 3);
    let first = estimator().fit(&data).unwrap();
    let second = estimator().fit(&data).unwrap();

    assert_eq!(first.cluster_centers(), second.cluster_centers());
    assert_eq!(first.labels(), second.labels());
    assert_eq!(first.inertia(), second.inertia());
    assert_eq!(first.iterations_taken(), second.iterations_taken());
}

#[test]
fn acceptance_determinism_is_independent_of_the_thread_count() {
    let (data, _) = planted_design(2_000, 3);
    let single_thread = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four_threads = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let single = single_thread.install(|| estimator().fit(&data)).unwrap();
    let parallel = four_threads.install(|| estimator().fit(&data)).unwrap();

    assert_eq!(single.cluster_centers(), parallel.cluster_centers());
    assert_eq!(single.labels(), parallel.labels());
    assert_eq!(single.inertia(), parallel.inertia());
}

#[test]
fn acceptance_determinism_multiple_initializations_are_stable() {
    let (data, _) = planted_design(1_000, 3);
    let estimator = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .initialization(crate::SKKMeansInitialization::Random)
        .number_of_initializations(6)
        .random_seed(77)
        .build()
        .unwrap();
    let first = estimator.fit(&data).unwrap();
    let second = estimator.fit(&data).unwrap();

    assert_eq!(first.cluster_centers(), second.cluster_centers());
    assert_eq!(first.labels(), second.labels());
    assert_eq!(first.inertia(), second.inertia());
}
