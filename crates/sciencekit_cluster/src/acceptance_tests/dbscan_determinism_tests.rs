//! Determinism acceptance scenarios for `SKDBSCAN`.

use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::dbscan_fixtures::planted_density_chains;
use crate::{SKDBSCAN, SKDBSCANAlgorithm};

#[test]
fn dbscan_acceptance_determinism_repeated_fits_and_thread_counts_are_identical() {
    let (data, _) = planted_density_chains(1_200, 3, 20, 40);

    for algorithm in [
        SKDBSCANAlgorithm::Auto,
        SKDBSCANAlgorithm::BallTree,
        SKDBSCANAlgorithm::BruteForce,
    ] {
        let estimator = SKDBSCAN::<f64>::build()
            .eps(0.45)
            .minimum_samples(5)
            .algorithm(algorithm)
            .build()
            .unwrap();

        let baseline = estimator.fit(&data).unwrap();
        let repeated = estimator.fit(&data).unwrap();
        assert_eq!(baseline.labels(), repeated.labels());
        assert_eq!(
            baseline.core_sample_indices(),
            repeated.core_sample_indices()
        );
        assert_eq!(baseline.number_of_clusters(), repeated.number_of_clusters());

        for threads in [1_usize, 2, 4] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let threaded = pool.install(|| estimator.fit(&data).unwrap());
            assert_eq!(threaded.labels(), baseline.labels());
            assert_eq!(
                threaded.core_sample_indices(),
                baseline.core_sample_indices()
            );
            assert_eq!(threaded.number_of_clusters(), baseline.number_of_clusters());
        }
    }
}
