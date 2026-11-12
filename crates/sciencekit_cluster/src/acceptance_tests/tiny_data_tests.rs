//! Tiny-data acceptance scenarios.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::planted_design;
use crate::{SKKMeans, SKKMeansInitialization};

#[test]
fn acceptance_tiny_two_row_fit_converges_to_exact_centers() {
    let data = array![[0.0_f64, 0.0], [5.0, 5.0]];
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(2)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 2);
    assert!(model.inertia().abs() < 1e-12);
    assert!(model.converged());
    assert_eq!(model.labels().len(), 2);
    assert_ne!(model.labels()[0], model.labels()[1]);
}

#[test]
fn acceptance_tiny_single_cluster_returns_the_mean() {
    let data = array![[0.0_f64, 2.0], [2.0, 4.0]];
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(1)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.cluster_centers()[[0, 0]], 1.0);
    assert_eq!(model.cluster_centers()[[0, 1]], 3.0);
    assert!((model.inertia() - 4.0).abs() < 1e-12);
}

#[test]
fn acceptance_tiny_multiple_initializations_produce_a_finite_model() {
    let (data, _) = planted_design(9, 3);
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(4)
        .random_seed(1)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 3);
    assert!(model.inertia().is_finite());
    assert!(model.labels().iter().all(|label| !label.is_noise()));
}
