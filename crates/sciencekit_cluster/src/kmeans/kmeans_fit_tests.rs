//! Companion tests for the `SKUnsupervisedFit` surface and model state.

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClusterLabel, SKExecutionMode, SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::{SKKMeans, SKKMeansAlgorithm, SKKMeansInitialization};

fn blobs() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1]
    ]
}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

fn estimator() -> SKKMeans<f64> {
    SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(17)
        .build()
        .unwrap()
}

#[test]
fn kmeans_fit_separates_two_well_separated_blobs() {
    let data = blobs();
    let model = estimator().fit(&data).unwrap();

    assert_eq!(model.cluster_centers().dim(), (2, 2));
    assert_eq!(model.number_of_clusters(), 2);
    assert_eq!(model.number_of_features(), 2);
    assert_eq!(model.labels().len(), 6);
    assert!(model.inertia() < 0.05);
    assert!(model.converged());
    assert!(model.iterations_taken() >= 1);
    assert_eq!(model.algorithm(), SKKMeansAlgorithm::Elkan);
    assert_eq!(model.backend(), SKBackendKind::Faer);
    assert_eq!(model.execution_mode(), SKExecutionMode::Automatic);

    let first = model.labels()[0];
    for row in 0..3 {
        assert_eq!(model.labels()[row], first);
    }
    for row in 3..6 {
        assert_ne!(model.labels()[row], first);
    }
}

#[test]
fn kmeans_fit_exhausted_budget_returns_an_unconverged_model() {
    let data = array![[0.0_f64], [1.0], [10.0], [11.0]];
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .maximum_number_of_iterations(1)
        .random_seed(3)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert!(!model.converged());
    assert_eq!(model.iterations_taken(), 1);
}

#[test]
fn kmeans_fit_labels_are_cluster_variants_never_noise() {
    let data = blobs();
    let model = estimator().fit(&data).unwrap();

    for label in model.labels().iter() {
        assert!(!label.is_noise());
        let index = label.cluster_index().expect("KMeans emits only clusters");
        assert!((index as f64) < 2.0);
    }
    assert!(model.labels().iter().all(|label| *label != SKClusterLabel::NOISE));
}

#[test]
fn kmeans_fit_inertia_equals_the_sum_of_squared_distances() {
    let data = blobs();
    let model = estimator().fit(&data).unwrap();
    let centers = model.cluster_centers();
    let mut expected = 0.0;
    for (row, label) in model.labels().iter().enumerate() {
        let cluster = label.cluster_index().unwrap() as usize;
        for feature in 0..data.ncols() {
            let difference = data[[row, feature]] - centers[[cluster, feature]];
            expected += difference * difference;
        }
    }
    assert!((model.inertia() - expected).abs() < 1e-9);
}

#[test]
fn kmeans_fit_transfers_the_configured_backend_into_the_model() {
    let data = blobs();
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

#[test]
fn kmeans_fit_multiple_initializations_never_lose_to_the_first() {
    let data = blobs();
    let single = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(1)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let multiple = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(5)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert!(multiple.inertia() <= single.inertia() + 1e-12);
}

#[test]
fn kmeans_fit_sparse_matches_dense() {
    let dense = blobs();
    let sparse = csr_from_dense(&dense);

    let dense_model = estimator().fit(&dense).unwrap();
    let sparse_model = estimator().fit(sparse.view()).unwrap();

    assert_eq!(dense_model.labels(), sparse_model.labels());
    for (dense_value, sparse_value) in dense_model
        .cluster_centers()
        .iter()
        .zip(sparse_model.cluster_centers().iter())
    {
        assert!((dense_value - sparse_value).abs() < 1e-9);
    }
    assert!((dense_model.inertia() - sparse_model.inertia()).abs() < 1e-9);
    assert_eq!(sparse_model.algorithm(), SKKMeansAlgorithm::Lloyd);
}
