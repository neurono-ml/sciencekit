//! Large-data acceptance scenarios (at least 50 000 rows).

use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::{csr_from_dense, majority_accuracy, planted_design};
use crate::{SKKMeans, SKKMeansAlgorithm};

#[test]
fn acceptance_large_dense_fit_recovers_planted_clusters() {
    let (data, truth) = planted_design(50_000, 4);
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(4)
        .maximum_number_of_iterations(50)
        .tolerance(1e-4)
        .random_seed(0)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 4);
    assert_eq!(model.labels().len(), 50_000);
    assert!(model.inertia().is_finite());
    assert!(model.inertia() > 0.0);
    assert!(model.converged());
    assert_eq!(model.algorithm(), SKKMeansAlgorithm::Elkan);
    assert_eq!(
        majority_accuracy(&truth, &model.labels().to_owned(), 4),
        1.0
    );
}

#[test]
fn acceptance_large_sparse_fit_matches_the_dense_fit() {
    let (data, truth) = planted_design(50_000, 4);
    let sparse = csr_from_dense(&data);
    let dense_model = SKKMeans::<f64>::build()
        .number_of_clusters(4)
        .maximum_number_of_iterations(50)
        .random_seed(0)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let sparse_model = SKKMeans::<f64>::build()
        .number_of_clusters(4)
        .maximum_number_of_iterations(50)
        .random_seed(0)
        .build()
        .unwrap()
        .fit(sparse.view())
        .unwrap();

    assert_eq!(sparse_model.labels(), dense_model.labels());
    assert!(
        (sparse_model.inertia() - dense_model.inertia()).abs()
            <= 1e-6 * (1.0 + dense_model.inertia().abs())
    );
    assert_eq!(sparse_model.algorithm(), SKKMeansAlgorithm::Lloyd);
    assert_eq!(
        majority_accuracy(&truth, &sparse_model.labels().to_owned(), 4),
        1.0
    );
}
