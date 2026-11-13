//! Representation-parity acceptance scenarios.

use sciencekit_common::{SKBuilder, SKClustererPredictor, SKUnsupervisedFit};
use sprs::TriMat;

use super::fixtures::{csr_from_dense, planted_design};
use crate::SKKMeans;

#[test]
fn acceptance_representation_dense_and_sparse_agree() {
    let (data, truth) = planted_design(500, 4);
    let sparse = csr_from_dense(&data);
    let dense_model = SKKMeans::<f64>::build()
        .number_of_clusters(4)
        .random_seed(11)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let sparse_model = SKKMeans::<f64>::build()
        .number_of_clusters(4)
        .random_seed(11)
        .build()
        .unwrap()
        .fit(sparse.view())
        .unwrap();

    assert_eq!(dense_model.labels(), sparse_model.labels());
    assert!(
        (dense_model.inertia() - sparse_model.inertia()).abs()
            <= 1e-6 * (1.0 + dense_model.inertia().abs())
    );
    assert_eq!(
        super::fixtures::majority_accuracy(&truth, &dense_model.labels().to_owned(), 4),
        1.0
    );
}

#[test]
fn acceptance_representation_fit_predicts_sparse_queries_from_a_dense_model() {
    let (data, _) = planted_design(300, 3);
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .random_seed(4)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let sparse_queries = csr_from_dense(&data);

    assert_eq!(
        model.predict_clusters(&data).unwrap(),
        model.predict_clusters(sparse_queries.view()).unwrap()
    );
}

#[test]
fn acceptance_representation_wide_sparse_design_is_not_densified() {
    let width = 1_000_000;
    let mut triplet = TriMat::new((2, width));
    triplet.add_triplet(0, 0, 1.0);
    triplet.add_triplet(1, width - 1, 1.0);
    let sparse = triplet.to_csr();
    let nnz_before = sparse.nnz();

    let model = SKKMeans::<f64>::build()
        .number_of_clusters(1)
        .maximum_number_of_iterations(10)
        .build()
        .unwrap()
        .fit(sparse.view())
        .unwrap();

    assert_eq!(model.number_of_features(), width);
    assert_eq!(model.labels().len(), 2);
    assert_eq!(sparse.nnz(), nnz_before);
}
