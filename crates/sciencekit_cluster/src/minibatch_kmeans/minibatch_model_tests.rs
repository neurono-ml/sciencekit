//! Companion tests for [`SKMiniBatchKMeansModel`] traits, `predict`,
//! `transform`, `fit_predict`, scalar independence and state export (spec
//! `mini-batch-k-means`, task 4.2).

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClustererPredictor, SKExportable, SKImportable, SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::{SKMiniBatchKMeans, SKMiniBatchKMeansModel};

fn blobs() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.2, 0.1],
        [0.1, -0.1],
        [0.0, 0.2],
        [20.0, 20.0],
        [20.2, 19.9],
        [19.9, 20.1],
        [20.1, 20.2]
    ]
}

fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    dense
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, column), &value)| triplet.add_triplet(row, column, value));
    triplet.to_csr()
}

#[test]
fn minibatch_model_accessors_predict_clusters_and_predict_nearest_center() {
    let data = blobs();
    let model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .maximum_number_of_iterations(10)
        .backend(SKBackendKind::MatrixMultiply)
        .random_seed(13)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.cluster_centers().dim(), (2, 2));
    assert_eq!(model.counts().len(), 2);
    assert_eq!(model.number_of_clusters(), 2);
    assert_eq!(model.number_of_features(), 2);
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert!(
        model
            .labels()
            .unwrap()
            .iter()
            .all(|label| !label.is_noise())
    );

    let queries = array![[0.05_f64, 0.05], [20.05, 20.05], [3.0, 3.0]];
    let sparse_queries = to_csr(&queries);

    let dense_labels = model.predict_clusters(&queries).unwrap();
    let inherent_labels = model.predict(&queries).unwrap();
    let sparse_labels = model.predict_clusters(sparse_queries.view()).unwrap();
    assert_eq!(dense_labels, inherent_labels);
    assert_eq!(dense_labels, sparse_labels);
    assert!(dense_labels.iter().all(|label| !label.is_noise()));
    assert_ne!(dense_labels[0], dense_labels[1]);
    assert_eq!(dense_labels[0], dense_labels[2]);
}

#[test]
fn minibatch_model_transform_and_fit_predict_shapes_and_distances() {
    let data = blobs();
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .maximum_number_of_iterations(8)
        .random_seed(29)
        .build()
        .unwrap();

    let model = estimator.fit(&data).unwrap();
    let fit_predicted = estimator.fit_predict(&data).unwrap();
    assert_eq!(Some(fit_predicted.view()), model.labels());

    let queries = array![[0.1_f64, 0.0], [20.0, 20.1], [10.0, 10.0]];
    let sparse_queries = to_csr(&queries);
    let dense_distances = model.transform(&queries).unwrap();
    let sparse_distances = model.transform(sparse_queries.view()).unwrap();

    assert_eq!(dense_distances.dim(), (3, 2));
    assert_eq!(sparse_distances.dim(), (3, 2));
    dense_distances
        .iter()
        .zip(sparse_distances.iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-10));

    let labels = model.predict(&queries).unwrap();
    queries
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, query_row)| {
            let assigned = labels[row_index].cluster_index().unwrap() as usize;
            let other = 1 - assigned;
            assert!(dense_distances[[row_index, assigned]] <= dense_distances[[row_index, other]]);
            let center = model.cluster_centers().row(assigned).to_owned();
            let manual_euclidean = query_row
                .iter()
                .zip(center.iter())
                .map(|(&q, &c)| (q - c) * (q - c))
                .sum::<f64>()
                .sqrt();
            assert!((dense_distances[[row_index, assigned]] - manual_euclidean).abs() < 1e-10);
        });
}

#[test]
fn minibatch_model_scalar_independence_between_f32_and_f64() {
    let data64 = blobs();
    let data32 = data64.mapv(|value| value as f32);

    let model64 = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .maximum_number_of_iterations(8)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(&data64)
        .unwrap();
    let model32 = SKMiniBatchKMeans::<f32>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .maximum_number_of_iterations(8)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(&data32)
        .unwrap();

    let labels64 = model64.predict(&data64).unwrap();
    let labels32 = model32.predict(&data32).unwrap();
    assert_eq!(labels64, labels32);

    model64
        .cluster_centers()
        .iter()
        .zip(model32.cluster_centers().iter())
        .for_each(|(&value64, &value32)| assert!((value64 - value32 as f64).abs() < 1e-4));
}

#[test]
fn minibatch_model_export_and_import_round_trip_preserves_state_and_predictions() {
    let data = blobs();
    let model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .maximum_number_of_iterations(6)
        .backend(SKBackendKind::MatrixMultiply)
        .random_seed(41)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let exported = model.export_state().unwrap();
    let restored = SKMiniBatchKMeansModel::<f64>::import_state(&exported).unwrap();

    assert_eq!(restored.cluster_centers(), model.cluster_centers());
    assert_eq!(restored.counts(), model.counts());
    assert!((restored.inertia() - model.inertia()).abs() < 1e-12);
    assert_eq!(restored.batches_processed(), model.batches_processed());
    assert_eq!(restored.epochs_consumed(), model.epochs_consumed());
    assert_eq!(restored.converged(), model.converged());
    assert_eq!(restored.backend(), model.backend());
    assert_eq!(
        restored.predict(&data).unwrap(),
        model.predict(&data).unwrap()
    );
}
