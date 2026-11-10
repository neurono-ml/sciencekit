//! Companion tests for assignment prediction, fit_predict and transform.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClustererPredictor, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKKMeans, SKKMeansModel};

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

fn fitted() -> SKKMeansModel<f64> {
    SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(17)
        .build()
        .unwrap()
        .fit(&blobs())
        .unwrap()
}

fn cluster_indices(labels: &Array1<sciencekit_common::SKClusterLabel>) -> Vec<u32> {
    labels
        .iter()
        .map(|label| label.cluster_index().expect("KMeans never emits noise"))
        .collect()
}

#[test]
fn kmeans_predict_assigns_new_rows_to_their_nearest_center() {
    let model = fitted();
    let queries = array![[0.05_f64, 0.05], [10.05, 10.05]];
    let labels = model.predict_clusters(&queries).unwrap();
    let indices = cluster_indices(&labels);

    assert_eq!(indices.len(), 2);
    assert_ne!(indices[0], indices[1]);
    let centers = model.cluster_centers();
    for (row, expected) in queries.rows().into_iter().enumerate() {
        let nearest = (0..centers.nrows())
            .min_by(|left, right| {
                let left_distance = squared_distance(expected, centers.row(*left));
                let right_distance = squared_distance(expected, centers.row(*right));
                left_distance
                    .partial_cmp(&right_distance)
                    .expect("finite distances compare")
            })
            .unwrap();
        assert_eq!(indices[row] as usize, nearest);
    }
}

#[test]
fn kmeans_predict_sparse_matches_dense() {
    let model = fitted();
    let queries = array![[0.05_f64, 0.05], [10.05, 10.05], [4.0, 4.0]];
    let sparse = csr_from_dense(&queries);

    let dense_labels = model.predict_clusters(&queries).unwrap();
    let sparse_labels = model.predict_clusters(sparse.view()).unwrap();
    assert_eq!(dense_labels, sparse_labels);
}

#[test]
fn kmeans_predict_is_scalar_independent() {
    let data64 = blobs();
    let data32 = data64.mapv(|value| value as f32);
    let model64 = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(9)
        .build()
        .unwrap()
        .fit(&data64)
        .unwrap();
    let model32 = SKKMeans::<f32>::build()
        .number_of_clusters(2)
        .random_seed(9)
        .build()
        .unwrap()
        .fit(&data32)
        .unwrap();

    let labels64 = model64.predict_clusters(&data64).unwrap();
    let labels32 = model32.predict_clusters(&data32).unwrap();
    assert_eq!(
        cluster_indices(&labels64),
        labels32
            .iter()
            .map(|label| label.cluster_index().unwrap())
            .collect::<Vec<_>>()
    );
}

#[test]
fn kmeans_predict_rejects_feature_count_mismatch() {
    let model = fitted();
    let queries = array![[0.0_f64, 0.0, 0.0]];
    let error = model.predict_clusters(&queries).unwrap_err();

    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(found, vec![1, 3]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn kmeans_transform_returns_the_distance_to_every_center() {
    let model = fitted();
    let queries = array![[0.05_f64, 0.05], [10.05, 10.05], [4.0, 4.0]];
    let distances = model.transform(&queries).unwrap();
    assert_eq!(distances.dim(), (3, 2));

    let labels = model.predict_clusters(&queries).unwrap();
    let centers = model.cluster_centers();
    for (row, label) in labels.iter().enumerate() {
        let cluster = label.cluster_index().unwrap() as usize;
        let smallest = (0..distances.ncols())
            .min_by(|left, right| {
                distances[[row, *left]]
                    .partial_cmp(&distances[[row, *right]])
                    .expect("finite distances compare")
            })
            .unwrap();
        assert_eq!(smallest, cluster);
        let expected = squared_distance(queries.row(row), centers.row(cluster)).sqrt();
        assert!((distances[[row, cluster]] - expected).abs() < 1e-12);
    }
}

#[test]
fn kmeans_fit_predict_returns_the_training_assignments() {
    let data = blobs();
    let estimator = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(17)
        .build()
        .unwrap();
    let model = estimator.fit(&data).unwrap();
    let assignments = estimator.fit_predict(&data).unwrap();

    assert_eq!(assignments, model.labels());
}

fn squared_distance(
    left: ndarray::ArrayView1<'_, f64>,
    right: ndarray::ArrayView1<'_, f64>,
) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| {
            let difference = left_value - right_value;
            difference * difference
        })
        .sum()
}
