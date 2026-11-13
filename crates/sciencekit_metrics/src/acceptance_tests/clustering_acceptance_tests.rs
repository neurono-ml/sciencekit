//! Large-data and concurrency acceptance tests for the clustering metrics.

use std::thread;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKClusterLabel, SKDataView};

use crate::clustering::{sk_calinski_harabasz_score, sk_davies_bouldin_score, sk_silhouette_score};
use crate::scorers::SKSilhouetteScorer;

const NUMBER_OF_ROWS: usize = 2_000;
const NUMBER_OF_FEATURES: usize = 3;

/// Two well-separated Gaussian-like blobs with deterministic offsets.
fn large_dataset() -> (Array2<f64>, Array1<SKClusterLabel>) {
    let mut features = Array2::<f64>::zeros((NUMBER_OF_ROWS, NUMBER_OF_FEATURES));
    let mut assignments = Vec::with_capacity(NUMBER_OF_ROWS);
    for (row, mut feature_row) in features.rows_mut().into_iter().enumerate() {
        let cluster = row / (NUMBER_OF_ROWS / 2);
        let base = if cluster == 0 { 0.0 } else { 10.0 };
        for (column, value) in feature_row.iter_mut().enumerate() {
            *value = base + ((row + column * 7) % 11) as f64 * 0.01;
        }
        assignments.push(SKClusterLabel::cluster(cluster as u32));
    }
    (features, Array1::from_vec(assignments))
}

#[test]
fn clustering_large_data_produces_the_expected_verdict() {
    let (features, assignments) = large_dataset();
    let silhouette =
        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    let davies_bouldin =
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    let calinski_harabasz =
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();

    assert!(silhouette > 0.9, "silhouette {silhouette}");
    assert!(davies_bouldin < 0.1, "Davies-Bouldin {davies_bouldin}");
    assert!(
        calinski_harabasz > silhouette.max(davies_bouldin),
        "Calinski-Harabasz {calinski_harabasz}"
    );
}

#[test]
fn clustering_concurrent_calls_agree_with_the_sequential_run() {
    let (features, assignments) = large_dataset();
    let sequential = (
        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view()).unwrap(),
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), assignments.view()).unwrap(),
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view()).unwrap(),
    );

    thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    (
                        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view())
                            .unwrap(),
                        sk_davies_bouldin_score(
                            SKDataView::Dense(features.view()),
                            assignments.view(),
                        )
                        .unwrap(),
                        sk_calinski_harabasz_score(
                            SKDataView::Dense(features.view()),
                            assignments.view(),
                        )
                        .unwrap(),
                    )
                })
            })
            .collect();
        for handle in handles {
            let observed = handle.join().expect("the worker did not panic");
            assert_eq!(observed.0, sequential.0);
            assert_eq!(observed.1, sequential.1);
            assert_eq!(observed.2, sequential.2);
        }
    });
}

fn assert_send_and_sync<T: Send + Sync>() {}

#[test]
fn clustering_scorer_is_send_and_sync() {
    assert_send_and_sync::<SKSilhouetteScorer>();
}
