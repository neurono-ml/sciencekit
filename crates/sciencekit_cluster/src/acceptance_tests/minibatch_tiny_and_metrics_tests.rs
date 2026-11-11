//! Tiny-data, full-KMeans comparison, export and `sciencekit_metrics`
//! acceptance scenarios for `SKMiniBatchKMeans` (spec `mini-batch-k-means`,
//! task 5.1).

use ndarray::{Array1, array};
use sciencekit_common::{
    SKBuilder, SKClustererPredictor, SKDataView, SKExportable, SKImportable, SKUnsupervisedFit,
    SKUnsupervisedScorer,
};
use sciencekit_metrics::clustering::{
    sk_adjusted_rand_score, sk_silhouette_score, sk_v_measure_score,
};
use sciencekit_metrics::scorers::SKSilhouetteScorer;

use super::fixtures::{majority_accuracy, planted_design};
use crate::{SKKMeans, SKKMeansInitialization, SKMiniBatchKMeans, SKMiniBatchKMeansModel};

#[test]
fn minibatch_acceptance_tiny_two_row_and_single_cluster_fits_plus_export() {
    let two_rows = array![[0.0_f64, 0.0], [6.0, 6.0]];
    let two_cluster_model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(2)
        .maximum_number_of_iterations(5)
        .random_seed(2)
        .build()
        .unwrap()
        .fit(&two_rows)
        .unwrap();

    assert_eq!(two_cluster_model.number_of_clusters(), 2);
    assert!(two_cluster_model.inertia().abs() < 1e-12);
    assert_ne!(
        two_cluster_model.labels().unwrap()[0],
        two_cluster_model.labels().unwrap()[1]
    );

    let single_cluster_data = array![[0.0_f64, 2.0], [2.0, 4.0]];
    let single_model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(1)
        .batch_size(2)
        .maximum_number_of_iterations(4)
        .build()
        .unwrap()
        .fit(&single_cluster_data)
        .unwrap();

    assert!((single_model.cluster_centers()[[0, 0]] - 1.0).abs() < 1e-12);
    assert!((single_model.cluster_centers()[[0, 1]] - 3.0).abs() < 1e-12);
    assert!((single_model.inertia() - 4.0).abs() < 1e-12);

    let exported = two_cluster_model.export_state().unwrap();
    let imported = SKMiniBatchKMeansModel::<f64>::import_state(&exported).unwrap();
    assert_eq!(
        imported.predict_clusters(&two_rows).unwrap(),
        two_cluster_model.predict_clusters(&two_rows).unwrap()
    );
}

#[test]
fn minibatch_acceptance_tiny_random_multi_init_produces_valid_non_noise_labels() {
    let (data, _) = planted_design(9, 3);
    let model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(3)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(4)
        .batch_size(3)
        .maximum_number_of_iterations(10)
        .random_seed(1)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    assert_eq!(model.number_of_clusters(), 3);
    assert!(model.inertia().is_finite());
    assert!(model.labels().unwrap().iter().all(|label| !label.is_noise()));
}

#[test]
fn minibatch_acceptance_agrees_with_full_kmeans_and_produces_clustering_metrics() {
    let (data, truth) = planted_design(600, 3);

    let full_kmeans = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .maximum_number_of_iterations(50)
        .random_seed(12)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let minibatch = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(3)
        .batch_size(60)
        .initialization_batch_size(180)
        .maximum_number_of_iterations(25)
        .random_seed(12)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let minibatch_labels = minibatch.labels().unwrap().to_owned();
    assert_eq!(majority_accuracy(&truth, &minibatch_labels, 3), 1.0);

    let kmeans_i64 = Array1::from_iter(full_kmeans.labels().iter().map(|label| label.to_i64()));
    let minibatch_i64 = Array1::from_iter(minibatch_labels.iter().map(|label| label.to_i64()));
    assert!((sk_adjusted_rand_score(kmeans_i64.view(), minibatch_i64.view()).unwrap() - 1.0).abs() < 1e-12);
    assert!((sk_adjusted_rand_score(truth.view(), minibatch_i64.view()).unwrap() - 1.0).abs() < 1e-12);
    assert!((sk_v_measure_score(truth.view(), minibatch_i64.view()).unwrap() - 1.0).abs() < 1e-12);

    let direct_silhouette =
        sk_silhouette_score(SKDataView::Dense(data.view()), minibatch_labels.view()).unwrap();
    let scorer_silhouette = SKSilhouetteScorer
        .score(&minibatch, SKDataView::Dense(data.view()))
        .unwrap();
    assert!(direct_silhouette > 0.99);
    assert!((direct_silhouette - scorer_silhouette).abs() < 1e-12);

    let mut kmeans_centers: Vec<[f64; 3]> = full_kmeans
        .cluster_centers()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let mut minibatch_centers: Vec<[f64; 3]> = minibatch
        .cluster_centers()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    kmeans_centers.sort_by(|left, right| left[0].partial_cmp(&right[0]).unwrap());
    minibatch_centers.sort_by(|left, right| left[0].partial_cmp(&right[0]).unwrap());

    kmeans_centers
        .iter()
        .zip(minibatch_centers.iter())
        .for_each(|(expected, actual)| {
            (0..3).for_each(|coord| assert!((expected[coord] - actual[coord]).abs() < 1e-2));
        });
    assert!(
        (minibatch.inertia() - full_kmeans.inertia()).abs()
            <= 1e-2 * (1.0 + full_kmeans.inertia().abs())
    );
}
