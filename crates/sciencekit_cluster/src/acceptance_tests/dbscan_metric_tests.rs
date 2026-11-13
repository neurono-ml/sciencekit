//! Clustering-metric acceptance scenarios for `SKDBSCAN`.

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKDataView, SKUnsupervisedFit, SKUnsupervisedScorer};
use sciencekit_metrics::clustering::{
    sk_adjusted_rand_score, sk_silhouette_score, sk_v_measure_score,
};
use sciencekit_metrics::scorers::SKSilhouetteScorer;

use super::dbscan_fixtures::{concentric_rings_with_noise, planted_density_chains};
use crate::{SKDBSCAN, SKKMeans};

#[test]
fn dbscan_acceptance_metrics_silhouette_excludes_noise_and_scores_fitted_model() {
    let (data, _) = planted_density_chains(240, 3, 4, 15);
    let model = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(5)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let direct_silhouette =
        sk_silhouette_score(SKDataView::Dense(data.view()), model.labels()).unwrap();
    let scorer_silhouette = SKSilhouetteScorer
        .score(&model, SKDataView::Dense(data.view()))
        .unwrap();

    assert!(direct_silhouette > 0.95);
    assert!((direct_silhouette - scorer_silhouette).abs() < 1e-12);
}

#[test]
fn dbscan_acceptance_metrics_recovers_non_convex_concentric_rings_where_kmeans_cannot() {
    let (rings, truth) = concentric_rings_with_noise(120, 8);
    let dbscan_model = SKDBSCAN::<f64>::build()
        .eps(0.35)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(&rings)
        .unwrap();

    assert_eq!(dbscan_model.number_of_clusters(), 2);
    let dbscan_i64 = Array1::from_iter(dbscan_model.labels().iter().map(|label| label.to_i64()));
    let ari_dbscan = sk_adjusted_rand_score(truth.view(), dbscan_i64.view()).unwrap();
    let v_measure_dbscan = sk_v_measure_score(truth.view(), dbscan_i64.view()).unwrap();
    assert!((ari_dbscan - 1.0).abs() < 1e-12);
    assert!((v_measure_dbscan - 1.0).abs() < 1e-12);

    let kmeans_model = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(0)
        .build()
        .unwrap()
        .fit(&rings)
        .unwrap();
    let kmeans_i64 = Array1::from_iter(kmeans_model.labels().iter().map(|label| label.to_i64()));
    let ari_kmeans = sk_adjusted_rand_score(truth.view(), kmeans_i64.view()).unwrap();
    assert!(ari_kmeans < 0.25);
}
