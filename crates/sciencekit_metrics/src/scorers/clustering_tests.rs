//! Tests for the unsupervised silhouette scorer (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKClusterLabel, SKClustererPredictor, SKDataView, SKError, SKFloat, SKUnsupervisedScorer,
};

use super::SKSilhouetteScorer;
use crate::clustering::sk_silhouette_score;

/// A fitted model that returns a fixed assignment vector.
struct FixedClusterer {
    assignments: Vec<SKClusterLabel>,
}

impl<F: SKFloat> SKClustererPredictor<F> for FixedClusterer {
    type Error = SKError;

    fn predict_clusters<'a, X>(&self, _features: X) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array1::from_vec(self.assignments.clone()))
    }
}

fn two_compact_clusters() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ]
}

fn two_cluster_assignments() -> Vec<SKClusterLabel> {
    vec![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ]
}

#[test]
fn silhouette_scorer_equals_the_free_function() {
    let features = two_compact_clusters();
    let assignments = Array1::from_vec(two_cluster_assignments());
    let scorer = SKSilhouetteScorer;

    let from_assignments =
        <SKSilhouetteScorer as SKUnsupervisedScorer<f64, FixedClusterer>>::score_from_assignments(
            &scorer,
            SKDataView::Dense(features.view()),
            assignments.view(),
        )
        .unwrap();
    let expected =
        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!((from_assignments - expected).abs() < 1e-12);

    let model = FixedClusterer {
        assignments: two_cluster_assignments(),
    };
    let from_model = <SKSilhouetteScorer as SKUnsupervisedScorer<f64, FixedClusterer>>::score(
        &scorer,
        &model,
        SKDataView::Dense(features.view()),
    )
    .unwrap();
    assert!((from_model - expected).abs() < 1e-12);
}

#[test]
fn silhouette_scorer_filters_noise_like_the_free_function() {
    let features = two_compact_clusters();
    let mut assignments = two_cluster_assignments();
    assignments[1] = SKClusterLabel::NOISE;
    let from_assignments =
        <SKSilhouetteScorer as SKUnsupervisedScorer<f64, FixedClusterer>>::score_from_assignments(
            &SKSilhouetteScorer,
            SKDataView::Dense(features.view()),
            Array1::from_vec(assignments.clone()).view(),
        )
        .unwrap();
    let expected = sk_silhouette_score(
        SKDataView::Dense(features.view()),
        Array1::from_vec(assignments).view(),
    )
    .unwrap();
    assert!((from_assignments - expected).abs() < 1e-12);
}
