//! Companion tests for classifier label prediction
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::SKKNeighborsClassifier;
use crate::neighbors::SKNeighborsWeights;

fn fit_fixture(
    features: Array2<f64>,
    labels: Array1<i64>,
) -> super::SKKNeighborsClassifierModel<f64> {
    SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap()
}

/// Map a canonical index back to the label text the fixture used.
fn label_text(model: &super::SKKNeighborsClassifierModel<f64>, canonical: i64) -> String {
    model
        .classes()
        .label_of(canonical as usize)
        .expect("predictions stay inside the class table")
        .to_string()
}

#[test]
fn classifier_predict_majority_vote_wins() {
    let features = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let labels = array![0_i64, 0, 0, 1, 1];
    let model = fit_fixture(features, labels);
    let queries = array![[2.6_f64]];

    let predictions = model.predict_labels(queries.view()).unwrap();

    assert_eq!(predictions.len(), 1);
    assert_eq!(label_text(&model, predictions[0]), "0");
}

#[test]
fn classifier_predict_uses_only_the_nearest_neighbors() {
    let features = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let labels = array![0_i64, 1, 0, 0, 1];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[1.0_f64]];

    let predictions = model.predict_labels(queries.view()).unwrap();

    assert_eq!(label_text(&model, predictions[0]), "0");
}

#[test]
fn classifier_predict_distance_weighting_changes_the_vote() {
    let features = array![[1.0_f64], [10.0], [11.0], [12.0], [13.0]];
    let labels = array![1_i64, 0, 0, 0, 0];
    let queries = array![[1.0_f64]];

    let uniform = fit_fixture(features.clone(), labels.clone());
    let uniform_vote = uniform.predict_labels(queries.view()).unwrap();
    assert_eq!(label_text(&uniform, uniform_vote[0]), "0");

    let weighted = SKKNeighborsClassifier::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let weighted_vote = weighted.predict_labels(queries.view()).unwrap();
    assert_eq!(label_text(&weighted, weighted_vote[0]), "1");
}

#[test]
fn classifier_predict_breaks_a_vote_tie_by_the_smallest_canonical_index() {
    let features = array![[1.0_f64], [2.0], [3.0], [4.0]];
    let labels = array![2_i64, 2, 1, 1];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(4)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[2.5_f64]];

    let predictions = model.predict_labels(queries.view()).unwrap();

    // Label 2 is the first seen, so it owns canonical index 0 and wins the tie.
    assert_eq!(model.classes().index_of("2"), Some(0));
    assert_eq!(predictions[0], 0);
    assert_eq!(label_text(&model, predictions[0]), "2");
}

#[test]
fn classifier_predict_returns_one_label_per_query_row() {
    let features = array![[1.0_f64], [2.0], [40.0], [41.0]];
    let labels = array![0_i64, 0, 1, 1];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[1.5_f64], [40.5]];

    let predictions = model.predict_labels(queries.view()).unwrap();

    assert_eq!(predictions.len(), 2);
    assert_eq!(label_text(&model, predictions[0]), "0");
    assert_eq!(label_text(&model, predictions[1]), "1");
}

/// Inverse-distance weighting lets two very close neighbours outvote three far
/// ones — the scenario as written in the spec, with no zero distance involved.
#[test]
fn classifier_predict_distance_weighting_outvotes_a_far_majority() {
    let features = array![[1.0_f64], [2.0], [100.0], [101.0], [102.0]];
    let labels = array![0_i64, 0, 1, 1, 1];
    let queries = array![[0.0_f64]];

    let uniform = fit_fixture(features.clone(), labels.clone());
    assert_eq!(
        label_text(&uniform, uniform.predict_labels(queries.view()).unwrap()[0]),
        "1"
    );

    let weighted = SKKNeighborsClassifier::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(
        label_text(
            &weighted,
            weighted.predict_labels(queries.view()).unwrap()[0]
        ),
        "0"
    );
}
