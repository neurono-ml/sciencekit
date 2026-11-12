//! Companion tests for classifier target canonicalization
//! (spec `nearest-neighbor-estimators`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::SKKNeighborsClassifier;

#[test]
fn classifier_targets_canonicalize_nominal_labels_in_first_occurrence_order() {
    let features = array![[1.0_f64], [2.0], [10.0], [11.0]];
    let labels = ["zulu", "alpha", "zulu", "alpha"];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), &labels[..])
        .unwrap();

    assert_eq!(model.classes().number_of_classes(), 2);
    assert_eq!(model.classes().index_of("zulu"), Some(0));
    assert_eq!(model.classes().index_of("alpha"), Some(1));

    let predictions = model.predict_labels(array![[1.0_f64]].view()).unwrap();
    assert_eq!(predictions[0], 0);
    assert_eq!(model.classes().label_of(0), Some("zulu"));
}

#[test]
fn classifier_targets_return_canonical_indices_for_nominal_labels() {
    let features = array![[1.0_f64], [5.0]];
    let labels = ["left", "right"];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(features.view(), &labels[..])
        .unwrap();

    let predictions = model.predict_labels(array![[4.9_f64]].view()).unwrap();

    assert_eq!(predictions[0], 1);
    assert_eq!(model.classes().label_of(1), Some("right"));
}

#[test]
fn classifier_targets_preserve_integer_values_through_the_table() {
    let features = array![[1.0_f64], [2.0], [10.0], [11.0]];
    let labels = array![7_i64, 7, 3, 3];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.classes().index_of("7"), Some(0));
    assert_eq!(model.classes().index_of("3"), Some(1));

    let predictions = model.predict_labels(array![[2.0_f64]].view()).unwrap();
    assert_eq!(predictions[0], 0);
    assert_eq!(model.classes().label_of(predictions[0] as usize), Some("7"));
}

#[test]
fn classifier_targets_follow_first_occurrence_not_a_numeric_sort() {
    let features = array![[1.0_f64], [2.0], [10.0], [11.0]];
    // First occurrence is 9 then -4; a numeric sort would put -4 first.
    let labels = array![9_i64, -4, -4, 9];
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    // Canonical order is first-occurrence: 9 → 0, -4 → 1. A numeric sort would
    // invert it, so the assertion proves the table drives the ordering.
    assert_eq!(
        model.classes().labels().collect::<Vec<_>>(),
        vec!["9", "-4"]
    );
    assert_eq!(model.classes().index_of("9"), Some(0));
}
