//! Tiny-data acceptance: one sample and one split, every criterion.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::classifier::SKDecisionTreeClassifier;
use crate::criterion::SKTreeCriterion;
use crate::regressor::SKDecisionTreeRegressor;

/// A one-sample regression fit predicts that sample's response exactly.
#[test]
fn one_sample_regression_predicts_the_only_response() {
    let features = array![[1.5, -2.0, 0.0]];
    let responses = array![7.25];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(model.node_count(), 1);
    assert_eq!(
        model.predict(features.view()).expect("predict"),
        array![7.25]
    );
}

/// The smallest possible split separates two rows into two pure leaves.
#[test]
fn one_split_classification_separates_two_rows() {
    let features = array![[0.0], [1.0]];
    let labels = array![0_i64, 1];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(model.node_count(), 3);
    assert_eq!(
        model.predict_labels(features.view()).expect("predict"),
        array![0_i64, 1]
    );
}

/// Both classification criteria fit a tiny data set.
#[test]
fn classification_criteria_fit_tiny_data() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let labels = array![0_i64, 0, 1, 1];
    for criterion in [SKTreeCriterion::Gini, SKTreeCriterion::Entropy] {
        let model = SKDecisionTreeClassifier::<f64>::build()
            .criterion(criterion)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit succeeds");
        assert_eq!(
            model.predict_labels(features.view()).expect("predict"),
            array![0_i64, 0, 1, 1]
        );
    }
}

/// Both regression criteria fit a tiny data set.
#[test]
fn regression_criteria_fit_tiny_data() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let responses = array![0.0, 1.0, 2.0, 3.0];
    for criterion in [
        SKTreeCriterion::SquaredError,
        SKTreeCriterion::AbsoluteError,
    ] {
        let model = SKDecisionTreeRegressor::<f64>::build()
            .criterion(criterion)
            .build()
            .expect("builds")
            .fit(&features, responses.view())
            .expect("fit succeeds");
        assert_eq!(model.predict(features.view()).expect("predict"), responses);
    }
}
