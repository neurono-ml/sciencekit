//! Feature-importance tests (task 6.1).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use crate::classifier::SKDecisionTreeClassifier;
use crate::regressor::SKDecisionTreeRegressor;

/// On data where feature 0 separates the classes and feature 1 carries no
/// information, feature 0 owns the whole normalized importance.
#[test]
fn the_separating_feature_dominates_the_importances() {
    let features = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],];
    let labels = array![0_i64, 0, 1, 1];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let importances = model.feature_importances();
    assert!((importances[0] - 1.0).abs() < 1e-12);
    assert_eq!(importances[1], 0.0);
    assert!((importances.sum() - 1.0).abs() < 1e-12);
    assert_eq!(importances.len(), 2);
}

/// A tree that splits on both features normalizes both contributions to one.
#[test]
fn every_used_feature_contributes_and_the_total_is_one() {
    // Feature 0 isolates a mixed block that feature 1 then splits.
    let features = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    let labels = array![0_i64, 1, 0, 1, 0, 0, 0, 0];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let importances = model.feature_importances();
    assert!(importances[0] > 0.0);
    assert!(importances[1] > 0.0);
    assert!((importances.sum() - 1.0).abs() < 1e-12);
}

/// A single-leaf tree never splits, so every importance is zero.
#[test]
fn a_tree_without_splits_has_zero_importances() {
    let features = array![[0.0], [1.0]];
    let labels = array![0_i64, 1];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(model.feature_importances().to_vec(), vec![0.0]);
}

/// The regressor attributes importance from the impurity decrease as well.
#[test]
fn regressor_importances_follow_the_response() {
    let features = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    let responses = array![0.0, 1.0, 2.0, 3.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    let importances = model.feature_importances();
    assert!((importances[0] - 1.0).abs() < 1e-12);
    assert_eq!(importances[1], 0.0);
    assert!((importances.sum() - 1.0).abs() < 1e-12);
}
