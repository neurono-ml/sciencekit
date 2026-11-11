//! Companion tests for normalized mean forest feature importances (task 3.2).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKSupervisedFit};
use sciencekit_tree::SKMaximumFeatures;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

/// When feature 0 cleanly separates the classes/responses and the remaining
/// features are constant, feature 0 owns the entire normalized importance `1.0`.
#[test]
fn separating_feature_dominates_normalized_forest_importances() {
    let features = array![
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.5, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(16)
        .maximum_features(SKMaximumFeatures::All)
        .random_seed(11)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let cls_importances = classifier.feature_importances();
    assert!((cls_importances[0] - 1.0).abs() < 1e-12);
    assert_eq!(cls_importances[1], 0.0);
    assert_eq!(cls_importances[2], 0.0);
    assert!((cls_importances.sum() - 1.0).abs() < 1e-12);

    let responses = array![1.0, 1.5, 2.0, 10.0, 10.5, 11.0];
    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(16)
        .maximum_features(SKMaximumFeatures::All)
        .random_seed(11)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    let reg_importances = regressor.feature_importances();
    assert!((reg_importances[0] - 1.0).abs() < 1e-12);
    assert_eq!(reg_importances[1], 0.0);
    assert_eq!(reg_importances[2], 0.0);
    assert!((reg_importances.sum() - 1.0).abs() < 1e-12);
}

/// When trees assign different importances to features, the forest importance
/// equals the normalized arithmetic mean of the per-tree importances.
#[test]
fn forest_importances_equal_mean_of_per_tree_importances() {
    let rows = 48usize;
    let features = Array2::from_shape_fn((rows, 4), |(row, column)| {
        ((row * (column + 3) + column * 7) % 19) as f64
    });
    let responses = Array1::from_shape_fn(rows, |row| {
        3.0 * features[[row, 0]] + 1.5 * features[[row, 1]] - 0.5 * features[[row, 2]]
    });

    let model = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(2)
        .maximum_depth(3)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .random_seed(9)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");

    let tree0 = model.trees()[0].feature_importances();
    let tree1 = model.trees()[1].feature_importances();
    assert_ne!(tree0, tree1, "the two trees should differ in importances");

    let forest = model.feature_importances();
    assert!((forest.sum() - 1.0).abs() < 1e-12);
    for feature in 0..4 {
        let expected_mean = 0.5 * (tree0[feature] + tree1[feature]);
        assert!((forest[feature] - expected_mean).abs() < 1e-12);
    }
}

/// A zero-depth forest never splits any node, so every feature importance is zero.
#[test]
fn forest_without_splits_reports_zero_importances() {
    let features = array![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]];
    let labels = array![0_i64, 0, 1, 1];
    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(5)
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(classifier.feature_importances().to_vec(), vec![0.0, 0.0]);

    let responses = array![1.0, 2.0, 3.0, 4.0];
    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(5)
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(regressor.feature_importances().to_vec(), vec![0.0, 0.0]);
}
