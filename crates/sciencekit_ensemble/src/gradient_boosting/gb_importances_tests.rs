//! Companion tests for aggregated stage feature importances and normalization
//! (task 6.1).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKSupervisedFit};
use sciencekit_tree::SKMaximumFeatures;

use super::importances::sk_aggregate_weighted_importances;
use super::{SKGradientBoostingClassifier, SKGradientBoostingRegressor};

/// Scenario: Importances aggregate the stages weighted by sample count and sum
/// to one when at least one split exists.
#[test]
fn two_stages_with_different_importances_produce_sample_weighted_aggregate() {
    let features = Array2::from_shape_fn((20, 2), |(row, column)| {
        if column == 0 {
            if row < 10 { 0.0 } else { 10.0 }
        } else if row % 2 == 0 {
            0.0
        } else {
            5.0
        }
    });
    let responses = Array1::from_shape_fn(20, |row| {
        let first = if row < 10 { -4.0 } else { 4.0 };
        let second = if row % 2 == 0 { -1.5 } else { 1.5 };
        first + second
    });

    // With learning_rate = 1.0 and depth = 1, stage 0 splits purely on feature 0
    // (removing the dominant +-4 step) and stage 1 splits purely on feature 1
    // (removing the remaining +-1.5 step).
    let model = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(2)
        .maximum_depth(1)
        .learning_rate(1.0)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fits");

    let stage_zero = model.stage_feature_importances(0).expect("stage 0");
    let stage_one = model.stage_feature_importances(1).expect("stage 1");
    assert!((stage_zero[0] - 1.0).abs() < 1e-12);
    assert!((stage_one[1] - 1.0).abs() < 1e-12);

    let count_zero = model
        .stage_tree(0)
        .and_then(|tree| tree.node(0))
        .expect("root 0")
        .sample_count() as f64;
    let count_one = model
        .stage_tree(1)
        .and_then(|tree| tree.node(0))
        .expect("root 1")
        .sample_count() as f64;
    let expected_first =
        (count_zero * stage_zero[0] + count_one * stage_one[0]) / (count_zero + count_one);
    let expected_second =
        (count_zero * stage_zero[1] + count_one * stage_one[1]) / (count_zero + count_one);

    let importances = model.feature_importances();
    assert!((importances[0] - expected_first).abs() < 1e-12);
    assert!((importances[1] - expected_second).abs() < 1e-12);
    assert!((importances.sum() - 1.0).abs() < 1e-12);

    // Unequal sample counts (30 vs 10) weight the stages 3:1.
    let stage_zero_slice = [1.0_f64, 0.0];
    let stage_one_slice = [0.0_f64, 1.0];
    let weighted = sk_aggregate_weighted_importances(
        [
            (30usize, &stage_zero_slice[..]),
            (10usize, &stage_one_slice[..]),
        ],
        2,
    );
    assert!((weighted[0] - 0.75).abs() < 1e-12);
    assert!((weighted[1] - 0.25).abs() < 1e-12);
}

/// Informative features dominate across binary and multiclass classifiers, and
/// zero-split depth-0 boosters report all-zero importances without NaNs.
#[test]
fn classifier_importances_normalize_and_zero_split_boosters_return_zeros() {
    let features = Array2::from_shape_fn((36, 4), |(row, column)| match column {
        0 => (row / 12) as f64 * 10.0,
        1 => ((row * 7) % 3) as f64 * 0.01,
        2 => 1.0,
        _ => 0.0,
    });
    let labels = Array1::from_shape_fn(36, |row| (row / 12) as i64);

    let classifier = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(12)
        .maximum_depth(2)
        .maximum_features(SKMaximumFeatures::All)
        .random_seed(5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fits");

    let importances = classifier.feature_importances();
    assert_eq!(importances.len(), 4);
    assert!((importances.sum() - 1.0).abs() < 1e-12);
    assert!(
        importances[0] > 0.85,
        "informative feature should dominate: {importances:?}"
    );
    assert_eq!(importances[2], 0.0);
    assert_eq!(importances[3], 0.0);
    assert!(classifier.stage_feature_importances(0, 0).is_some());

    let no_splits_regressor = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(5)
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&array![[0.0, 1.0], [2.0, 3.0]], array![1.0, 5.0].view())
        .expect("fits");
    assert_eq!(
        no_splits_regressor.feature_importances().to_vec(),
        vec![0.0, 0.0]
    );

    let no_splits_classifier = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(4)
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&array![[0.0, 1.0], [2.0, 3.0]], array![0_i64, 1].view())
        .expect("fits");
    assert_eq!(
        no_splits_classifier.feature_importances().to_vec(),
        vec![0.0, 0.0]
    );
}
