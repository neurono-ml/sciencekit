//! Companion tests for out-of-bag accuracy and R² estimation (task 3.1).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

/// When `oob_score` is false, the fitted model refuses to fabricate a score and
/// stores no OOB row masks.
#[test]
fn disabled_oob_score_reports_unsupported_operation() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let labels = array![0_i64, 0, 1, 1];
    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(8)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(classifier.oob_scored_rows(), 0);
    assert!(classifier.out_of_bag_rows(0).is_none());
    assert!(matches!(
        classifier.oob_score(),
        Err(SKError::UnsupportedOperation {
            operation: "oob_score",
            ..
        })
    ));

    let responses = array![1.0, 2.0, 3.0, 4.0];
    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(8)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(regressor.oob_scored_rows(), 0);
    assert!(regressor.out_of_bag_rows(0).is_none());
    assert!(matches!(
        regressor.oob_score(),
        Err(SKError::UnsupportedOperation {
            operation: "oob_score",
            ..
        })
    ));
}

/// When `oob_score` is enabled, but every row was drawn into every tree's
/// bootstrap sample, the model exposes `0` scored rows and returns
/// `UnsupportedOperation`.
#[test]
fn no_scored_oob_rows_reports_unsupported_operation() {
    let single_feature = array![[1.0]];
    let single_response = array![5.0];
    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(1)
        .oob_score(true)
        .random_seed(0)
        .build()
        .expect("builds")
        .fit(&single_feature, single_response.view())
        .expect("fit succeeds");
    assert_eq!(regressor.oob_scored_rows(), 0);
    assert_eq!(regressor.out_of_bag_rows(0), Some(&[][..]));
    assert!(matches!(
        regressor.oob_score(),
        Err(SKError::UnsupportedOperation {
            operation: "oob_score",
            ..
        })
    ));

    let pair_features = array![[0.0], [1.0]];
    let pair_labels = array![0_i64, 1];
    let classifier = (0_u64..64)
        .map(|seed| {
            SKRandomForestClassifier::<f64>::build()
                .number_of_trees(1)
                .oob_score(true)
                .random_seed(seed)
                .build()
                .expect("builds")
                .fit(&pair_features, pair_labels.view())
                .expect("fit succeeds")
        })
        .find(|model| model.oob_scored_rows() == 0)
        .expect("a 2-row bootstrap draw covers both rows for ~50% of seeds");
    assert!(matches!(
        classifier.oob_score(),
        Err(SKError::UnsupportedOperation {
            operation: "oob_score",
            ..
        })
    ));
}

/// Every row covered by OOB trees is scored exclusively by trees that excluded
/// it — never leaking in-bag memorization — for both classification and
/// regression.
#[test]
fn oob_score_covers_rows_and_excludes_in_bag_trees() {
    let rows = 40usize;
    let noisy_features = Array2::from_shape_fn((rows, 1), |(row, _)| row as f64);
    let alternating_labels = Array1::from_shape_fn(rows, |row| (row % 2) as i64);
    let memorized = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(40)
        .oob_score(true)
        .random_seed(19)
        .build()
        .expect("builds")
        .fit(&noisy_features, alternating_labels.view())
        .expect("fit succeeds");

    assert_eq!(memorized.oob_scored_rows(), rows);
    let in_bag_predictions = memorized
        .predict_labels(&noisy_features)
        .expect("in-bag predict");
    let in_bag_accuracy: f64 = sk_accuracy_score(
        alternating_labels.as_slice().expect("slice"),
        in_bag_predictions.as_slice().expect("slice"),
    )
    .expect("accuracy");
    let oob_accuracy = memorized.oob_score().expect("oob score");
    assert!(in_bag_accuracy > 0.95, "in-bag accuracy {in_bag_accuracy}");
    assert!(
        oob_accuracy < 0.35,
        "OOB score {oob_accuracy} must exclude in-bag trees that memorized the row"
    );

    let signal_features = Array2::from_shape_fn((60, 2), |(row, column)| {
        if column == 0 {
            row as f64 / 10.0
        } else {
            (row % 5) as f64
        }
    });
    let clean_labels = Array1::from_shape_fn(60, |row| i64::from(row >= 30));
    let clean_classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(40)
        .oob_score(true)
        .random_seed(23)
        .build()
        .expect("builds")
        .fit(&signal_features, clean_labels.view())
        .expect("fit succeeds");
    assert_eq!(clean_classifier.oob_scored_rows(), 60);
    assert!(clean_classifier.oob_score().expect("classifier oob") > 0.90);

    let clean_responses = Array1::from_shape_fn(60, |row| if row < 30 { 5.0 } else { 25.0 });
    let clean_regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(40)
        .oob_score(true)
        .random_seed(23)
        .build()
        .expect("builds")
        .fit(&signal_features, clean_responses.view())
        .expect("fit succeeds");
    assert_eq!(clean_regressor.oob_scored_rows(), 60);

    let mut manual_sums = vec![0.0_f64; 60];
    let mut manual_counts = vec![0usize; 60];
    for (tree_index, tree_model) in clean_regressor.trees().iter().enumerate() {
        let tree_predictions = tree_model
            .predict(signal_features.view())
            .expect("tree predict");
        for &row in clean_regressor
            .out_of_bag_rows(tree_index)
            .expect("oob rows")
        {
            manual_sums[row] += tree_predictions[row];
            manual_counts[row] += 1;
        }
    }
    let manual_oob_predictions =
        Array1::from_iter((0..60).map(|row| manual_sums[row] / manual_counts[row] as f64));
    let expected_r2 =
        sk_r2_score(clean_responses.view(), manual_oob_predictions.view()).expect("r2");
    let actual_r2 = clean_regressor.oob_score().expect("regressor oob");
    assert!((actual_r2 - expected_r2).abs() < 1e-12);
    assert!(actual_r2 > 0.85);
}
