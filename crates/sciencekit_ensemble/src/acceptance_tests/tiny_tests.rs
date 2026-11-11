//! Tiny-data acceptance tests: `f32`/`f64`, single-class bootstrap resilience,
//! OOB on/off and scorer adapters (task 4.3).

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKLabelScorer, SKRegressorPredictor, SKSupervisedFit,
    SKSupervisedScorer,
};
use sciencekit_metrics::scorers::{SKAccuracyScorer, SKR2Scorer};

use crate::{SKRandomForestClassifier, SKRandomForestRegressor};

/// A 4-row dataset with 50 trees inevitably produces single-class bootstrap
/// draws (~12.5% per tree), yet fits, predicts and scores cleanly in `f32` and
/// `f64`.
#[test]
fn tiny_classification_and_regression_fit_and_score_cleanly() {
    let features_f64 = array![[0.0, 0.1], [0.1, 0.2], [5.0, 5.1], [5.1, 5.2]];
    let labels = array![0_i64, 0, 1, 1];

    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(50)
        .oob_score(true)
        .random_seed(7)
        .build()
        .expect("builds")
        .fit(&features_f64, labels.view())
        .expect("fit succeeds even when individual bootstrap draws hit one class");
    let accuracy = SKAccuracyScorer
        .score(
            &classifier,
            features_f64.view().try_into().expect("data view"),
            labels.view().try_into().expect("target view"),
        )
        .expect("scorer succeeds");
    assert!((accuracy - 1.0).abs() < 1e-12);
    assert!(classifier.oob_scored_rows() > 0);
    assert!(classifier.oob_score().expect("oob") >= 0.0);

    let features_f32 = array![
        [0.0_f32, 0.1],
        [0.2, 0.3],
        [10.0, 10.1],
        [10.2, 10.3],
        [20.0, 20.1],
        [20.2, 20.3]
    ];
    let responses_f32 = array![1.0_f32, 1.2, 10.0, 10.2, 20.0, 20.2];
    let regressor = SKRandomForestRegressor::<f32>::build()
        .number_of_trees(30)
        .oob_score(true)
        .random_seed(13)
        .build()
        .expect("builds")
        .fit(&features_f32, responses_f32.view())
        .expect("f32 regressor fit");
    let predictions = regressor.predict(&features_f32).expect("f32 predict");
    assert_eq!(predictions.len(), 6);
    let r2 = SKR2Scorer
        .score(
            &regressor,
            features_f32.view().try_into().expect("data view"),
            responses_f32.view().try_into().expect("target view"),
        )
        .expect("r2 scorer");
    assert!(r2 > 0.85, "tiny f32 R² {r2}");
    let labels_f32 = array![0_i64, 0, 1, 1, 2, 2];
    let cls_f32 = SKRandomForestClassifier::<f32>::build()
        .number_of_trees(20)
        .random_seed(13)
        .build()
        .expect("builds")
        .fit(&features_f32, labels_f32.view())
        .expect("f32 cls fit");
    assert_eq!(cls_f32.predict_labels(&features_f32).expect("pred").len(), 6);
}
