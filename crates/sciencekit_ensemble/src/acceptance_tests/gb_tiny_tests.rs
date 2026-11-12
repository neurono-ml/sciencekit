//! Tiny-data acceptance tests for gradient boosting (`f32`/`f64`, binary and
//! multiclass, all regression losses, subsampling, early stopping, dense/CSR
//! and scorer adapters, task 7.1).

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKLabelScorer, SKRegressorPredictor, SKSupervisedFit,
    SKSupervisedScorer,
};
use sciencekit_metrics::scorers::{SKAccuracyScorer, SKR2Scorer};
use sprs::CsMat;

use crate::{
    SKGradientBoostingClassifier, SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor,
};

/// Tiny binary/multiclass classification and all three regression losses fit,
/// predict and score cleanly in `f32` and `f64` across dense and CSR inputs.
#[test]
fn tiny_gradient_boosting_fits_and_scores_in_f32_and_f64() {
    let dense_f64 = array![
        [0.0, 0.2],
        [0.0, 0.4],
        [0.2, 0.0],
        [5.0, 0.0],
        [5.2, 0.3],
        [5.4, 0.0],
    ];
    let sparse_f64 = CsMat::new(
        (6, 2),
        vec![0, 1, 2, 3, 4, 6, 7],
        vec![1, 1, 0, 0, 0, 1, 0],
        vec![0.2, 0.4, 0.2, 5.0, 5.2, 0.3, 5.4],
    );
    let binary_labels = array![0_i64, 0, 0, 1, 1, 1];

    let binary_model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(25)
        .learning_rate(0.25)
        .maximum_depth(2)
        .subsample(0.85)
        .n_iter_no_change(4)
        .validation_fraction(0.3)
        .random_seed(7)
        .build()
        .expect("builds")
        .fit(sparse_f64.view(), binary_labels.view())
        .expect("tiny sparse binary fit");
    let accuracy = SKAccuracyScorer
        .score(
            &binary_model,
            dense_f64.view().try_into().expect("data view"),
            binary_labels.view().try_into().expect("target view"),
        )
        .expect("accuracy scorer");
    assert!((accuracy - 1.0).abs() < 1e-12);

    let features_f32 = array![
        [0.0_f32, 0.1],
        [0.2, 0.0],
        [5.0, 5.1],
        [5.2, 5.0],
        [10.0, 0.1],
        [10.2, 0.0],
    ];
    let multiclass_labels = array![0_i64, 0, 1, 1, 2, 2];
    let multiclass_f32 = SKGradientBoostingClassifier::<f32>::build()
        .number_of_estimators(20)
        .learning_rate(0.3)
        .maximum_depth(2)
        .random_seed(11)
        .build()
        .expect("builds")
        .fit(&features_f32, multiclass_labels.view())
        .expect("f32 multiclass fit");
    assert_eq!(
        multiclass_f32
            .predict_labels(&features_f32)
            .expect("predict"),
        multiclass_labels
    );

    let responses_f32 = array![1.0_f32, 1.2, 8.0, 8.2, 16.0, 16.4];
    for loss in [
        SKGradientBoostingRegressionLoss::SquaredError,
        SKGradientBoostingRegressionLoss::AbsoluteError,
        SKGradientBoostingRegressionLoss::Huber,
    ] {
        let regressor = SKGradientBoostingRegressor::<f32>::build()
            .number_of_estimators(45)
            .learning_rate(0.3)
            .loss(loss)
            .huber_delta(2.0)
            .maximum_depth(2)
            .subsample(0.85)
            .random_seed(19)
            .build()
            .expect("builds")
            .fit(&features_f32, responses_f32.view())
            .expect("f32 regression fit");
        let predictions = regressor.predict(&features_f32).expect("f32 predict");
        assert_eq!(predictions.len(), 6);
        let r2 = SKR2Scorer
            .score(
                &regressor,
                features_f32.view().try_into().expect("data view"),
                responses_f32.view().try_into().expect("target view"),
            )
            .expect("r2 scorer");
        assert!(r2 > 0.85, "loss {loss:?} tiny f32 R² = {r2}");
    }
}
