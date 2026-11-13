//! Repeated-fit determinism, `Send + Sync` bounds and rayon thread-count
//! invariance tests (task 6.4).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_tree::SKMaximumFeatures;

use crate::boosting::{
    SKAbsoluteErrorLoss, SKBinaryLogLoss, SKHuberLoss, SKMulticlassLogLoss, SKSquaredErrorLoss,
};
use crate::gradient_boosting::{
    SKGradientBoostingClassifier, SKGradientBoostingClassifierBuilder,
    SKGradientBoostingClassifierModel, SKGradientBoostingRegressionLoss,
    SKGradientBoostingRegressor, SKGradientBoostingRegressorBuilder,
    SKGradientBoostingRegressorModel,
};

/// Builders, estimators, fitted models and loss types satisfy `Send + Sync`.
#[test]
fn gradient_boosting_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKGradientBoostingClassifierBuilder<f64>>();
    assert_send_sync::<SKGradientBoostingRegressorBuilder<f64>>();
    assert_send_sync::<SKGradientBoostingClassifier<f64>>();
    assert_send_sync::<SKGradientBoostingRegressor<f64>>();
    assert_send_sync::<SKGradientBoostingClassifierModel<f64>>();
    assert_send_sync::<SKGradientBoostingRegressorModel<f64>>();
    assert_send_sync::<SKGradientBoostingClassifierModel<f32>>();
    assert_send_sync::<SKGradientBoostingRegressorModel<f32>>();
    assert_send_sync::<SKSquaredErrorLoss>();
    assert_send_sync::<SKAbsoluteErrorLoss>();
    assert_send_sync::<SKHuberLoss<f64>>();
    assert_send_sync::<SKBinaryLogLoss>();
    assert_send_sync::<SKMulticlassLogLoss>();
}

/// Scenario: Repeated fits agree and rayon thread counts (1 vs 4) produce
/// identical stage trees, leaf values, feature importances and predictions.
#[test]
fn repeated_fits_and_rayon_thread_counts_produce_identical_models() {
    let rows = 180usize;
    let features = Array2::from_shape_fn((rows, 6), |(row, column)| {
        let base = ((row * 17 + column * 23) % 41) as f64 / 10.0 - 2.0;
        if (row + column) % 3 == 0 { 0.0 } else { base }
    });
    let labels = Array1::from_shape_fn(rows, |row| {
        let score = features[[row, 0]] - features[[row, 1]] + 0.5 * features[[row, 2]];
        if score < -0.5 {
            0_i64
        } else {
            i64::from(score >= 0.7) + 1
        }
    });
    let responses = Array1::from_shape_fn(rows, |row| {
        2.5 * features[[row, 0]] - 1.5 * features[[row, 1]] + features[[row, 3]]
    });

    let classifier = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(20)
        .learning_rate(0.15)
        .maximum_depth(3)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .subsample(0.75)
        .n_iter_no_change(4)
        .random_seed(2026)
        .build()
        .expect("classifier builds");
    let regressor = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(20)
        .learning_rate(0.15)
        .loss(SKGradientBoostingRegressionLoss::Huber)
        .maximum_depth(3)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .subsample(0.75)
        .n_iter_no_change(4)
        .random_seed(2026)
        .build()
        .expect("regressor builds");

    let run_on_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                let fitted_c = classifier.fit(&features, labels.view()).expect("fit c");
                let fitted_r = regressor.fit(&features, responses.view()).expect("fit r");
                let probs = fitted_c.predict_probabilities(&features).expect("probs");
                let values = fitted_r.predict(&features).expect("values");
                (fitted_c, fitted_r, probs, values)
            })
    };

    let (c_one, r_one, probs_one, values_one) = run_on_pool(1);
    let (c_repeat, r_repeat, probs_repeat, values_repeat) = run_on_pool(1);
    let (c_four, r_four, probs_four, values_four) = run_on_pool(4);

    assert_eq!(c_one.number_of_stages(), c_repeat.number_of_stages());
    assert_eq!(c_one.number_of_stages(), c_four.number_of_stages());
    (0..c_one.number_of_stages()).for_each(|stage| {
        assert_eq!(c_one.stage_trees(stage), c_repeat.stage_trees(stage));
        assert_eq!(c_one.stage_trees(stage), c_four.stage_trees(stage));
    });
    assert_eq!(r_one.stage_trees(), r_repeat.stage_trees());
    assert_eq!(r_one.stage_trees(), r_four.stage_trees());
    assert_eq!(r_one.feature_importances(), r_four.feature_importances());
    assert_eq!(probs_one, probs_repeat);
    assert_eq!(probs_one, probs_four);
    assert_eq!(values_one, values_repeat);
    assert_eq!(values_one, values_four);
}
