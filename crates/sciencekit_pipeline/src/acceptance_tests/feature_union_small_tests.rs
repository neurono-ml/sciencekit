//! Small-data acceptance tests for `SKFeatureUnion` covering tuple/boxed
//! construction, dense/sparse/mixed representations, nested pipelines and
//! metric production (spec `feature-union`, Task 3.2).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKFeatureBlock, SKRegressorPredictor};
use sciencekit_impute::missing_indicator::{SKMissingFeatureScope, SKMissingIndicator};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::robust_scaler::core_implementation::SKRobustScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::fixtures::{tiny_features, tiny_labels, tiny_regression_targets, tiny_sparse_features};
use crate::{SKBlockTransformer, SKConfiguredStages, SKFeatureUnion, SKPipeline};

#[test]
fn tiny_tuple_and_boxed_feature_unions_produce_matching_classifier_metrics() {
    let features = tiny_features();
    let labels = tiny_labels();

    let tuple_union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
            SKRobustScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();

    let boxed_members: Vec<Box<dyn SKBlockTransformer<f64>>> = vec![
        Box::new(SKStandardScaler::build().build().unwrap()),
        Box::new(SKMinMaxScaler::<f64>::build().build().unwrap()),
        Box::new(SKRobustScaler::<f64>::build().build().unwrap()),
    ];
    let boxed_union = SKFeatureUnion::build()
        .stages(boxed_members)
        .build()
        .unwrap();

    let tuple_fitted = SKPipeline::start::<f64>()
        .then(tuple_union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();
    let boxed_fitted = SKPipeline::start::<f64>()
        .then(boxed_union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();

    assert_eq!(tuple_fitted.number_of_features(), 6);
    assert_eq!(boxed_fitted.number_of_features(), 6);

    let tuple_predictions = tuple_fitted.predict_labels(&features).unwrap();
    let boxed_predictions = boxed_fitted.predict_labels(&features).unwrap();
    assert_eq!(tuple_predictions, boxed_predictions);

    let accuracy: f64 = sk_accuracy_score(
        labels.as_slice().unwrap(),
        tuple_predictions.as_slice().unwrap(),
    )
    .unwrap();
    assert!(accuracy >= 0.875, "expected high accuracy, got {accuracy}");
}

#[test]
fn tiny_sparse_and_mixed_feature_union_pipelines_preserve_representation_parity() {
    let sparse = tiny_sparse_features();
    let dense = tiny_features();
    let labels = tiny_labels();

    let sparse_union = SKFeatureUnion::build()
        .stages((
            SKSimpleImputer::<f64>::build().build().unwrap(),
            SKMissingIndicator::<f64>::build()
                .features(SKMissingFeatureScope::All)
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        ))
        .build()
        .unwrap();
    let sparse_pipeline = SKPipeline::start::<f64>()
        .then(sparse_union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap();

    let intermediate = sparse_pipeline
        .stages()
        .transform_stages(
            sparse_pipeline.fitted_stages(),
            SKFeatureBlock::Sparse(sparse.clone()),
        )
        .unwrap();
    assert!(intermediate.is_sparse());
    assert_eq!(intermediate.number_of_columns(), 4);

    let mixed_union = SKFeatureUnion::build()
        .stages((
            SKSimpleImputer::<f64>::build().build().unwrap(),
            SKMissingIndicator::<f64>::build()
                .features(SKMissingFeatureScope::All)
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        ))
        .build()
        .unwrap();
    let mixed_pipeline = SKPipeline::start::<f64>()
        .then(mixed_union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&dense, labels.view())
        .unwrap();

    assert_eq!(
        sparse_pipeline.predict_labels(sparse.view()).unwrap(),
        mixed_pipeline.predict_labels(&dense).unwrap()
    );
}

#[test]
fn nested_feature_unions_and_sub_pipelines_compose_into_regressor_metrics() {
    let features = array![
        [1.0_f64, f64::NAN],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, f64::NAN],
        [6.0, 12.0],
        [7.0, 14.0],
        [8.0, 16.0],
    ];
    let targets = tiny_regression_targets();

    let sub_pipeline = SKPipeline::start::<f64>()
        .then(SKSimpleImputer::<f64>::build().build().unwrap())
        .then(SKStandardScaler::build().build().unwrap());
    let inner_union = SKFeatureUnion::build()
        .stages((
            sub_pipeline,
            SKMissingIndicator::<f64>::build()
                .error_on_new(false)
                .build()
                .unwrap(),
        ))
        .build()
        .unwrap();
    let outer_union = SKFeatureUnion::build()
        .stages((
            inner_union,
            SKSimpleImputer::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();

    let fitted = SKPipeline::start::<f64>()
        .then(outer_union)
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, targets.view())
        .unwrap();

    assert_eq!(fitted.number_of_features(), 5);
    let predictions = fitted.predict(&features).unwrap();
    let r2: f64 = sk_r2_score(targets.view(), predictions.view()).unwrap();
    assert!(r2 > 0.99, "expected near-exact linear R2, got {r2}");
}
