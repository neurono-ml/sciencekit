//! Prediction through fitted [`super::SKPipelineGraphModel`] and the
//! [`super::SKPipelineGraphClassifier`] / [`super::SKPipelineGraphRegressor`]
//! wrappers (spec `pipeline-graph`, Task 2.3).

use ndarray::{Array1, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::{SKFeatureUnion, SKPipeline};

use super::{SKPipelineGraph, SKPipelineGraphClassifier, SKPipelineGraphRegressor};

#[test]
fn classifier_wrapper_predictions_and_probabilities_match_linear_equivalent_on_unseen_rows() {
    let training_features = array![
        [0.0_f64, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [10.0, 11.0],
        [11.0, 12.0],
        [12.0, 13.0],
        [13.0, 14.0],
    ];
    let training_labels = Array1::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);
    // Unseen single-row and multi-row queries whose statistics differ completely
    // from training_features, proving predict reuses the fitted transformers.
    let unseen_queries = array![[0.5_f64, 1.5], [12.5, 13.5]];

    let mut builder = SKPipelineGraph::start::<f64>();
    let imputer = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let standard = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let final_classifier = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder
        .connect(imputer, standard)
        .connect(imputer, minmax)
        .connect(standard, final_classifier)
        .connect(minmax, final_classifier);

    let graph = builder.build().unwrap();
    let classifier_wrapper: SKPipelineGraphClassifier<f64> = graph
        .fit_classifier(&training_features, training_labels.view())
        .unwrap();

    let linear = SKPipeline::start::<f64>()
        .then(SKSimpleImputer::<f64>::build().build().unwrap())
        .then(
            SKFeatureUnion::build()
                .stages((
                    SKStandardScaler::build().build().unwrap(),
                    SKMinMaxScaler::<f64>::build().build().unwrap(),
                ))
                .build()
                .unwrap(),
        )
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&training_features, training_labels.view())
        .unwrap();

    let graph_labels = classifier_wrapper.predict_labels(&unseen_queries).unwrap();
    let linear_labels = linear.predict_labels(&unseen_queries).unwrap();
    assert_eq!(graph_labels, linear_labels);
    assert_eq!(graph_labels, array![0_i64, 1]);

    let graph_probabilities = classifier_wrapper
        .predict_probabilities(&unseen_queries)
        .unwrap();
    let linear_probabilities = linear.predict_probabilities(&unseen_queries).unwrap();
    graph_probabilities
        .iter()
        .zip(linear_probabilities.iter())
        .for_each(|(&found, &expected)| {
            assert!((found - expected).abs() < 1e-12);
        });
}

#[test]
fn regressor_wrapper_predictions_match_linear_equivalent_on_unseen_rows() {
    let training_features = array![
        [1.0_f64, 2.0],
        [2.0, 5.0],
        [3.0, 7.0],
        [4.0, 11.0],
        [5.0, 14.0],
        [6.0, 18.0],
    ];
    let training_targets = array![7.0_f64, 16.0, 23.0, 35.0, 44.0, 56.0];
    let unseen_queries = array![[2.5_f64, 6.0], [5.5, 16.0]];

    let mut builder = SKPipelineGraph::start::<f64>();
    let standard = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let final_regressor = builder.add_regressor(SKLinearRegression::build().build().unwrap());
    builder
        .connect(standard, final_regressor)
        .connect(minmax, final_regressor);

    let regressor_wrapper: SKPipelineGraphRegressor<f64> = builder
        .build()
        .unwrap()
        .fit_regressor(&training_features, training_targets.view())
        .unwrap();

    let linear = SKPipeline::start::<f64>()
        .then(
            SKFeatureUnion::build()
                .stages((
                    SKStandardScaler::build().build().unwrap(),
                    SKMinMaxScaler::<f64>::build().build().unwrap(),
                ))
                .build()
                .unwrap(),
        )
        .finalize_regressor(SKLinearRegression::build().build().unwrap())
        .unwrap()
        .fit(&training_features, training_targets.view())
        .unwrap();

    let graph_responses = regressor_wrapper.predict(&unseen_queries).unwrap();
    let linear_responses = linear.predict(&unseen_queries).unwrap();
    graph_responses
        .iter()
        .zip(linear_responses.iter())
        .for_each(|(&found, &expected)| {
            assert!((found - expected).abs() < 1e-10);
        });
}

#[test]
fn wrapper_role_mismatch_and_query_width_mismatch_surface_structured_errors() {
    let features = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let labels = Array1::from_vec(vec![0_i64, 0, 1, 1]);

    let mut builder = SKPipelineGraph::start::<f64>();
    let scaler = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let classifier = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder.connect(scaler, classifier);

    let fitted = builder
        .build()
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();

    assert!(matches!(
        fitted.predict(&features),
        Err(SKError::UnsupportedOperation {
            operation: "predict",
            ..
        })
    ));

    let wrong_width = array![[1.0_f64, 2.0, 3.0]];
    assert!(matches!(
        fitted.predict_labels(&wrong_width),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        fitted.into_regressor(),
        Err(SKError::UnsupportedOperation {
            operation: "into_regressor",
            ..
        })
    ));
}
