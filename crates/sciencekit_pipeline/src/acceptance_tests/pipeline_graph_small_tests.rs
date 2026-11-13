//! Small-data acceptance tests for [`crate::SKPipelineGraph`] covering tiny
//! `f32`/`f64` graphs, linear-equivalent graphs, dense/sparse/mixed
//! representation parity and metric production (spec `pipeline-graph`, Task
//! 3.3).

use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor};
use sciencekit_impute::missing_indicator::{SKMissingFeatureScope, SKMissingIndicator};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::{sk_mean_squared_error, sk_r2_score};
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::fixtures::{tiny_features, tiny_labels, tiny_regression_targets, tiny_sparse_features};
use crate::{SKFeatureUnion, SKPipeline, SKPipelineGraph};

#[test]
fn tiny_diamond_graphs_match_linear_equivalents_across_f64_and_f32_and_produce_metrics() {
    let features = tiny_features();
    let labels = tiny_labels();
    let targets = tiny_regression_targets();

    let mut builder = SKPipelineGraph::start::<f64>();
    let imputer = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let standard = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let merge = builder.add_merge();
    let classifier = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder
        .connect(imputer, standard)
        .connect(imputer, minmax)
        .connect(standard, merge)
        .connect(minmax, merge)
        .connect(merge, classifier);

    let fitted_dag = builder
        .build()
        .unwrap()
        .fit_classifier(&features, labels.view())
        .unwrap();
    let linear_twin = SKPipeline::start::<f64>()
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
        .fit(&features, labels.view())
        .unwrap();

    let dag_labels = fitted_dag.predict_labels(&features).unwrap();
    assert_eq!(dag_labels, linear_twin.predict_labels(&features).unwrap());
    let accuracy: f64 =
        sk_accuracy_score(labels.as_slice().unwrap(), dag_labels.as_slice().unwrap()).unwrap();
    assert!(
        accuracy >= 0.875,
        "expected accuracy >= 0.875, got {accuracy}"
    );

    let features_f32 = features.mapv(|value| value as f32);
    let targets_f32 = targets.mapv(|value| value as f32);
    let mut regressor_builder = SKPipelineGraph::start::<f32>();
    let regressor_left =
        regressor_builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let regressor_right =
        regressor_builder.add_transformer(SKMinMaxScaler::<f32>::build().build().unwrap());
    let regressor_sink =
        regressor_builder.add_regressor(SKLinearRegression::build().build().unwrap());
    regressor_builder
        .connect(regressor_left, regressor_sink)
        .connect(regressor_right, regressor_sink);
    let fitted_regressor = regressor_builder
        .build()
        .unwrap()
        .fit_regressor(&features_f32, targets_f32.view())
        .unwrap();
    let regressor_predictions = fitted_regressor.predict(&features_f32).unwrap();
    let r2_score: f32 = sk_r2_score(targets_f32.view(), regressor_predictions.view()).unwrap();
    let mean_squared_error: f32 =
        sk_mean_squared_error(targets_f32.view(), regressor_predictions.view()).unwrap();
    assert!(
        r2_score > 0.99 && mean_squared_error < 1e-4,
        "expected exact fit, got r2={r2_score}, mse={mean_squared_error}"
    );
}

#[test]
fn tiny_sparse_and_mixed_graphs_preserve_csr_without_densification_and_attribute_rejections() {
    let sparse = tiny_sparse_features();
    let dense = tiny_features();
    let labels = tiny_labels();

    let build_sparse_capable = || {
        let mut builder = SKPipelineGraph::start::<f64>();
        let left = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
        let right = builder.add_transformer(
            SKMissingIndicator::<f64>::build()
                .features(SKMissingFeatureScope::All)
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        );
        let merge = builder.add_merge();
        let sink = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
        builder
            .connect(left, merge)
            .connect(right, merge)
            .connect(merge, sink);
        builder.build().unwrap()
    };

    let sparse_fitted = build_sparse_capable()
        .fit(sparse.view(), labels.view())
        .unwrap();
    sparse_fitted
        .topological_order()
        .iter()
        .for_each(|&node| assert!(sparse_fitted.fitted_node_block(node).unwrap().is_sparse()));
    assert!(
        sparse_fitted
            .transform_to_final(sparse.view())
            .unwrap()
            .is_sparse()
    );

    let mixed_fitted = build_sparse_capable().fit(&dense, labels.view()).unwrap();
    assert_eq!(
        sparse_fitted.predict_labels(sparse.view()).unwrap(),
        mixed_fitted.predict_labels(&dense).unwrap()
    );

    let mut dense_only = SKPipelineGraph::start::<f64>();
    let imputer = dense_only.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let scaler = dense_only.add_transformer(SKStandardScaler::build().build().unwrap());
    let sink = dense_only.add_classifier(SKLogisticRegression::build().build().unwrap());
    dense_only.connect(imputer, scaler).connect(scaler, sink);
    let rejection = dense_only
        .build()
        .unwrap()
        .fit_checked(sparse.view(), labels.view())
        .err()
        .unwrap();
    assert_eq!(rejection.node_identifier(), Some(scaler));
    assert!(rejection.to_string().contains("csr"));
}
