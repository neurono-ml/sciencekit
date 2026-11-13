//! Large-data, multi-level topological execution, sparse CSR and multi-threaded
//! concurrency acceptance tests for [`crate::SKPipelineGraph`] (spec
//! `pipeline-graph`, Task 3.3).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor};
use sciencekit_impute::missing_indicator::{SKMissingFeatureScope, SKMissingIndicator};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::robust_scaler::core_implementation::SKRobustScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sprs::{CsMat, TriMat};

use crate::{SKFeatureUnion, SKPipelineGraph};

fn large_separable_dense(rows: usize, columns: usize) -> (Array2<f64>, Array1<i64>, Array1<f64>) {
    let values: Vec<f64> = (0..rows)
        .flat_map(|row| {
            let offset = if row % 2 == 0 { -2.5 } else { 2.5 };
            (0..columns).map(move |column| ((row * 17 + column * 11) % 31) as f64 / 31.0 + offset)
        })
        .collect();
    let features = Array2::from_shape_vec((rows, columns), values).unwrap();
    let labels = Array1::from_vec((0..rows).map(|row| (row % 2) as i64).collect());
    let targets = Array1::from_vec(
        (0..rows)
            .map(|row| 3.0 * features[[row, 0]] - 1.5 * features[[row, 1]] + 4.0)
            .collect(),
    );
    (features, labels, targets)
}

fn large_separable_sparse(rows: usize, columns: usize) -> (CsMat<f64>, Array1<i64>) {
    let mut triplets = TriMat::new((rows, columns));
    (0..rows).for_each(|row| {
        let sign = if row % 2 == 0 { -1.75 } else { 1.75 };
        let half = columns / 2;
        triplets.add_triplet(row, row % half, sign);
        triplets.add_triplet(row, half + (row % half), sign * 2.0);
    });
    let labels = Array1::from_vec((0..rows).map(|row| (row % 2) as i64).collect());
    (triplets.to_csr(), labels)
}

#[test]
fn large_multi_level_dense_dags_fit_predict_and_produce_classifier_and_regressor_metrics() {
    let (features, labels, targets) = large_separable_dense(600, 6);

    let mut classifier_builder = SKPipelineGraph::start::<f64>();
    let root = classifier_builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let branch_standard =
        classifier_builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let branch_union = classifier_builder.add_feature_union(
        SKFeatureUnion::build()
            .stages((
                SKMinMaxScaler::<f64>::build().build().unwrap(),
                SKRobustScaler::<f64>::build().build().unwrap(),
            ))
            .build()
            .unwrap(),
    );
    let merge = classifier_builder.add_merge();
    let post_merge = classifier_builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let classifier =
        classifier_builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    classifier_builder
        .connect(root, branch_standard)
        .connect(root, branch_union)
        .connect(branch_standard, merge)
        .connect(branch_union, merge)
        .connect(merge, post_merge)
        .connect(post_merge, classifier);

    let fitted_classifier = classifier_builder
        .build()
        .unwrap()
        .fit_classifier(&features, labels.view())
        .unwrap();
    assert_eq!(fitted_classifier.node_count(), 6);
    assert_eq!(fitted_classifier.number_of_features(), 18);
    let predicted_labels = fitted_classifier.predict_labels(&features).unwrap();
    let accuracy: f64 = sk_accuracy_score(
        labels.as_slice().unwrap(),
        predicted_labels.as_slice().unwrap(),
    )
    .unwrap();
    assert!(accuracy > 0.95, "expected accuracy > 0.95, got {accuracy}");

    let mut regressor_builder = SKPipelineGraph::start::<f64>();
    let regressor_left =
        regressor_builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let regressor_right =
        regressor_builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let regressor_sink =
        regressor_builder.add_regressor(SKLinearRegression::build().build().unwrap());
    regressor_builder
        .connect(regressor_left, regressor_sink)
        .connect(regressor_right, regressor_sink);
    let responses = regressor_builder
        .build()
        .unwrap()
        .fit_regressor(&features, targets.view())
        .unwrap()
        .predict(&features)
        .unwrap();
    let r2_score: f64 = sk_r2_score(targets.view(), responses.view()).unwrap();
    assert!(r2_score > 0.99, "expected R2 > 0.99, got {r2_score}");
}

#[test]
fn large_sparse_csr_dag_fits_without_densification_and_produces_metrics() {
    let (sparse_features, labels) = large_separable_sparse(600, 16);

    let mut builder = SKPipelineGraph::start::<f64>();
    let branch_imputer = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let branch_union = builder.add_feature_union(
        SKFeatureUnion::build()
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
            .unwrap(),
    );
    let merge = builder.add_merge();
    let classifier = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder
        .connect(branch_imputer, merge)
        .connect(branch_union, merge)
        .connect(merge, classifier);

    let fitted = builder
        .build()
        .unwrap()
        .fit_classifier(sparse_features.view(), labels.view())
        .unwrap();
    assert_eq!(fitted.number_of_features(), 48);
    fitted
        .topological_order()
        .iter()
        .for_each(|&node| assert!(fitted.fitted_node_block(node).unwrap().is_sparse()));
    assert!(
        fitted
            .graph_model()
            .transform_to_final(sparse_features.view())
            .unwrap()
            .is_sparse()
    );
    let predictions = fitted.predict_labels(sparse_features.view()).unwrap();
    let accuracy: f64 =
        sk_accuracy_score(labels.as_slice().unwrap(), predictions.as_slice().unwrap()).unwrap();
    assert!(
        accuracy > 0.90,
        "expected sparse accuracy > 0.90, got {accuracy}"
    );
}

#[test]
fn large_dag_concurrent_predictions_and_refits_match_sequential_results() {
    let (features, labels, _) = large_separable_dense(300, 4);
    let mut builder = SKPipelineGraph::start::<f64>();
    let standard = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let sink = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder.connect(standard, sink).connect(minmax, sink);
    let graph = builder.build().unwrap();
    let fitted = graph.fit_classifier(&features, labels.view()).unwrap();
    let expected_labels = fitted.predict_labels(&features).unwrap();
    let expected_probabilities = fitted.predict_probabilities(&features).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let refitted = graph.fit_classifier(&features, labels.view()).unwrap();
                    (
                        fitted.predict_labels(&features).unwrap(),
                        fitted.predict_probabilities(&features).unwrap(),
                        refitted.predict_labels(&features).unwrap(),
                    )
                })
            })
            .collect();
        handles.into_iter().for_each(|handle| {
            let (found_labels, found_probabilities, refitted_labels) = handle.join().unwrap();
            assert_eq!(found_labels, expected_labels);
            assert_eq!(found_probabilities, expected_probabilities);
            assert_eq!(refitted_labels, expected_labels);
        });
    });
}
