//! Determinism across repeated fits, graph accessor invariants, `Send + Sync`
//! bounds and multi-threaded concurrent prediction tests for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`, Task 3.2).

use std::collections::BTreeSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKFeatureBlock, SKRegressorPredictor,
};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::{
    SKBoxedFinalModel, SKNodeBlockStore, SKNodeIdentifier, SKPipelineGraph,
    SKPipelineGraphClassifier, SKPipelineGraphModel, SKPipelineGraphRegressor,
};

fn training_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0,
        ],
    )
    .unwrap()
}

fn training_labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1])
}

fn regression_targets() -> Array1<f64> {
    Array1::from_vec(vec![2.0, 5.0, 8.0, 11.0, 32.0, 35.0, 38.0, 41.0])
}

fn query_features() -> Array2<f64> {
    Array2::from_shape_vec((4, 2), vec![0.5, 1.5, 2.5, 3.5, 10.5, 11.5, 12.5, 13.5]).unwrap()
}

fn dense_matrix(block: &SKFeatureBlock<f64>) -> Array2<f64> {
    match block.view() {
        SKDataView::Dense(matrix) => matrix.to_owned(),
        other => panic!("expected dense block, got {other:?}"),
    }
}

fn build_classifier_diamond() -> SKPipelineGraph<f64> {
    let mut builder = SKPipelineGraph::start::<f64>();
    let imputer = builder.add_transformer(SKSimpleImputer::<f64>::build().build().unwrap());
    let standard = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let sink = builder.add_classifier(SKLogisticRegression::build().build().unwrap());
    builder
        .connect(imputer, standard)
        .connect(imputer, minmax)
        .connect(standard, sink)
        .connect(minmax, sink);
    builder.build().unwrap()
}

fn build_regressor_diamond() -> SKPipelineGraph<f64> {
    let mut builder = SKPipelineGraph::start::<f64>();
    let standard = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let minmax = builder.add_transformer(SKMinMaxScaler::<f64>::build().build().unwrap());
    let sink = builder.add_regressor(SKLinearRegression::build().build().unwrap());
    builder.connect(standard, sink).connect(minmax, sink);
    builder.build().unwrap()
}

#[test]
fn graph_definitions_and_fitted_models_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKPipelineGraph<f64>>();
    assert_send_sync::<SKPipelineGraphModel<f64>>();
    assert_send_sync::<SKPipelineGraphClassifier<f64>>();
    assert_send_sync::<SKPipelineGraphRegressor<f64>>();
    assert_send_sync::<SKBoxedFinalModel<f64>>();
    assert_send_sync::<SKNodeBlockStore<f64>>();
}

#[test]
fn repeated_fits_produce_identical_node_blocks_and_predictions_and_accessors_cover_every_node() {
    let features = training_features();
    let labels = training_labels();
    let queries = query_features();
    let graph = build_classifier_diamond();

    let first_run = graph.fit_classifier(&features, labels.view()).unwrap();
    let second_run = build_classifier_diamond()
        .fit_classifier(&features, labels.view())
        .unwrap();

    assert_eq!(first_run.node_count(), 4);
    assert_eq!(first_run.edge_count(), 4);
    assert_eq!(first_run.number_of_features(), 4);
    assert_eq!(first_run.final_node(), SKNodeIdentifier::new(3));
    assert_eq!(
        first_run.topological_order(),
        second_run.topological_order()
    );

    let unique_nodes: BTreeSet<SKNodeIdentifier> =
        first_run.topological_order().iter().copied().collect();
    assert_eq!(unique_nodes.len(), first_run.node_count());

    first_run.topological_order().iter().for_each(|&node| {
        let left_block = dense_matrix(first_run.fitted_node_block(node).unwrap());
        let right_block = dense_matrix(second_run.fitted_node_block(node).unwrap());
        assert_eq!(left_block, right_block);
    });

    assert_eq!(
        first_run.predict_labels(&queries).unwrap(),
        second_run.predict_labels(&queries).unwrap()
    );
    assert_eq!(
        first_run.predict_probabilities(&queries).unwrap(),
        second_run.predict_probabilities(&queries).unwrap()
    );
}

#[test]
fn concurrent_prediction_and_refitting_across_threads_match_sequential_results() {
    let features = training_features();
    let labels = training_labels();
    let targets = regression_targets();
    let queries = query_features();

    let classifier_graph = build_classifier_diamond();
    let fitted_classifier = classifier_graph
        .fit_classifier(&features, labels.view())
        .unwrap();
    let expected_labels = fitted_classifier.predict_labels(&queries).unwrap();
    let expected_probabilities = fitted_classifier.predict_probabilities(&queries).unwrap();

    let fitted_regressor = build_regressor_diamond()
        .fit_regressor(&features, targets.view())
        .unwrap();
    assert_eq!(fitted_regressor.number_of_features(), 4);
    assert_eq!(fitted_regressor.final_node(), SKNodeIdentifier::new(2));
    assert!(
        fitted_regressor
            .fitted_node_block(SKNodeIdentifier::new(0))
            .is_some()
    );
    let expected_responses = fitted_regressor.predict(&queries).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let shared_classifier = &fitted_classifier;
                let shared_regressor = &fitted_regressor;
                let shared_graph = &classifier_graph;
                let shared_features = &features;
                let shared_labels = &labels;
                let shared_queries = &queries;
                scope.spawn(move || {
                    let refitted = shared_graph
                        .fit_classifier(shared_features, shared_labels.view())
                        .unwrap();
                    (
                        shared_classifier.predict_labels(shared_queries).unwrap(),
                        shared_classifier
                            .predict_probabilities(shared_queries)
                            .unwrap(),
                        shared_regressor.predict(shared_queries).unwrap(),
                        refitted.predict_labels(shared_queries).unwrap(),
                    )
                })
            })
            .collect();

        handles.into_iter().for_each(|handle| {
            let (found_labels, found_probabilities, found_responses, refitted_labels) =
                handle.join().unwrap();
            assert_eq!(found_labels, expected_labels);
            assert_eq!(found_probabilities, expected_probabilities);
            assert_eq!(found_responses, expected_responses);
            assert_eq!(refitted_labels, expected_labels);
        });
    });
}
