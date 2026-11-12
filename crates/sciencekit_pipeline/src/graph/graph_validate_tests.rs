//! Doomed-graph validation fixtures for cycle detection, single final estimator
//! enforcement and reachability (spec `pipeline-graph`, Task 1.2).

use sciencekit_common::{SKBuilder, SKError};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::SKPipelineGraph;

fn scaler() -> SKStandardScaler {
    SKStandardScaler::build().build().unwrap()
}

fn minmax() -> SKMinMaxScaler<f64> {
    SKMinMaxScaler::<f64>::build().build().unwrap()
}

fn classifier() -> SKLogisticRegression<f64> {
    SKLogisticRegression::build().build().unwrap()
}

fn regressor() -> SKLinearRegression {
    SKLinearRegression::build().build().unwrap()
}

#[test]
fn cyclic_graphs_are_rejected_with_invalid_hyperparameter_naming_the_cycle() {
    let mut builder = SKPipelineGraph::start::<f64>();
    let input = builder.add_transformer(scaler());
    let first = builder.add_transformer(minmax());
    let second = builder.add_transformer(scaler());
    let final_node = builder.add_classifier(classifier());
    builder
        .connect(input, first)
        .connect(first, second)
        .connect(second, first)
        .connect(second, final_node);

    let error = builder.validate().unwrap_err();
    match error {
        SKError::InvalidHyperparameter { reason, .. } => {
            assert!(
                reason.contains("node 1") && reason.contains("node 2"),
                "expected cycle reason to name node 1 and node 2, got: {reason}"
            );
        }
        other => panic!("expected InvalidHyperparameter for cycle, got {other:?}"),
    }
    assert!(matches!(
        builder.build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));

    let mut self_loop = SKPipelineGraph::start::<f64>();
    let loop_node = self_loop.add_transformer(scaler());
    let sink = self_loop.add_classifier(classifier());
    self_loop
        .connect(loop_node, loop_node)
        .connect(loop_node, sink);
    let self_loop_error = self_loop.build().err().unwrap();
    assert!(self_loop_error.to_string().contains("node 0"));
}

#[test]
fn zero_or_two_final_estimators_are_rejected_with_invalid_hyperparameter() {
    let empty_builder = SKPipelineGraph::start::<f64>();
    assert!(matches!(
        empty_builder.validate(),
        Err(SKError::InvalidHyperparameter {
            name: "final_estimator",
            ..
        })
    ));

    let mut transformer_only = SKPipelineGraph::start::<f64>();
    let first = transformer_only.add_transformer(scaler());
    let second = transformer_only.add_transformer(minmax());
    transformer_only.connect(first, second);
    assert!(matches!(
        transformer_only.build(),
        Err(SKError::InvalidHyperparameter {
            name: "final_estimator",
            ..
        })
    ));

    let mut two_finals = SKPipelineGraph::start::<f64>();
    let root = two_finals.add_transformer(scaler());
    let final_one = two_finals.add_classifier(classifier());
    let final_two = two_finals.add_regressor(regressor());
    two_finals.connect(root, final_one).connect(root, final_two);
    let error = two_finals.build().err().unwrap();
    match error {
        SKError::InvalidHyperparameter {
            name: "final_estimator",
            reason,
        } => {
            assert!(reason.contains("node 1") && reason.contains("node 2"));
        }
        other => panic!("expected final_estimator InvalidHyperparameter, got {other:?}"),
    }
}

#[test]
fn disconnected_and_dead_end_nodes_are_rejected_with_invalid_hyperparameter() {
    let mut isolated = SKPipelineGraph::start::<f64>();
    let root = isolated.add_transformer(scaler());
    let final_node = isolated.add_classifier(classifier());
    let orphan = isolated.add_transformer(minmax());
    isolated.connect(root, final_node);

    let isolated_error = isolated.build().err().unwrap();
    match isolated_error {
        SKError::InvalidHyperparameter { reason, .. } => {
            assert!(
                reason.contains(&orphan.to_string()),
                "expected orphan {orphan} in reason: {reason}"
            );
        }
        other => panic!("expected InvalidHyperparameter, got {other:?}"),
    }

    let mut dead_end = SKPipelineGraph::start::<f64>();
    let input = dead_end.add_transformer(scaler());
    let dangling = dead_end.add_transformer(minmax());
    let sink = dead_end.add_classifier(classifier());
    dead_end.connect(input, dangling).connect(input, sink);
    let dead_end_error = dead_end.build().err().unwrap();
    assert!(matches!(
        dead_end_error,
        SKError::InvalidHyperparameter { .. }
    ));
    assert!(dead_end_error.to_string().contains("node 1"));
}

#[test]
fn final_estimator_with_outgoing_edge_and_foreign_node_id_are_rejected() {
    let mut outgoing_final = SKPipelineGraph::start::<f64>();
    let first = outgoing_final.add_transformer(scaler());
    let final_node = outgoing_final.add_classifier(classifier());
    let after_final = outgoing_final.add_transformer(minmax());
    outgoing_final
        .connect(first, final_node)
        .connect(final_node, after_final);
    assert!(matches!(
        outgoing_final.build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));

    let mut larger_builder = SKPipelineGraph::start::<f64>();
    let _ = larger_builder.add_transformer(scaler());
    let _ = larger_builder.add_transformer(minmax());
    let foreign_identifier = larger_builder.add_classifier(classifier());

    let mut smaller_builder = SKPipelineGraph::start::<f64>();
    let valid_final = smaller_builder.add_classifier(classifier());
    smaller_builder.connect(foreign_identifier, valid_final);
    assert!(matches!(
        smaller_builder.build(),
        Err(SKError::InvalidHyperparameter { name: "edges", .. })
    ));
}

#[test]
fn valid_diamond_graph_passes_validation() {
    let mut builder = SKPipelineGraph::start::<f64>();
    let input = builder.add_transformer(scaler());
    let left = builder.add_transformer(minmax());
    let right = builder.add_transformer(scaler());
    let final_node = builder.add_classifier(classifier());
    builder
        .connect(input, left)
        .connect(input, right)
        .connect(left, final_node)
        .connect(right, final_node);

    assert!(builder.validate().is_ok());
    let graph = builder.build().unwrap();
    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 4);
    assert_eq!(graph.final_node(), final_node);
}
