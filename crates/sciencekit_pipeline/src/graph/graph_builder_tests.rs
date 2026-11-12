//! Builder construction, node identifier assignment, edge insertion and
//! configuration tests for [`super::SKPipelineGraphBuilder`] (spec
//! `pipeline-graph`, Task 1.1).

use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::SKFeatureUnion;

use super::{SKNodeIdentifier, SKPipelineGraph, SKPipelineGraphNodeKind};

#[test]
fn graph_builder_assigns_monotonic_node_identifiers_and_records_edges() {
    let mut builder = SKPipelineGraph::start::<f64>();
    assert_eq!(builder.node_count(), 0);
    assert_eq!(builder.edge_count(), 0);
    assert_eq!(builder.backend_kind(), SKBackendKind::Faer);
    assert_eq!(builder.execution_intent(), SKExecutionMode::Automatic);

    let scaler_node = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();
    let union_node = builder.add_feature_union(union);
    let merge_node = builder.add_merge();
    let classifier_node = builder.add_classifier(SKLogisticRegression::build().build().unwrap());

    assert_eq!(scaler_node.index(), 0);
    assert_eq!(union_node.index(), 1);
    assert_eq!(merge_node.index(), 2);
    assert_eq!(classifier_node.index(), 3);
    assert_eq!(scaler_node.to_string(), "node 0");
    assert!(scaler_node < union_node);

    assert_eq!(
        builder.node_kind(scaler_node),
        Some(SKPipelineGraphNodeKind::Transformer)
    );
    assert_eq!(
        builder.node_kind(union_node),
        Some(SKPipelineGraphNodeKind::FeatureUnion)
    );
    assert_eq!(
        builder.node_kind(merge_node),
        Some(SKPipelineGraphNodeKind::Merge)
    );
    assert_eq!(
        builder.node_kind(classifier_node),
        Some(SKPipelineGraphNodeKind::FinalEstimator)
    );

    builder
        .connect(scaler_node, merge_node)
        .connect(union_node, merge_node)
        .connect(merge_node, classifier_node);

    assert_eq!(builder.node_count(), 4);
    assert_eq!(builder.edge_count(), 3);
}

#[test]
fn graph_builder_carries_backend_and_execution_mode_overrides() {
    let mut builder = SKPipelineGraph::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous);
    let scaler_node = builder.add_transformer(SKStandardScaler::build().build().unwrap());
    let regressor_node = builder.add_regressor(SKLinearRegression::build().build().unwrap());
    builder.connect(scaler_node, regressor_node);

    assert_eq!(builder.backend_kind(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        builder.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
    let _: SKNodeIdentifier = regressor_node;
}
