//! Graph-level validation: cycles and duplicated value names.
//!
//! The shared IR ships a structural validator; the importer must run it before
//! executing a foreign graph so a cycle is diagnosed as a cycle instead of
//! surfacing later as a misleading missing-initializer error.

use serde_onnx::ir::{Dim, ElemType, Graph, Model, Node, OpsetId, ValueInfo, ValueType};

use super::SKOnnxImportError;
use crate::graph::SKOnnxModel;

#[test]
fn a_graph_with_a_cycle_is_rejected_with_the_cycle_path() {
    let mut graph = Graph::new("cycle");
    graph.inputs.push(ValueInfo::new("X", matrix()));
    graph.outputs.push(ValueInfo::new("a", matrix()));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["b".to_string()],
        vec!["a".to_string()],
        vec![],
    ));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["a".to_string()],
        vec!["b".to_string()],
        vec![],
    ));
    let bytes = encode(graph);
    let error = SKOnnxModel::from_bytes(bytes).expect_err("a cycle must be rejected");
    match error {
        SKOnnxImportError::GraphCycle { path } => {
            assert_eq!(path.len(), 3, "the path closes on its first node: {path:?}");
            assert_eq!(path.first(), path.last());
            assert!(path.iter().any(|label| label.contains("Identity(a)")));
            assert!(path.iter().any(|label| label.contains("Identity(b)")));
        }
        other => panic!("expected a cycle error, got {other}"),
    }
}

#[test]
fn a_duplicated_node_output_is_rejected_before_inference() {
    let mut graph = Graph::new("duplicate");
    graph.inputs.push(ValueInfo::new("X", matrix()));
    graph.outputs.push(ValueInfo::new("out", matrix()));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["X".to_string()],
        vec!["out".to_string()],
        vec![],
    ));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["X".to_string()],
        vec!["out".to_string()],
        vec![],
    ));
    let bytes = encode(graph);
    let error = SKOnnxModel::from_bytes(bytes).expect_err("a duplicate output must be rejected");
    match error {
        SKOnnxImportError::InvalidModel { detail } => {
            assert!(
                detail.contains("out"),
                "the diagnosis must name the value: {detail}"
            );
        }
        other => panic!("expected an invalid-model error, got {other}"),
    }
}

fn matrix() -> ValueType {
    ValueType::tensor(
        ElemType::Double,
        Some(vec![Dim::Param("batch".into()), Dim::Fixed(2)]),
    )
}

fn encode(graph: Graph) -> Vec<u8> {
    serde_onnx::proto::encode_model(&Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 17,
        }],
    ))
}
