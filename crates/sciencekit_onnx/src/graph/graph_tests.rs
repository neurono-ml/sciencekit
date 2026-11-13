//! Tests for the graph builder and the serialized model interface.

use sciencekit_common::SKError;

use super::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_ML_TREE_OPSET, SKOnnxDimension, SKOnnxGraphBuilder, SKOnnxNode,
    SKOnnxTensorDType, SKOnnxValueInfo, proto,
};

fn identity_builder() -> SKOnnxGraphBuilder {
    let mut builder = SKOnnxGraphBuilder::new("identity");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("features", SKOnnxTensorDType::Float, 3)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["features".to_string()],
            vec!["transformed".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Float, 3)
                .expect("a valid output"),
        );
    builder
}

#[test]
fn graph_identity_round_trips_through_protobuf() {
    let model = identity_builder()
        .build()
        .expect("the identity graph is valid");
    let decoded = <proto::SkOnnxModelProto as prost::Message>::decode(model.bytes())
        .expect("the bytes decode as a ModelProto");
    assert_eq!(decoded.producer_name, "sciencekit");
    assert_eq!(decoded.ir_version, super::SK_ONNX_IR_VERSION);
    assert_eq!(decoded.opset_import.len(), 2);
    assert_eq!(decoded.opset_import[0].version, SK_ONNX_DEFAULT_OPSET);
    assert_eq!(decoded.opset_import[1].domain, "ai.onnx.ml");
    assert_eq!(decoded.opset_import[1].version, SK_ONNX_ML_TREE_OPSET);
    let graph = decoded.graph.expect("the model carries a graph");
    assert_eq!(graph.name, "identity");
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.output.len(), 1);
    assert_eq!(model.opset(), SK_ONNX_DEFAULT_OPSET);
    assert_eq!(model.node_count(), 1);
}

#[test]
fn graph_declares_batch_and_feature_dimensions() {
    let model = identity_builder()
        .build()
        .expect("the identity graph is valid");
    let input = &model.inputs()[0];
    assert_eq!(input.name(), "features");
    assert_eq!(input.element_type(), SKOnnxTensorDType::Float);
    assert_eq!(input.feature_count(), Some(3));
    assert_eq!(input.shape()[0], SKOnnxDimension::batch());
    assert_eq!(model.outputs()[0].feature_count(), Some(3));
}

#[test]
fn graph_serializes_floating_initializers_as_raw_f32_bytes() {
    let mut builder = SKOnnxGraphBuilder::new("scaled");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("features", SKOnnxTensorDType::Float, 2)
                .expect("a valid input"),
        )
        .add_floating_initializer("scale", vec![2], &[2.0_f32, 4.0])
        .add_node(SKOnnxNode::new(
            "Mul",
            vec!["features".to_string(), "scale".to_string()],
            vec!["transformed".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Float, 2)
                .expect("a valid output"),
        );
    let model = builder.build().expect("the graph is valid");
    let decoded = <proto::SkOnnxModelProto as prost::Message>::decode(model.bytes())
        .expect("the bytes decode as a ModelProto");
    let initializer = &decoded.graph.expect("a graph").initializer[0];
    assert_eq!(initializer.name, "scale");
    assert_eq!(initializer.data_type, 1);
    assert_eq!(initializer.dims, vec![2]);
    let values: Vec<f32> = initializer
        .raw_data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    assert_eq!(values, vec![2.0, 4.0]);
}

#[test]
fn graph_rejects_an_output_no_node_produces() {
    let mut builder = SKOnnxGraphBuilder::new("broken");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("features", SKOnnxTensorDType::Float, 2)
                .expect("a valid input"),
        )
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Float, 2)
                .expect("a valid output"),
        );
    let error = builder
        .build()
        .expect_err("an unproduced output is rejected");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "onnx_graph_output",
            ..
        }
    ));
}

#[test]
fn graph_rejects_a_node_reading_an_undefined_input() {
    let mut builder = SKOnnxGraphBuilder::new("broken");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("features", SKOnnxTensorDType::Float, 2)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Add",
            vec!["features".to_string(), "missing".to_string()],
            vec!["transformed".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Float, 2)
                .expect("a valid output"),
        );
    let error = builder
        .build()
        .expect_err("an undefined node input is rejected");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "onnx_node_input",
            ..
        }
    ));
}

#[test]
fn graph_rejects_an_initializer_whose_values_do_not_match_its_shape() {
    let mut builder = SKOnnxGraphBuilder::new("broken");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("features", SKOnnxTensorDType::Float, 2)
                .expect("a valid input"),
        )
        .add_floating_initializer("scale", vec![3], &[1.0_f32, 2.0])
        .add_node(SKOnnxNode::new(
            "Mul",
            vec!["features".to_string(), "scale".to_string()],
            vec!["transformed".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Float, 2)
                .expect("a valid output"),
        );
    let error = builder
        .build()
        .expect_err("a shape/value disagreement is rejected");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn graph_rejects_an_empty_interface() {
    let error = SKOnnxGraphBuilder::new("empty")
        .build()
        .expect_err("a graph with no input is rejected");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "onnx_graph_inputs",
            ..
        }
    ));
}
