//! The `serde-onnx` zero-attribute shim is narrow and compensates for one
//! upstream bug: the pinned decoder decides that a scalar attribute carries a
//! value from the encoded protobuf field instead of from `AttributeProto.type`,
//! so an INT `0`, a FLOAT `0.0` or an empty STRING looks absent and decoding
//! fails with "carries no value". These tests prove both halves: the pinned
//! decoder really rejects those legal attributes, and the shim restores exactly
//! those values without touching non-zero ones.

use prost::Message;
use serde_onnx::ir::AttributeValue;
use serde_onnx::proto::ModelProto;

use super::decode::decode_model;
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

#[test]
fn the_pinned_decoder_rejects_a_legal_zero_integer_attribute() {
    let bytes = flatten_model(0);
    let proto = ModelProto::decode(bytes.as_slice()).expect("the bytes are a ModelProto");
    let error = serde_onnx::ir::Model::try_from(&proto).expect_err("the upstream bug rejects it");
    assert!(
        error.to_string().contains("carries no value"),
        "the pinned decoder must diagnose the zero attribute as absent: {error}"
    );
}

#[test]
fn the_shim_restores_zero_integer_and_float_attributes() {
    let decoded = decode_model(&flatten_model(0)).expect("the shim decodes it");
    let axis = &decoded.graph.nodes[0].attributes[0];
    assert_eq!(axis.name, "axis");
    assert_eq!(axis.value, AttributeValue::Int(0));

    let decoded = decode_model(&clipped_model(0.0)).expect("the shim decodes it");
    let minimum = decoded.graph.nodes[0]
        .attributes
        .iter()
        .find(|attribute| attribute.name == "min")
        .expect("the min attribute");
    assert_eq!(minimum.value, AttributeValue::Float(0.0));
}

#[test]
fn the_shim_leaves_non_zero_attributes_untouched() {
    let decoded = decode_model(&flatten_model(1)).expect("the model decodes");
    assert_eq!(
        decoded.graph.nodes[0].attributes[0].value,
        AttributeValue::Int(1)
    );

    let decoded = decode_model(&clipped_model(0.5)).expect("the model decodes");
    let minimum = decoded.graph.nodes[0]
        .attributes
        .iter()
        .find(|attribute| attribute.name == "min")
        .expect("the min attribute");
    assert_eq!(minimum.value, AttributeValue::Float(0.5));
}

#[test]
fn the_pinned_decoder_rejects_an_empty_string_attribute() {
    let mut builder = SKOnnxGraphBuilder::new("empty_string");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(
            SKOnnxNode::new(
                "Custom",
                vec!["X".to_string()],
                vec!["out".to_string()],
            )
            .attribute_string("label", ""),
        )
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 2)
                .expect("a valid output"),
        );
    let bytes = builder.build().expect("the graph encodes").bytes().to_vec();
    let proto = ModelProto::decode(bytes.as_slice()).expect("the bytes are a ModelProto");
    let error = serde_onnx::ir::Model::try_from(&proto).expect_err("the upstream bug rejects it");
    assert!(error.to_string().contains("carries no value"), "{error}");

    let decoded = decode_model(&bytes).expect("the shim decodes it");
    assert_eq!(
        decoded.graph.nodes[0].attributes[0].value,
        AttributeValue::String(String::new())
    );
}

fn flatten_model(axis: i64) -> Vec<u8> {
    let mut builder = SKOnnxGraphBuilder::new("flatten");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(
            SKOnnxNode::new(
                "Flatten",
                vec!["X".to_string()],
                vec!["out".to_string()],
            )
            .attribute_integer("axis", axis),
        )
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 2)
                .expect("a valid output"),
        );
    builder.build().expect("the graph encodes").bytes().to_vec()
}

fn clipped_model(minimum: f64) -> Vec<u8> {
    let mut builder = SKOnnxGraphBuilder::new("clip");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(
            SKOnnxNode::new("Clip", vec!["X".to_string()], vec!["out".to_string()])
                .attribute_float("min", minimum),
        )
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 2)
                .expect("a valid output"),
        );
    builder.build().expect("the graph encodes").bytes().to_vec()
}
