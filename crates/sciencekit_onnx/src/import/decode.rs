//! A compatibility shim over the pinned `serde-onnx` decoder.
//!
//! The pinned revision rejects a scalar attribute whose protobuf value equals
//! the scalar default (`0` for `INT`/`FLOAT`, `""` for `STRING`) because it
//! decides "attribute carries a value" from field presence alone. Zero-valued
//! attributes are legal ONNX — `axis = 0` and `keepdims = 0` are common — so
//! this module substitutes a sentinel before decoding and restores the zero
//! afterwards. The IR produced is exactly the shared `serde-onnx` IR.

use prost::Message;
use serde_onnx::ir::Model;
use serde_onnx::proto::{AttributeProto, ModelProto};

use super::SKOnnxImportError;

/// Decode bytes into the shared IR, tolerating legal zero-valued attributes.
pub(crate) fn decode_model(bytes: &[u8]) -> Result<Model, SKOnnxImportError> {
    let mut proto = ModelProto::decode(bytes).map_err(|error| SKOnnxImportError::InvalidModel {
        detail: error.to_string(),
    })?;
    let mut patches = Vec::new();
    if let Some(graph) = proto.graph.as_mut() {
        for (node_index, node) in graph.node.iter_mut().enumerate() {
            for (attribute_index, attribute) in node.attribute.iter_mut().enumerate() {
                if let Some(patch) = sentinel(attribute) {
                    patches.push((node_index, attribute_index, patch));
                }
            }
        }
    }
    let mut model = Model::try_from(&proto)?;
    for (node_index, attribute_index, patch) in patches {
        let attribute = &mut model.graph.nodes[node_index].attributes[attribute_index];
        patch.restore(&mut attribute.value);
    }
    Ok(model)
}

#[derive(Clone, Copy)]
enum Patch {
    Integer,
    Float,
    String,
}

fn sentinel(attribute: &mut AttributeProto) -> Option<Patch> {
    use serde_onnx::proto::attribute_proto::AttributeType;
    match AttributeType::try_from(attribute.r#type).ok()? {
        AttributeType::Int if attribute.i == 0 => {
            attribute.i = i64::MAX;
            Some(Patch::Integer)
        }
        AttributeType::Float if attribute.f == 0.0 => {
            attribute.f = f32::MIN;
            Some(Patch::Float)
        }
        AttributeType::String if attribute.s.is_empty() => {
            attribute.s = vec![0];
            Some(Patch::String)
        }
        _ => None,
    }
}

impl Patch {
    fn restore(self, value: &mut serde_onnx::ir::AttributeValue) {
        use serde_onnx::ir::AttributeValue;
        match (self, value) {
            (Patch::Integer, AttributeValue::Int(slot)) => *slot = 0,
            (Patch::Float, AttributeValue::Float(slot)) => *slot = 0.0,
            (Patch::String, AttributeValue::String(slot)) => slot.clear(),
            _ => {}
        }
    }
}
