//! A graph node and its typed attributes.

use super::{SK_ONNX_ML_DOMAIN, proto};

/// The protobuf `AttributeProto.AttributeType` discriminants.
const ATTRIBUTE_TYPE_FLOAT: i32 = 1;
const ATTRIBUTE_TYPE_INT: i32 = 2;
const ATTRIBUTE_TYPE_STRING: i32 = 3;
const ATTRIBUTE_TYPE_FLOATS: i32 = 6;
const ATTRIBUTE_TYPE_INTS: i32 = 7;
const ATTRIBUTE_TYPE_STRINGS: i32 = 8;

/// An ONNX operator invocation with named inputs, outputs and attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SKOnnxNode {
    op_type: String,
    domain: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<proto::SkOnnxAttributeProto>,
}

impl SKOnnxNode {
    /// Create a node of the default ONNX domain.
    pub fn new(op_type: impl Into<String>, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        SKOnnxNode {
            op_type: op_type.into(),
            domain: String::new(),
            inputs,
            outputs,
            attributes: Vec::new(),
        }
    }

    /// Create a node in the ONNX-ML domain.
    pub fn ml(op_type: impl Into<String>, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        SKOnnxNode {
            op_type: op_type.into(),
            domain: SK_ONNX_ML_DOMAIN.to_string(),
            inputs,
            outputs,
            attributes: Vec::new(),
        }
    }

    /// The operator type recorded in the node.
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// The operator domain (`""` is the default ONNX domain).
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// The input tensor names in operator order.
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// The output tensor names.
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Attach a scalar float attribute.
    pub fn attribute_float(mut self, name: impl Into<String>, value: f64) -> Self {
        self.attributes.push(proto::SkOnnxAttributeProto {
            name: name.into(),
            f: value as f32,
            r#type: ATTRIBUTE_TYPE_FLOAT,
            ..Default::default()
        });
        self
    }

    /// Attach a scalar integer attribute.
    pub fn attribute_integer(mut self, name: impl Into<String>, value: i64) -> Self {
        self.attributes.push(proto::SkOnnxAttributeProto {
            name: name.into(),
            i: value,
            r#type: ATTRIBUTE_TYPE_INT,
            ..Default::default()
        });
        self
    }

    /// Attach a string attribute.
    pub fn attribute_string(mut self, name: impl Into<String>, value: impl AsRef<str>) -> Self {
        self.attributes.push(proto::SkOnnxAttributeProto {
            name: name.into(),
            s: value.as_ref().as_bytes().to_vec(),
            r#type: ATTRIBUTE_TYPE_STRING,
            ..Default::default()
        });
        self
    }

    /// Attach a list of floats.
    pub fn attribute_floats(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.attributes.push(proto::SkOnnxAttributeProto {
            name: name.into(),
            floats: values.into_iter().map(|value| value as f32).collect(),
            r#type: ATTRIBUTE_TYPE_FLOATS,
            ..Default::default()
        });
        self
    }

    /// Attach a list of integers.
    pub fn attribute_integers(mut self, name: impl Into<String>, values: Vec<i64>) -> Self {
        self.attributes.push(proto::SkOnnxAttributeProto {
            name: name.into(),
            ints: values,
            r#type: ATTRIBUTE_TYPE_INTS,
            ..Default::default()
        });
        self
    }

    /// Attach a list of strings.
    pub fn attribute_strings<S: AsRef<str>>(
        mut self,
        name: impl Into<String>,
        values: Vec<S>,
    ) -> Self {
        self.attributes.push(proto::SkOnnxAttributeProto {
            name: name.into(),
            strings: values
                .into_iter()
                .map(|value| value.as_ref().as_bytes().to_vec())
                .collect(),
            r#type: ATTRIBUTE_TYPE_STRINGS,
            ..Default::default()
        });
        self
    }

    pub(super) fn to_proto(&self) -> proto::SkOnnxNodeProto {
        proto::SkOnnxNodeProto {
            input: self.inputs.clone(),
            output: self.outputs.clone(),
            name: String::new(),
            op_type: self.op_type.clone(),
            attribute: self.attributes.clone(),
            domain: self.domain.clone(),
        }
    }
}
