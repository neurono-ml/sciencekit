//! A graph node and its typed attributes.
//!
//! The node stores the shared [`serde_onnx::ir::Attribute`] values directly, so
//! the exporter and the importer speak exactly the same attribute vocabulary.

use serde_onnx::ir::{Attribute, AttributeValue, Node};

use super::SK_ONNX_ML_DOMAIN;

/// An ONNX operator invocation with named inputs, outputs and attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SKOnnxNode {
    op_type: String,
    domain: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<Attribute>,
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

    /// The typed attributes, in declaration order.
    pub fn attributes(&self) -> &[Attribute] {
        &self.attributes
    }

    /// Attach a scalar float attribute.
    pub fn attribute_float(mut self, name: impl Into<String>, value: f64) -> Self {
        self.attributes.push(Attribute::float(name, value as f32));
        self
    }

    /// Attach a scalar integer attribute.
    pub fn attribute_integer(mut self, name: impl Into<String>, value: i64) -> Self {
        self.attributes.push(Attribute::int(name, value));
        self
    }

    /// Attach a string attribute.
    pub fn attribute_string(mut self, name: impl Into<String>, value: impl AsRef<str>) -> Self {
        self.attributes
            .push(Attribute::string(name, value.as_ref().to_string()));
        self
    }

    /// Attach a list of floats.
    pub fn attribute_floats(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.attributes.push(Attribute::floats(
            name,
            values.into_iter().map(|value| value as f32).collect(),
        ));
        self
    }

    /// Attach a list of integers.
    pub fn attribute_integers(mut self, name: impl Into<String>, values: Vec<i64>) -> Self {
        self.attributes.push(Attribute::ints(name, values));
        self
    }

    /// Attach a list of strings.
    pub fn attribute_strings<S: AsRef<str>>(
        mut self,
        name: impl Into<String>,
        values: Vec<S>,
    ) -> Self {
        self.attributes.push(Attribute::strings(
            name,
            values
                .into_iter()
                .map(|value| value.as_ref().to_string())
                .collect(),
        ));
        self
    }

    pub(super) fn to_ir(&self) -> Node {
        Node::new(
            self.op_type.clone(),
            self.domain.clone(),
            self.inputs.clone(),
            self.outputs.clone(),
            self.attributes.clone(),
        )
    }

    /// The scalar float attribute named `name`, when present.
    pub fn float_attribute(&self, name: &str) -> Option<f32> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
            .and_then(|attribute| match &attribute.value {
                AttributeValue::Float(value) => Some(*value),
                _ => None,
            })
    }

    /// The scalar integer attribute named `name`, when present.
    pub fn integer_attribute(&self, name: &str) -> Option<i64> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
            .and_then(|attribute| match &attribute.value {
                AttributeValue::Int(value) => Some(*value),
                _ => None,
            })
    }
}
