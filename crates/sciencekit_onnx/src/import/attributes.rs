//! Attribute and optional-input accessors shared by the interpreter.

use std::collections::BTreeMap;

use serde_onnx::ir::{AttributeValue, Node};

use super::SKOnnxImportError;
use super::value::SKOnnxValue;

/// The typed attribute named `name`, when present.
pub(crate) fn attribute<'a>(node: &'a Node, name: &str) -> Option<&'a AttributeValue> {
    node.attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| &attribute.value)
}

/// A scalar integer attribute with a default.
pub(crate) fn attr_int(node: &Node, name: &str, default: i64) -> i64 {
    match attribute(node, name) {
        Some(AttributeValue::Int(value)) => *value,
        _ => default,
    }
}

/// A scalar float attribute, when present.
pub(crate) fn attr_float(node: &Node, name: &str) -> Option<f64> {
    match attribute(node, name) {
        Some(AttributeValue::Float(value)) => Some(f64::from(*value)),
        _ => None,
    }
}

/// An integer-list attribute, when present.
pub(crate) fn attr_ints(node: &Node, name: &str) -> Option<Vec<i64>> {
    match attribute(node, name) {
        Some(AttributeValue::Ints(values)) => Some(values.clone()),
        _ => None,
    }
}

/// Read an integer list from the optional input at `position`.
pub(crate) fn int_input(
    node: &Node,
    values: &BTreeMap<String, SKOnnxValue>,
    position: usize,
) -> Option<Vec<i64>> {
    node.inputs
        .get(position)
        .filter(|name| !name.is_empty())
        .and_then(|name| values.get(name))
        .map(SKOnnxValue::to_i64_values)
}

/// Read an integer list from an attribute or, failing that, from input 1.
pub(crate) fn attr_ints_required(
    node: &Node,
    name: &str,
    values: &BTreeMap<String, SKOnnxValue>,
) -> Result<Vec<i64>, SKOnnxImportError> {
    attr_ints(node, name)
        .or_else(|| int_input(node, values, 1))
        .ok_or_else(|| SKOnnxImportError::InvalidOperator {
            operator: node.op_type.clone(),
            detail: format!("the required `{name}` attribute or input is missing"),
        })
}

/// Read the first value of the optional float input at `position`.
pub(crate) fn option_float_input(
    node: &Node,
    values: &BTreeMap<String, SKOnnxValue>,
    position: usize,
) -> Option<f64> {
    node.inputs
        .get(position)
        .filter(|name| !name.is_empty())
        .and_then(|name| values.get(name))
        .and_then(|value| value.to_f64_values().first().copied())
}
