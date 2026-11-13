//! Dispatching one graph node onto the documented operator implementations.

use std::collections::BTreeMap;

use serde_onnx::ir::{AttributeValue, ElemType, Node};

use super::attributes::{
    attr_float, attr_int, attr_ints, attr_ints_required, int_input, option_float_input,
};
use super::value::SKOnnxValue;
use super::{
    SKOnnxImportError, elementwise, matrix, reductions, selection, shape_ops, tree_ensemble,
};

pub(crate) fn run_node(
    index: usize,
    node: &Node,
    values: &BTreeMap<String, SKOnnxValue>,
) -> Result<Vec<SKOnnxValue>, SKOnnxImportError> {
    if !super::sk_onnx_supports_operator(&node.domain, &node.op_type) {
        return Err(SKOnnxImportError::UnsupportedOperator {
            domain: node.domain.clone(),
            operator: node.op_type.clone(),
            node_index: index,
        });
    }
    let operand = |position: usize| -> Result<&SKOnnxValue, SKOnnxImportError> {
        let name = &node.inputs[position];
        values
            .get(name)
            .ok_or_else(|| SKOnnxImportError::MissingInitializer {
                name: name.clone(),
                node_index: index,
            })
    };
    let operator = node.op_type.as_str();
    let result = match operator {
        "Add" => elementwise::binary(operand(0)?, operand(1)?, |a, b| a + b)?,
        "Sub" => elementwise::binary(operand(0)?, operand(1)?, |a, b| a - b)?,
        "Mul" => elementwise::binary(operand(0)?, operand(1)?, |a, b| a * b)?,
        "Div" => elementwise::binary(operand(0)?, operand(1)?, |a, b| a / b)?,
        "Pow" => elementwise::binary(operand(0)?, operand(1)?, f64::powf)?,
        "Equal" => elementwise::equal(operand(0)?, operand(1)?)?,
        "Greater" => elementwise::compare(operand(0)?, operand(1)?, |a, b| a > b)?,
        "Less" => elementwise::compare(operand(0)?, operand(1)?, |a, b| a < b)?,
        "Neg" => elementwise::unary(operand(0)?, |value| -value)?,
        "Abs" => elementwise::unary(operand(0)?, f64::abs)?,
        "Exp" => elementwise::unary(operand(0)?, f64::exp)?,
        "Sigmoid" => elementwise::unary(operand(0)?, sigmoid)?,
        "Tanh" => elementwise::unary(operand(0)?, f64::tanh)?,
        "Clip" => {
            let minimum = option_float_input(node, values, 1).or(attr_float(node, "min"));
            let maximum = option_float_input(node, values, 2).or(attr_float(node, "max"));
            elementwise::clip(operand(0)?, minimum, maximum)?
        }
        "MatMul" => matrix::mat_mul(operand(0)?, operand(1)?)?,
        "Transpose" => matrix::transpose(operand(0)?, attr_ints(node, "perm").as_deref())?,
        "Reshape" => matrix::reshape(operand(0)?, &attr_ints_required(node, "shape", values)?)?,
        "Flatten" => shape_ops::flatten(operand(0)?, attr_int(node, "axis", 1))?,
        "Concat" => {
            let inputs = (0..node.inputs.len())
                .map(|position| operand(position).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            shape_ops::concat(&inputs, attr_int(node, "axis", 0))?
        }
        "Unsqueeze" => {
            let axes = attr_ints_required(node, "axes", values)?;
            shape_ops::unsqueeze(operand(0)?, &axes)?
        }
        "Squeeze" => {
            let axes = attr_ints(node, "axes").or_else(|| int_input(node, values, 1));
            shape_ops::squeeze(operand(0)?, axes.as_deref())?
        }
        "Gather" => selection::gather(operand(0)?, operand(1)?, attr_int(node, "axis", 0))?,
        "Cast" => {
            let declared = attr_int(node, "to", 1);
            let element_type = ElemType::from_disc(declared as i32).ok_or(
                SKOnnxImportError::UnsupportedDtype {
                    element_type: format!("unknown element type {declared}"),
                },
            )?;
            selection::cast(operand(0)?, element_type)?
        }
        "ReduceSum" | "ReduceMean" => {
            let axes = attr_ints(node, "axes")
                .or_else(|| int_input(node, values, 1))
                .unwrap_or_default();
            let keep = attr_int(node, "keepdims", 1) != 0;
            if operator == "ReduceSum" {
                reductions::reduce_sum(operand(0)?, &axes, keep)?
            } else {
                reductions::reduce_mean(operand(0)?, &axes, keep)?
            }
        }
        "ArgMax" => reductions::arg_max(
            operand(0)?,
            attr_int(node, "axis", 0),
            attr_int(node, "keepdims", 1) != 0,
        )?,
        "ArgMin" => reductions::arg_min(
            operand(0)?,
            attr_int(node, "axis", 0),
            attr_int(node, "keepdims", 1) != 0,
        )?,
        "Softmax" => reductions::softmax(operand(0)?, attr_int(node, "axis", -1))?,
        "Identity" => operand(0)?.clone(),
        "Constant" => constant_value(node)?,
        "TopK" => {
            let count = operand(1)?
                .to_i64_values()
                .first()
                .copied()
                .unwrap_or(0)
                .max(0) as usize;
            let (values, indices) =
                selection::top_k(operand(0)?, count, attr_int(node, "axis", -1))?;
            return Ok(vec![values, indices]);
        }
        "TreeEnsembleClassifier" => return tree_ensemble::classifier(node, operand(0)?),
        "TreeEnsembleRegressor" => return tree_ensemble::regressor(node, operand(0)?),
        other => {
            return Err(SKOnnxImportError::UnsupportedOperator {
                domain: node.domain.clone(),
                operator: other.to_string(),
                node_index: index,
            });
        }
    };
    Ok(vec![result])
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn constant_value(node: &Node) -> Result<SKOnnxValue, SKOnnxImportError> {
    let value = node
        .attributes
        .iter()
        .find(|attribute| attribute.name == "value")
        .map(|attribute| &attribute.value);
    match value {
        Some(AttributeValue::Tensor(tensor)) => super::initializers::tensor_value(tensor),
        _ => Err(SKOnnxImportError::InvalidOperator {
            operator: "Constant".to_string(),
            detail: "only a tensor `value` attribute is supported".to_string(),
        }),
    }
}
