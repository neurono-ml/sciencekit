//! Initializer access for the native ONNX mapper.

use serde_onnx::ir::{Tensor, TensorData, TensorStorage};

use crate::graph::SKOnnxModel;

/// One initializer's flat values plus its declared shape.
#[derive(Debug, Clone)]
pub(crate) struct SKOnnxConstant {
    /// The declared dimensions.
    pub(crate) shape: Vec<usize>,
    /// The flat values, promoted to `f64`.
    pub(crate) values: Vec<f64>,
}

impl SKOnnxConstant {
    /// The single value, when the tensor holds exactly one.
    pub(crate) fn scalar(&self) -> Option<f64> {
        self.values.first().copied()
    }
}

/// Look up an initializer by name.
pub(crate) fn constant(model: &SKOnnxModel, name: &str) -> Option<SKOnnxConstant> {
    let graph = &model.model().graph;
    graph
        .initializers
        .iter()
        .find(|tensor| tensor.name == name)
        .and_then(constant_from_tensor)
}

fn constant_from_tensor(tensor: &Tensor) -> Option<SKOnnxConstant> {
    if let TensorStorage::External(_) = tensor.data {
        return None;
    }
    let shape: Vec<usize> = tensor
        .shape
        .iter()
        .map(|dimension| dimension.fixed_value().unwrap_or(0).max(0) as usize)
        .collect();
    let values = match tensor.data()? {
        TensorData::F32(values) => values.iter().map(|value| f64::from(*value)).collect(),
        TensorData::F64(values) => values.clone(),
        TensorData::I64(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::I32(values) => values.iter().map(|value| f64::from(*value)).collect(),
        _ => return None,
    };
    Some(SKOnnxConstant { shape, values })
}
