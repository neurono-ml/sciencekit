//! Initializer access for the native ONNX mapper.

use serde_onnx::ir::{Tensor, TensorData, TensorStorage};

use crate::graph::{SKOnnxModel, SKOnnxTensorDType};

/// One initializer's flat values plus its declared shape.
#[derive(Debug, Clone)]
pub(crate) struct SKOnnxConstant {
    /// The declared dimensions.
    pub(crate) shape: Vec<usize>,
    /// The flat values, promoted to `f64`.
    pub(crate) values: Vec<f64>,
    /// The declared element type, when the importer recognizes it.
    pub(crate) data_type: Option<SKOnnxTensorDType>,
}

impl SKOnnxConstant {
    /// The single value, when the tensor holds exactly one.
    pub(crate) fn scalar(&self) -> Option<f64> {
        self.values.first().copied()
    }

    /// Whether the declared element type stores floating-point values.
    pub(crate) fn is_floating(&self) -> bool {
        self.data_type.is_some_and(SKOnnxTensorDType::is_floating)
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
    let data_type = SKOnnxTensorDType::from_element_type(tensor.elem).ok();
    let values = match tensor.data()? {
        TensorData::F32(values) => values.iter().map(|value| f64::from(*value)).collect(),
        TensorData::F64(values) => values.clone(),
        TensorData::I64(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::I32(values) => values.iter().map(|value| f64::from(*value)).collect(),
        _ => return None,
    };
    Some(SKOnnxConstant {
        shape,
        values,
        data_type,
    })
}
