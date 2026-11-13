//! A declared graph input or output.

use sciencekit_common::SKError;
use serde_onnx::ir::{Dim, TensorType, ValueInfo, ValueType};

use super::SKOnnxTensorDType;

/// One dimension of a declared tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SKOnnxDimension {
    /// A fixed extent known at export time.
    Fixed(usize),
    /// A symbolic extent resolved at run time (the batch dimension).
    Symbolic(String),
}

impl SKOnnxDimension {
    /// The conventional symbolic batch dimension.
    pub fn batch() -> Self {
        SKOnnxDimension::Symbolic("batch_size".to_string())
    }

    pub(super) fn to_ir(&self) -> Dim {
        match self {
            SKOnnxDimension::Fixed(value) => Dim::Fixed(*value as i64),
            SKOnnxDimension::Symbolic(name) => Dim::Param(name.clone()),
        }
    }

    /// Recover a dimension from the shared IR.
    pub fn from_ir(dimension: &Dim) -> Self {
        match dimension {
            Dim::Fixed(value) => SKOnnxDimension::Fixed((*value).max(0) as usize),
            Dim::Param(name) => SKOnnxDimension::Symbolic(name.clone()),
            Dim::Unknown => SKOnnxDimension::Symbolic("unknown".to_string()),
        }
    }
}

/// A typed graph input or output declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKOnnxValueInfo {
    name: String,
    element_type: SKOnnxTensorDType,
    shape: Vec<SKOnnxDimension>,
}

impl SKOnnxValueInfo {
    /// Declare a tensor with a name, element type and shape.
    pub fn new(
        name: impl Into<String>,
        element_type: SKOnnxTensorDType,
        shape: Vec<SKOnnxDimension>,
    ) -> Result<Self, SKError> {
        let name = name.into();
        if name.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "graph_value_name",
                reason: "a graph input or output name must not be empty".to_string(),
            });
        }
        Ok(SKOnnxValueInfo {
            name,
            element_type,
            shape,
        })
    }

    /// Declare a two-dimensional floating matrix input whose rows are the batch.
    pub fn floating_matrix(
        name: impl Into<String>,
        element_type: SKOnnxTensorDType,
        feature_count: usize,
    ) -> Result<Self, SKError> {
        SKOnnxValueInfo::new(
            name,
            element_type,
            vec![
                SKOnnxDimension::batch(),
                SKOnnxDimension::Fixed(feature_count),
            ],
        )
    }

    /// The tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The declared element type.
    pub fn element_type(&self) -> SKOnnxTensorDType {
        self.element_type
    }

    /// The declared shape.
    pub fn shape(&self) -> &[SKOnnxDimension] {
        &self.shape
    }

    /// The fixed feature count of a matrix declaration, when the last dimension
    /// is fixed.
    pub fn feature_count(&self) -> Option<usize> {
        match self.shape.last()? {
            SKOnnxDimension::Fixed(count) => Some(*count),
            SKOnnxDimension::Symbolic(_) => None,
        }
    }

    pub(super) fn to_ir(&self) -> ValueInfo {
        let tensor = TensorType::shaped(
            self.element_type.to_element_type(),
            self.shape.iter().map(SKOnnxDimension::to_ir),
        );
        ValueInfo::new(self.name.clone(), ValueType::Tensor(tensor))
    }

    pub(super) fn from_ir(info: &ValueInfo) -> Result<Self, SKError> {
        let value_type = info.value_type.as_ref().ok_or_else(|| {
            SKError::Conversion(format!(
                "the ONNX value `{}` does not declare a tensor type",
                info.name
            ))
        })?;
        let ValueType::Tensor(tensor) = value_type else {
            return Err(SKError::UnsupportedOperation {
                operation: "onnx_import",
                reason: format!(
                    "the ONNX value `{}` is not a dense tensor and is outside the supported subset",
                    info.name
                ),
            });
        };
        let element_type = SKOnnxTensorDType::from_element_type(tensor.elem)?;
        let shape = tensor
            .shape
            .as_ref()
            .map(|dims| dims.iter().map(SKOnnxDimension::from_ir).collect())
            .unwrap_or_default();
        SKOnnxValueInfo::new(info.name.clone(), element_type, shape)
    }
}

impl From<&SKOnnxValueInfo> for ValueInfo {
    fn from(info: &SKOnnxValueInfo) -> Self {
        info.to_ir()
    }
}
