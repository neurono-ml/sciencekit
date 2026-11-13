//! A declared graph input or output.

use sciencekit_common::SKError;

use super::{SKOnnxTensorDType, proto};

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

    pub(super) fn to_proto(&self) -> proto::SkOnnxDimensionProto {
        match self {
            SKOnnxDimension::Fixed(value) => proto::SkOnnxDimensionProto {
                dim_value: Some(*value as i64),
                dim_param: None,
            },
            SKOnnxDimension::Symbolic(name) => proto::SkOnnxDimensionProto {
                dim_value: None,
                dim_param: Some(name.clone()),
            },
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

    pub(super) fn to_proto(&self) -> proto::SkOnnxValueInfoProto {
        proto::SkOnnxValueInfoProto {
            name: self.name.clone(),
            r#type: Some(proto::SkOnnxTypeProto {
                tensor_type: Some(proto::SkOnnxTensorTypeProto {
                    elem_type: self.element_type.element_type(),
                    shape: Some(proto::SkOnnxTensorShapeProto {
                        dim: self.shape.iter().map(SKOnnxDimension::to_proto).collect(),
                    }),
                }),
            }),
            doc_string: String::new(),
        }
    }
}
