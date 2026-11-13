//! The interface analysis: feature input plus the prediction head.

use serde_onnx::ir::{ElemType, ValueType};

use super::SKOnnxImportError;
use crate::graph::{SKOnnxModel, SKOnnxTensorDType};

/// The prediction surface an imported model exposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SKOnnxPredictionHead {
    /// One float output with a single column per row.
    Regression {
        /// The graph output tensor name.
        output: String,
        /// The element type of the scores.
        element_type: ElemType,
        /// The number of columns the output carries.
        width: usize,
    },
    /// An integer label output plus a class-probability matrix.
    Classification {
        /// The integer label tensor name.
        labels: String,
        /// The probability matrix tensor name.
        probabilities: String,
        /// The number of classes declared by the matrix width.
        class_count: usize,
        /// The element type of the probabilities.
        element_type: ElemType,
    },
    /// A single integer label output, without a probability matrix.
    LabelsOnly {
        /// The integer label tensor name.
        labels: String,
    },
}

/// The part of the interface a dense predictor needs.
#[derive(Debug, Clone)]
pub(crate) struct SKOnnxInterface {
    /// The single required feature-matrix input name.
    pub(crate) input_name: String,
    /// The model input element type.
    pub(crate) input_element_type: ElemType,
    /// The declared feature count.
    pub(crate) feature_count: usize,
    /// The analysed output head.
    pub(crate) head: SKOnnxPredictionHead,
}

/// Resolve the single feature input and the prediction head.
pub(crate) fn interface(model: &SKOnnxModel) -> Result<SKOnnxInterface, SKOnnxImportError> {
    let graph = &model.model().graph;
    let initializer_names: Vec<&str> = graph
        .initializers
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let required: Vec<&serde_onnx::ir::ValueInfo> = graph
        .inputs
        .iter()
        .filter(|input| !initializer_names.contains(&input.name.as_str()))
        .collect();
    if required.len() != 1 {
        return Err(SKOnnxImportError::UnsupportedOutput {
            detail: format!(
                "expected one feature-matrix input, the graph declares {} required inputs",
                required.len()
            ),
        });
    }
    let input = required[0];
    let (input_element_type, shape) = tensor_signature(input)?;
    if shape.len() != 2 {
        return Err(SKOnnxImportError::ShapeMismatch {
            expected: vec![2],
            found: vec![shape.len()],
        });
    }
    if !matches!(input_element_type, ElemType::Float | ElemType::Double) {
        return Err(SKOnnxImportError::UnsupportedDtype {
            element_type: input_element_type.name().to_string(),
        });
    }
    if shape[1] == 0 {
        return Err(SKOnnxImportError::ShapeMismatch {
            expected: vec![2],
            found: shape,
        });
    }
    let feature_count = shape[1];
    let head = prediction_head(graph)?;
    Ok(SKOnnxInterface {
        input_name: input.name.clone(),
        input_element_type,
        feature_count,
        head,
    })
}

fn prediction_head(
    graph: &serde_onnx::ir::Graph,
) -> Result<SKOnnxPredictionHead, SKOnnxImportError> {
    let mut labels = None;
    let mut float_outputs = Vec::new();
    for output in &graph.outputs {
        let (element_type, shape) = tensor_signature(output)?;
        if element_type == ElemType::Int64 {
            labels = Some((output.name.clone(), shape));
        } else if matches!(element_type, ElemType::Float | ElemType::Double) {
            float_outputs.push((output.name.clone(), element_type, shape));
        } else {
            return Err(SKOnnxImportError::UnsupportedDtype {
                element_type: element_type.name().to_string(),
            });
        }
    }
    if let Some((label_name, _)) = labels {
        if let Some((probabilities, element_type, shape)) = float_outputs.first() {
            if shape.len() != 2 {
                return Err(SKOnnxImportError::UnsupportedOutput {
                    detail: format!("the probability output must be a matrix, got shape {shape:?}"),
                });
            }
            return Ok(SKOnnxPredictionHead::Classification {
                labels: label_name,
                probabilities: probabilities.clone(),
                class_count: shape[1],
                element_type: *element_type,
            });
        }
        return Ok(SKOnnxPredictionHead::LabelsOnly { labels: label_name });
    }
    match float_outputs.as_slice() {
        [(name, element_type, shape)] if shape.len() == 2 => Ok(SKOnnxPredictionHead::Regression {
            output: name.clone(),
            element_type: *element_type,
            width: shape[1],
        }),
        [(name, element_type, _)] => Ok(SKOnnxPredictionHead::Regression {
            output: name.clone(),
            element_type: *element_type,
            width: 1,
        }),
        _ => Err(SKOnnxImportError::UnsupportedOutput {
            detail: "expected one regression output or an integer label plus a probability matrix"
                .to_string(),
        }),
    }
}

fn tensor_signature(
    info: &serde_onnx::ir::ValueInfo,
) -> Result<(ElemType, Vec<usize>), SKOnnxImportError> {
    match info.value_type.as_ref() {
        Some(ValueType::Tensor(tensor)) => {
            let shape = tensor
                .shape
                .as_ref()
                .map(|dimensions| {
                    dimensions
                        .iter()
                        .map(|dimension| dimension.fixed_value().unwrap_or(0).max(0) as usize)
                        .collect()
                })
                .unwrap_or_default();
            let _ = SKOnnxTensorDType::from_element_type(tensor.elem)?;
            Ok((tensor.elem, shape))
        }
        _ => Err(SKOnnxImportError::UnsupportedOutput {
            detail: format!(
                "the value `{}` does not declare a dense tensor type",
                info.name
            ),
        }),
    }
}
