//! The estimator-name dispatcher from a neutral model state to an ONNX graph.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxModel, SKOnnxTensorDType, SKOnnxValueInfo};

use super::helpers::unsupported;
use super::linear::append_linear;
use super::metadata::attach_metadata;
use super::preprocessing::append_preprocessing;
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

/// The single graph input every exported model declares.
pub(crate) const SK_ONNX_FEATURE_INPUT: &str = "features";

/// Translate a neutral model state into an ONNX graph.
///
/// Dispatch happens exactly once, on the state's estimator name. A family with
/// no exporter returns `UnsupportedOperation` naming the family instead of a
/// partial graph.
pub fn sk_export_state_to_onnx<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<SKOnnxModel, SKError> {
    if state.number_of_tensors() == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "onnx_model_state",
            reason: format!(
                "the exported `{}` state carries no learned tensors",
                state.metadata().estimator_name()
            ),
        });
    }
    let element_type =
        SKOnnxTensorDType::from_tensor_data_type(state.metadata().scalar_data_type())?;
    let feature_count = state.metadata().feature_count();
    let mut builder = SKOnnxGraphBuilder::new(state.metadata().estimator_name());
    builder.add_input(SKOnnxValueInfo::floating_matrix(
        SK_ONNX_FEATURE_INPUT,
        element_type,
        feature_count,
    )?);
    let output = append_estimator(
        state.metadata().estimator_name(),
        state,
        &mut builder,
        SK_ONNX_FEATURE_INPUT,
        "",
        feature_count,
    )?;
    declare_outputs(&mut builder, &output, element_type)?;
    attach_metadata(&mut builder, state);
    builder.build()
}

/// Append one estimator's operators, dispatching on the family.
pub(crate) fn append_estimator<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
    input_width: usize,
) -> Result<SKOnnxStageOutput, SKError> {
    super::helpers::require_feature_count(state, input_width, estimator_name)?;
    if let Some(output) = append_preprocessing(estimator_name, state, builder, input, namespace)? {
        return Ok(output);
    }
    if let Some(output) = append_linear(estimator_name, state, builder, input, namespace)? {
        return Ok(output);
    }
    Err(unsupported(
        estimator_name,
        "no exporter is registered for this family",
    ))
}

/// Declare the graph outputs a stage produced.
pub(crate) fn declare_outputs(
    builder: &mut SKOnnxGraphBuilder,
    output: &SKOnnxStageOutput,
    element_type: SKOnnxTensorDType,
) -> Result<(), SKError> {
    match &output.kind {
        SKOnnxStageKind::Transformed => {
            builder.add_output(SKOnnxValueInfo::new(
                output.tensor.clone(),
                element_type,
                vec![
                    crate::graph::SKOnnxDimension::batch(),
                    crate::graph::SKOnnxDimension::Fixed(output.width),
                ],
            )?);
        }
        SKOnnxStageKind::Regression => {
            builder.add_output(SKOnnxValueInfo::new(
                output.tensor.clone(),
                element_type,
                vec![
                    crate::graph::SKOnnxDimension::batch(),
                    crate::graph::SKOnnxDimension::Fixed(output.width),
                ],
            )?);
        }
        SKOnnxStageKind::Classification {
            label_tensor,
            probability_tensor,
            class_count,
        } => {
            builder.add_output(SKOnnxValueInfo::new(
                label_tensor.clone(),
                SKOnnxTensorDType::Int64,
                vec![crate::graph::SKOnnxDimension::batch()],
            )?);
            builder.add_output(SKOnnxValueInfo::new(
                probability_tensor.clone(),
                element_type,
                vec![
                    crate::graph::SKOnnxDimension::batch(),
                    crate::graph::SKOnnxDimension::Fixed(*class_count),
                ],
            )?);
        }
    }
    Ok(())
}
