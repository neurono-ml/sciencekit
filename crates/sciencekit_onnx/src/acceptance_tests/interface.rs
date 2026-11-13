//! Interface assertions shared by the exporter tests.

use crate::graph::{SKOnnxModel, SKOnnxTensorDType};

/// Assert the declared matrix input and the single regression output.
pub(crate) fn assert_regressor_interface(
    model: &SKOnnxModel,
    feature_count: usize,
    output_width: usize,
    element_type: SKOnnxTensorDType,
) {
    assert_eq!(model.inputs().len(), 1);
    let input = &model.inputs()[0];
    assert_eq!(input.feature_count(), Some(feature_count));
    assert_eq!(input.element_type(), element_type);
    assert_eq!(model.outputs().len(), 1);
    let output = &model.outputs()[0];
    assert_eq!(output.feature_count(), Some(output_width));
    assert_eq!(output.element_type(), element_type);
}

/// Assert the declared matrix input plus the label and score outputs.
#[allow(dead_code)]
pub(crate) fn assert_classifier_interface(
    model: &SKOnnxModel,
    feature_count: usize,
    class_count: usize,
    element_type: SKOnnxTensorDType,
) {
    assert_eq!(model.inputs().len(), 1);
    assert_eq!(model.inputs()[0].feature_count(), Some(feature_count));
    let labels = model
        .outputs()
        .iter()
        .find(|output| output.element_type() == SKOnnxTensorDType::Int64)
        .expect("the graph declares the label output");
    assert_eq!(labels.element_type(), SKOnnxTensorDType::Int64);
    assert_eq!(labels.shape().len(), 1);
    let scores = model
        .outputs()
        .iter()
        .find(|output| output.element_type() == element_type)
        .expect("the graph declares the probability output");
    assert_eq!(scores.feature_count(), Some(class_count));
    assert_eq!(scores.element_type(), element_type);
}
