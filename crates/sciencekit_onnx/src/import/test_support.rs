//! Graph fixtures shared by the importer tests.

use crate::graph::{
    SKOnnxDimension, SKOnnxGraphBuilder, SKOnnxModel, SKOnnxNode, SKOnnxTensorDType,
    SKOnnxValueInfo,
};

/// An affine regressor: `y = X · coefficients + intercept`.
pub(crate) fn affine_regressor_model() -> SKOnnxModel {
    let mut builder = SKOnnxGraphBuilder::new("affine");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_floating_initializer("coefficients", vec![2, 1], &[2.0_f64, -1.0])
        .add_floating_initializer("intercept", vec![1, 1], &[0.5_f64])
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string(), "coefficients".to_string()],
            vec!["product".to_string()],
        ))
        .add_node(SKOnnxNode::new(
            "Add",
            vec!["product".to_string(), "intercept".to_string()],
            vec!["prediction".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("prediction", SKOnnxTensorDType::Double, 1)
                .expect("a valid output"),
        );
    builder.build().expect("the fixture graph is valid")
}

/// A three-class softmax classifier with a label output.
pub(crate) fn softmax_classifier_model() -> SKOnnxModel {
    let mut builder = SKOnnxGraphBuilder::new("classifier");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_floating_initializer("weights", vec![2, 3], &[1.0_f64, 0.0, -1.0, 0.0, 1.0, 0.0])
        .add_floating_initializer("bias", vec![1, 3], &[0.0_f64, 0.0, 0.0])
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string(), "weights".to_string()],
            vec!["scores".to_string()],
        ))
        .add_node(SKOnnxNode::new(
            "Add",
            vec!["scores".to_string(), "bias".to_string()],
            vec!["logits".to_string()],
        ))
        .add_node(
            SKOnnxNode::new(
                "Softmax",
                vec!["logits".to_string()],
                vec!["probabilities".to_string()],
            )
            .attribute_integer("axis", 1),
        )
        .add_node(
            SKOnnxNode::new(
                "ArgMax",
                vec!["probabilities".to_string()],
                vec!["label".to_string()],
            )
            .attribute_integer("axis", 1)
            .attribute_integer("keepdims", 0),
        )
        .add_output(
            SKOnnxValueInfo::new(
                "label",
                SKOnnxTensorDType::Int64,
                vec![SKOnnxDimension::batch()],
            )
            .expect("a valid label output"),
        )
        .add_output(
            SKOnnxValueInfo::floating_matrix("probabilities", SKOnnxTensorDType::Double, 3)
                .expect("a valid probability output"),
        );
    builder.build().expect("the fixture graph is valid")
}

/// An identity graph over a fixed-width `f32` matrix.
pub(crate) fn identity_model_f32() -> SKOnnxModel {
    let mut builder = SKOnnxGraphBuilder::new("identity_f32");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Float, 2)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["X".to_string()],
            vec!["out".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Float, 2)
                .expect("a valid output"),
        );
    builder.build().expect("the fixture graph is valid")
}

/// An identity graph over a fixed-width matrix.
pub(crate) fn identity_model() -> SKOnnxModel {
    let mut builder = SKOnnxGraphBuilder::new("identity");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["X".to_string()],
            vec!["out".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 2)
                .expect("a valid output"),
        );
    builder.build().expect("the fixture graph is valid")
}
