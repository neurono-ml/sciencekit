//! The hand-checkable identity graph, validated by the reference runtime.

use serde_json::json;

use crate::acceptance_tests::{assert_regressor_interface, write_fixture};
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

#[test]
fn identity_graph_matches_the_reference_runtime() {
    let mut builder = SKOnnxGraphBuilder::new("identity");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("features", SKOnnxTensorDType::Float, 2)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["features".to_string()],
            vec!["transformed".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("transformed", SKOnnxTensorDType::Float, 2)
                .expect("a valid output"),
        );
    let model = builder.build().expect("the identity graph is valid");
    assert_regressor_interface(&model, 2, 2, SKOnnxTensorDType::Float);
    let input = vec![vec![1.5_f32, -2.0], vec![0.0, 3.25]];
    write_fixture(
        "identity",
        &model,
        "features",
        json!(input),
        json!({"transformed": input}),
        1e-6,
    );
}
