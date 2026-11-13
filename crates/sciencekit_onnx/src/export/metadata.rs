//! Model metadata entries derived from the neutral state (spec `onnx-export`,
//! requirement "Metadata").

use sciencekit_common::{SKFloat, SKModelState};

use crate::graph::SKOnnxGraphBuilder;

/// Copy the neutral estimator metadata onto the ONNX model.
///
/// The class table is written one entry per index so a label containing any
/// character still round-trips; the feature count lets a consumer validate the
/// declared input width without guessing.
pub(crate) fn attach_metadata<F: SKFloat>(
    builder: &mut SKOnnxGraphBuilder,
    state: &SKModelState<F>,
) {
    let metadata = state.metadata();
    builder
        .add_metadata("sciencekit.estimator_name", metadata.estimator_name())
        .add_metadata(
            "sciencekit.format_version",
            metadata.format_version().to_string(),
        )
        .add_metadata(
            "sciencekit.scalar_data_type",
            metadata.scalar_data_type().as_str(),
        )
        .add_metadata(
            "sciencekit.feature_count",
            metadata.feature_count().to_string(),
        );
    if let Some(labels) = metadata.class_labels() {
        builder.add_metadata("sciencekit.number_of_classes", labels.len().to_string());
        for (index, label) in labels.iter().enumerate() {
            builder.add_metadata(format!("sciencekit.class_label.{index}"), label);
        }
    }
}
