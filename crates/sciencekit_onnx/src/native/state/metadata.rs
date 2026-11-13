//! The sciencekit metadata entries an exported ONNX graph carries.

use crate::graph::SKOnnxModel;

/// The estimator identity copied from a sciencekit export.
#[derive(Debug, Clone)]
pub(crate) struct SKOnnxNativeMetadata {
    pub(crate) estimator_name: String,
    pub(crate) format_version: u32,
    pub(crate) feature_count: usize,
    pub(crate) class_labels: Option<Vec<String>>,
}

impl SKOnnxNativeMetadata {
    /// Read the metadata, or `None` when the graph was not exported by sciencekit.
    pub(crate) fn read(model: &SKOnnxModel) -> Option<Self> {
        let entries = model.metadata();
        let estimator_name = entries.get("sciencekit.estimator_name")?.clone();
        let format_version = entries
            .get("sciencekit.format_version")
            .and_then(|value| value.parse().ok())
            .unwrap_or_default();
        let feature_count = entries
            .get("sciencekit.feature_count")
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| feature_count(model).unwrap_or_default());
        let number_of_classes = entries
            .get("sciencekit.number_of_classes")
            .and_then(|value| value.parse::<usize>().ok());
        let class_labels = number_of_classes.map(|count| {
            (0..count)
                .map(|index| {
                    entries
                        .get(&format!("sciencekit.class_label.{index}"))
                        .cloned()
                        .unwrap_or_else(|| index.to_string())
                })
                .collect()
        });
        Some(Self {
            estimator_name,
            format_version,
            feature_count,
            class_labels,
        })
    }
}

/// The feature width the single required input declares.
pub(crate) fn feature_count(model: &SKOnnxModel) -> Option<usize> {
    model
        .inputs()
        .iter()
        .find_map(|input| input.feature_count())
        .or_else(|| {
            model
                .inputs()
                .first()
                .and_then(|input| input.shape().last())
                .and_then(|dimension| match dimension {
                    crate::graph::SKOnnxDimension::Fixed(width) => Some(*width),
                    crate::graph::SKOnnxDimension::Symbolic(_) => None,
                })
        })
}
