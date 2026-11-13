//! The ONNX opset and IR versions the exporter targets.

/// The default ONNX operator-set version.
///
/// Opset 17 is widely supported by every maintained runtime and carries all the
/// standard operators this exporter emits. The ONNX-ML operator set travels as a
/// separate import entry.
pub const SK_ONNX_DEFAULT_OPSET: i64 = 17;

/// The ONNX model IR version paired with [`SK_ONNX_DEFAULT_OPSET`].
pub const SK_ONNX_IR_VERSION: i64 = 8;

/// The ONNX-ML operator-set domain the tree ensembles and label encoders use.
pub const SK_ONNX_ML_DOMAIN: &str = "ai.onnx.ml";

/// The ONNX-ML operator-set version.
///
/// Version 3 is the first that defines `TreeEnsembleClassifier`/`TreeEnsembleRegressor`
/// with the attribute names this exporter writes (`nodes_*`, `class_*`, `target_*`).
pub const SK_ONNX_ML_TREE_OPSET: i64 = 3;
