//! The `interop` family: dataset interop, model formats and ONNX.
//!
//! Every item here resolves to `sciencekit_interop` or `sciencekit_onnx`. The
//! engine adapters (`arrow`, `polars`, `datafusion`, `memmap`, `parquet`) are
//! feature-gated in the sub-crate; the umbrella mirrors those flags so enabling
//! one makes the adapter reachable here too.

#[cfg(feature = "parquet")]
pub use sciencekit_interop::arrow;
#[cfg(feature = "datafusion")]
pub use sciencekit_interop::datafusion;
#[cfg(feature = "memmap")]
pub use sciencekit_interop::memmap;
#[cfg(feature = "polars")]
pub use sciencekit_interop::polars;
pub use sciencekit_interop::{
    batch, bridge, columns, json, missing, native, observability, safetensors, schema, selection,
    source, tabular, views,
};
#[cfg(feature = "onnx")]
pub use sciencekit_onnx::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_IR_VERSION, SK_ONNX_ML_DOMAIN, SK_ONNX_ML_TREE_OPSET,
    SK_ONNX_SUPPORTED_ML_OPERATORS, SK_ONNX_SUPPORTED_OPERATORS, SKOnnxDimension,
    SKOnnxFamilyCoverage, SKOnnxGraphBuilder, SKOnnxImportError, SKOnnxModel, SKOnnxNativeError,
    SKOnnxNativeImporter, SKOnnxNode, SKOnnxPredictor, SKOnnxPredictorBuilder, SKOnnxTensorDType,
    SKOnnxValueInfo, SKToOnnx, sk_export_state_to_onnx, sk_onnx_coverage,
    sk_onnx_supported_families, sk_onnx_supports_operator,
};
