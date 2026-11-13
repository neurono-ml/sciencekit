//! ONNX export for sciencekit (spec `onnx-export`).
//!
//! A fitted model already describes itself through the neutral
//! [`sciencekit_common::SKExportable`] contract: metadata plus stable, named
//! tensors. This crate is a **translator** from that state to an ONNX
//! `ModelProto`; it is not a second model format. Learned tensors become graph
//! initializers, the estimator identity becomes model metadata and the operators
//! come from the documented subset in [`sk_onnx_coverage`].
//!
//! This is a pure dispatcher module: `lib.rs` only declares and re-exports
//! submodules.

mod export;
mod graph;

#[cfg(test)]
mod acceptance_tests;

pub use export::{
    SKOnnxFamilyCoverage, SKToOnnx, sk_export_state_to_onnx, sk_onnx_coverage,
    sk_onnx_supported_families,
};
pub use graph::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_IR_VERSION, SK_ONNX_ML_DOMAIN, SK_ONNX_ML_TREE_OPSET,
    SKOnnxDimension, SKOnnxGraphBuilder, SKOnnxModel, SKOnnxNode, SKOnnxTensorDType,
    SKOnnxValueInfo,
};
