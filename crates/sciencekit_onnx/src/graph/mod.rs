//! The ONNX graph builder, the shared model handle and their interface types
//! (spec `onnx-export`, requirement "Graph builder"; spec `onnx-import`,
//! requirement "Shared ONNX IR").
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod model;
mod node;
mod opset;
mod tensor_data_type;
mod value_info;

pub use builder::SKOnnxGraphBuilder;
pub use model::SKOnnxModel;
pub use node::SKOnnxNode;
pub use opset::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_IR_VERSION, SK_ONNX_ML_DOMAIN, SK_ONNX_ML_TREE_OPSET,
};
pub use tensor_data_type::SKOnnxTensorDType;
pub use value_info::{SKOnnxDimension, SKOnnxValueInfo};

#[cfg(test)]
mod graph_tests;
