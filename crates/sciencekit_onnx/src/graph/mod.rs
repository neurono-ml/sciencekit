//! The ONNX graph builder and its protobuf encoding (spec `onnx-export`,
//! requirement "Graph builder").
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod model;
mod opset;
mod proto;
mod tensor_data_type;
mod value_info;

pub use model::SKOnnxModel;
pub use opset::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_IR_VERSION, SK_ONNX_ML_DOMAIN, SK_ONNX_ML_TREE_OPSET,
};
pub use tensor_data_type::SKOnnxTensorDType;
pub use value_info::{SKOnnxDimension, SKOnnxValueInfo};

#[cfg(test)]
mod proto_tests;
