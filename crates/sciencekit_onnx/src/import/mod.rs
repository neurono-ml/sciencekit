//! ONNX import: decode a model, validate its interface and run it as a generic
//! predictor (spec `onnx-import`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items; every implementation lives in its own file.



#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod classifier_tests;
#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod graph_validation_tests;
#[cfg(test)]
mod interpreter_tests;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod operand_tests;
#[cfg(test)]
mod observability_tests;
#[cfg(test)]
mod operators_tests;
#[cfg(test)]
mod predictor_tests;
#[cfg(test)]
mod round_trip_tests;
#[cfg(test)]
mod shape_validation_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod validation_tests;

mod attributes;
mod decode;
mod dispatch;
mod elementwise;
mod error;
mod initializers;
mod interface;
mod interpreter;
mod matrix;
mod model;
mod predictor;
mod predictor_builder;
mod predictor_traits;
mod reductions;
mod selection;
mod shape_ops;
mod tree_ensemble;
mod tree_walk;
mod value;
pub use error::SKOnnxImportError;
pub use interpreter::{
    SK_ONNX_SUPPORTED_ML_OPERATORS, SK_ONNX_SUPPORTED_OPERATORS, sk_onnx_supports_operator,
};
pub use predictor::SKOnnxPredictor;
pub use predictor_builder::SKOnnxPredictorBuilder;
