//! Mapping an ONNX-ML tree ensemble to tree/forest state
//! (spec `native-model-import`, requirement "ONNX native import").
//!
//! Pure dispatcher: declares the mapping pieces and re-exports the entry point.

mod encode;
mod rebuild;
mod state;

pub(crate) use state::state_from_trees;
