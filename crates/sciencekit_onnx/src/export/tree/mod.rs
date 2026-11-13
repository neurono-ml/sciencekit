//! Decision-tree and random-forest exporters through the ONNX-ML tree
//! ensembles (spec `onnx-export`, requirement "Per-family operators").
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classifier;
mod dispatcher;
mod ensemble;
mod layout;
mod regressor;

pub(crate) use dispatcher::append_tree;
