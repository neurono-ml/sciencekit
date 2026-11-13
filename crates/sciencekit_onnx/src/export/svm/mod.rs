//! Support-vector-machine exporters (spec `onnx-export`, requirement
//! "Per-family operators").
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod classification;
mod dispatcher;
mod expansion;
mod kernel;
mod linear;
mod multi_class;
mod regression;

pub(crate) use dispatcher::append_svm;
