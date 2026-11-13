//! Mapping an ONNX preprocessing graph back to scaler state
//! (spec `native-model-import`, requirement "ONNX native import").
//!
//! Pure dispatcher: declares the mapping and re-exports its entry point.

mod families;
mod state;

pub(crate) use state::state_from_scaler;
