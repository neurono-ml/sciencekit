//! Uniform prediction over the reconstructed native models
//! (spec `native-model-import`, requirement "Safetensors native import").
//!
//! One facade hides the concrete model type without erasing it: every method
//! delegates to the model's existing prediction trait, so the imported model
//! keeps its own numerics, observability span and dense/sparse handling.
//!
//! Pure dispatcher: declares the surface and its delegation, no logic here.

mod delegation;
mod surface;
