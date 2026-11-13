//! Compile-time proof that the umbrella exposes every family module.
//!
//! A missing or renamed family module fails the build, so this companion test
//! is the executable form of the "Sub-crate modules remain addressable"
//! scenario.

#[test]
fn every_family_module_is_addressable() {
    #[allow(unused_imports)]
    use crate::{
        accelerators, cluster, decomposition, ensemble, ffi, impute, interop, linear_model, math,
        metrics, model_selection, neighbors, outlier, pipeline, preprocessing, svm, telemetry,
        tree,
    };
}
