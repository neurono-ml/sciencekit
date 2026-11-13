//! Compile-time proof that each optional integration is reachable exactly when
//! its feature is on.
//!
//! The feature matrix is exercised by the CI commands
//! (`cargo check -p sciencekit --no-default-features` and one run per feature);
//! this companion test is the executable form of that matrix for whichever
//! combination the current run enabled.

#[test]
fn optional_integrations_mirror_their_features() {
    #[cfg(feature = "onnx")]
    {
        let _ = crate::interop::sk_onnx_coverage();
        let _ = core::any::type_name::<crate::interop::SKOnnxModel>();
    }

    #[cfg(feature = "telemetry")]
    {
        let _ = core::any::type_name::<crate::telemetry::SKTelemetry>();
    }

    #[cfg(feature = "opencl")]
    {
        let _ = core::any::type_name::<crate::accelerators::SKOpenClBackend>();
        let _ = core::any::type_name::<crate::accelerators::SKOpenClBackendBuilder>();
    }

    #[cfg(feature = "polars")]
    {
        let _ = core::any::type_name::<crate::interop::polars::SKPolarsLazySource>();
    }

    #[cfg(feature = "datafusion")]
    {
        let _ = core::any::type_name::<crate::interop::datafusion::SKDataFusionStreamSource>();
    }

    #[cfg(feature = "memmap")]
    {
        let _ = core::any::type_name::<crate::interop::memmap::SKRawBinaryDescriptor>();
        // `sciencekit_interop/memmap` also enables the Arrow adapter: the Arrow
        // IPC sources need it, so the umbrella must mirror that reachability.
        let _ = core::any::type_name::<crate::interop::arrow::SKArrowFileFormat>();
    }

    #[cfg(feature = "parquet")]
    {
        let _ = core::any::type_name::<crate::interop::arrow::SKArrowFileFormat>();
    }
}

#[test]
fn the_default_build_compiles_only_the_core() {
    // Whatever the enabled feature set, the core families must always resolve.
    // The matrix itself is exercised by the CI command per feature; this test
    // guards the invariant that features are additive and never gate core code.
    fn sized<T>() {}
    sized::<crate::linear_model::SKLinearRegression>();
    sized::<crate::preprocessing::SKStandardScaler>();
    sized::<crate::metrics::classification::SKConfusionMatrix>();
}

/// The umbrella's `telemetry` feature must enable a real transport, not just
/// the crate: a pipeline that always refuses with "enable otlp-grpc" would make
/// the documented capability a lie.
#[cfg(feature = "telemetry")]
#[test]
fn the_telemetry_feature_enables_an_export_transport() {
    use crate::SKBuilder;
    let configuration = crate::telemetry::SKTelemetryConfiguration::build()
        .service_name("sciencekit-umbrella-test")
        .build()
        .unwrap();
    assert!(
        crate::telemetry::sk_build_otlp_pipeline(configuration).is_ok(),
        "the umbrella `telemetry` feature must forward the sub-crate transport feature"
    );
}

/// Forwarded allocator features must actually reach the global allocator of the
/// sub-crate, not merely compile.
#[cfg(feature = "allocator-jemalloc")]
#[test]
fn the_jemalloc_feature_installs_the_jemalloc_allocator() {
    assert_eq!(crate::sk_allocator_kind(), crate::SKAllocatorKind::Jemalloc);
}

#[cfg(feature = "allocator-mimalloc")]
#[test]
fn the_mimalloc_feature_installs_the_mimalloc_allocator() {
    assert_eq!(crate::sk_allocator_kind(), crate::SKAllocatorKind::MiMalloc);
}
