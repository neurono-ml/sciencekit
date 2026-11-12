//! Tests for the backend-kind vocabulary (spec `backend-selection`, spec
//! `gpu-compute-contracts`).
//!
//! The kind is a bounded value recorded on every operation span, so its display
//! names are a public contract and its parse surface round-trips them.

use std::str::FromStr;

use super::SKBackendKind;

/// Every backend kind round-trips through its documented display name.
#[test]
fn display_names_round_trip_through_parsing() {
    for kind in SKBackendKind::ALL {
        let rendered = kind.to_string();
        let parsed = SKBackendKind::from_str(&rendered)
            .unwrap_or_else(|error| panic!("`{rendered}` must parse back: {error}"));
        assert_eq!(parsed, kind, "`{rendered}` must parse to its own kind");
    }
}

/// The display names are the documented, stable spellings.
#[test]
fn display_names_are_the_documented_spellings() {
    let names: Vec<String> = SKBackendKind::ALL
        .iter()
        .map(|kind| kind.to_string())
        .collect();
    assert_eq!(
        names,
        vec![
            "faer",
            "matrixmultiply",
            "ndarray-linalg",
            "opencl",
            "cuda",
            "rocm",
        ]
    );
}

/// An unknown name is rejected instead of silently mapping to a default.
#[test]
fn parsing_rejects_an_unknown_name() {
    let error = SKBackendKind::from_str("vulkan").expect_err("`vulkan` is not a backend kind");
    let message = error.to_string();
    assert!(
        message.contains("vulkan"),
        "the error must name the rejected value, got `{message}`"
    );
}

/// The three device kinds report themselves as devices; the CPU kinds do not.
#[test]
fn device_kinds_are_distinguished_from_host_kinds() {
    assert!(SKBackendKind::OpenCl.is_device());
    assert!(SKBackendKind::Cuda.is_device());
    assert!(SKBackendKind::Rocm.is_device());
    assert!(!SKBackendKind::Faer.is_device());
    assert!(!SKBackendKind::MatrixMultiply.is_device());
    assert!(!SKBackendKind::NdArrayLinalg.is_device());
    assert!(SKBackendKind::Faer.is_host());
    assert!(!SKBackendKind::OpenCl.is_host());
}

/// The documented default backend is the pure-Rust host backend.
#[test]
fn the_default_kind_is_the_pure_rust_host_backend() {
    assert_eq!(SKBackendKind::default(), SKBackendKind::Faer);
}
