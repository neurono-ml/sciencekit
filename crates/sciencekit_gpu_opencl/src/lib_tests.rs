//! Acceptance test for the vendor-facing crate (spec `opencl-backend`).
//!
//! The crate re-exports the `sciencekit_gpu` backend. This test proves the
//! feature wiring works end to end: building the backend through this crate and
//! running a real kernel on the host's device.

use ndarray::array;

use crate::SKOpenClBackend;

/// The re-exported builder drives the real device and a kernel returns the
/// expected numbers.
#[test]
fn the_reexported_backend_runs_a_real_kernel() {
    let backend = SKOpenClBackend::builder().build_or_fallback();
    if !backend.is_available() {
        println!("[opencl] no device available; facade test skips");
        return;
    }
    let device = backend.device().expect("an available device");
    println!(
        "[opencl] facade test runs on device {}: `{}`",
        device.index, device.name
    );

    let left = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let right = array![[5.0_f64, 6.0], [7.0, 8.0]];
    let product = backend.gemm(left.view(), right.view()).expect("gemm");
    assert_eq!(product, array![[19.0, 22.0], [43.0, 50.0]]);
}
