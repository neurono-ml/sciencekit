//! Default backend selection for the current build configuration (spec
//! `blas-interface`).
//!
//! [`sk_default_math_backend`] returns the backend configured for the crate
//! build: the pure-Rust [`SKFaerBackend`] by default, or the opt-in
//! `ndarray-linalg` backend when the `blas-backend` feature is enabled.

use sciencekit_common::SKFloat;

use super::kernel::SKMathBackend;

#[cfg(not(feature = "blas-backend"))]
use super::faer_backend::SKFaerBackend;
#[cfg(feature = "blas-backend")]
use super::ndarray_backend::SKNdArrayLinalgBackend;

/// The default math backend for the current build configuration.
///
/// With the `blas-backend` feature enabled this returns the `ndarray-linalg`
/// backend; otherwise the pure-Rust [`SKFaerBackend`]. Execution planning calls
/// this to route dense algebra.
#[cfg(feature = "blas-backend")]
pub fn sk_default_math_backend<F: SKFloat>() -> Box<dyn SKMathBackend<F>>
where
    SKNdArrayLinalgBackend: SKMathBackend<F>,
{
    Box::new(SKNdArrayLinalgBackend::new())
}

/// The default math backend for the current build configuration.
///
/// With the `blas-backend` feature enabled this returns the `ndarray-linalg`
/// backend; otherwise the pure-Rust [`SKFaerBackend`]. Execution planning calls
/// this to route dense algebra.
#[cfg(not(feature = "blas-backend"))]
pub fn sk_default_math_backend<F: SKFloat>() -> Box<dyn SKMathBackend<F>>
where
    SKFaerBackend: SKMathBackend<F>,
{
    Box::new(SKFaerBackend::new())
}
