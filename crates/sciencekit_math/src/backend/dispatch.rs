//! Default backend selection for the current build configuration (spec
//! `blas-interface`).
//!
//! [`sk_default_math_backend`] returns the backend configured for the crate
//! build: the pure-Rust [`SKFaerBackend`] by default, or the opt-in
//! `ndarray-linalg` backend when the `blas-backend` feature is enabled.

use sciencekit_common::{SKBackendKind, SKFloat};

use super::faer_backend::{SKFaerBackend, SKMatrixMultiplyBackend};
use super::kernel::SKMathBackend;

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

/// Selects the concrete [`SKMathBackend`] for a configured [`SKBackendKind`].
///
/// Implemented for the sealed scalar types, so an algorithm crate can dispatch
/// a kind to a backend **without naming the feature-gated backend types** — the
/// `blas-backend` configuration is resolved here, inside the crate that owns the
/// feature, and never leaks into a generic bound. This is the per-operation
/// dispatch of the `backend-selection` capability: the kind is turned into a
/// concrete backend at operation time, never in `build()`.
pub trait SKBackendSelection: SKFloat {
    /// The concrete backend for `kind` in this build configuration.
    fn select_math_backend(kind: SKBackendKind) -> Box<dyn SKMathBackend<Self>>;
}

/// The kind→backend mapping for one concrete scalar, with the feature-gated
/// `ndarray-linalg` arm resolved inside the crate that owns the feature.
macro_rules! impl_backend_selection {
    ($float:ty) => {
        impl SKBackendSelection for $float {
            fn select_math_backend(kind: SKBackendKind) -> Box<dyn SKMathBackend<$float>> {
                match kind {
                    SKBackendKind::Faer => Box::new(SKFaerBackend::new()),
                    SKBackendKind::MatrixMultiply => Box::new(SKMatrixMultiplyBackend::new()),
                    // Without the `blas-backend` feature there is no
                    // `ndarray-linalg` backend to build; the pure-Rust default
                    // is used and the configured kind is still what the caller
                    // records in its span.
                    #[cfg(feature = "blas-backend")]
                    SKBackendKind::NdArrayLinalg => Box::new(SKNdArrayLinalgBackend::new()),
                    #[cfg(not(feature = "blas-backend"))]
                    SKBackendKind::NdArrayLinalg => Box::new(SKFaerBackend::new()),
                }
            }
        }
    };
}

impl_backend_selection!(f64);
impl_backend_selection!(f32);

/// The concrete backend for a configured kind, for the sealed scalar types.
///
/// The free-function form of [`SKBackendSelection::select_math_backend`].
pub fn sk_math_backend_for_kind<F: SKBackendSelection>(
    kind: SKBackendKind,
) -> Box<dyn SKMathBackend<F>> {
    F::select_math_backend(kind)
}
