//! The sealed floating-point bound and its private seal.

use num_traits::Float;

/// The single numeric bound accepted by continuous generic APIs.
///
/// Implemented only by the standard floating-point types `f32` and `f64`.
/// Sealed via a private supertrait — external crates cannot implement it.
pub trait SKFloat: Float + Send + Sync + 'static + private::SKFloatSealed {}

impl SKFloat for f32 {}
impl SKFloat for f64 {}

/// Private sealing module. Kept out of the public API so that only the
/// implementations declared here can ever satisfy [`SKFloat`].
mod private {
    /// Hidden supertrait whose only implementations are the supported floats.
    #[doc(hidden)]
    pub trait SKFloatSealed {}

    impl SKFloatSealed for f32 {}
    impl SKFloatSealed for f64 {}
}
