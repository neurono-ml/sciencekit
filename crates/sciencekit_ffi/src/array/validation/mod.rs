//! Descriptor validation that runs before any element is read (spec
//! `ffi-abi`). Carries no implementation logic.

mod error;
mod functions;
mod shape;

pub use error::SKFfiArrayError;
pub use functions::{sk_ffi_validate_contiguous_strides, sk_ffi_validate_length};
pub use shape::SKFfiArrayShape;
