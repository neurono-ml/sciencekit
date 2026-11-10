//! Pure dispatcher: the C array descriptor, its validation and the owned
//! output buffer. Carries no implementation logic.

mod definition;
mod owned;
mod validation;

pub use definition::{SKFfiArray, SKFfiDataType};
pub use owned::{SKFfiOwnedArray, SKFfiOwnedValues};
pub use validation::{
    SKFfiArrayError, SKFfiArrayShape, sk_ffi_validate_contiguous_strides, sk_ffi_validate_length,
};

#[cfg(test)]
mod array_tests;
