//! Raw-pointer to safe-value conversions. This is the only module allowed to
//! dereference pointers coming from C. Carries no logic beyond validation and
//! ownership transfer.

mod c_string;
mod descriptor;
mod output;

pub(crate) use c_string::{sk_ffi_handle_identifier, sk_ffi_path_from_c};
pub(crate) use descriptor::sk_ffi_validate_descriptor;
pub(crate) use output::{sk_ffi_publish_output, sk_ffi_release_output};

#[cfg(test)]
#[path = "../boundary_tests.rs"]
mod boundary_tests;
