//! Array output release entry point.

use crate::array::SKFfiArray;
use crate::boundary::guard::sk_ffi_guard_status;
use crate::boundary::pointers::sk_ffi_release_output;

/// Release an output array produced by `sk_ffi_predict` or `sk_ffi_transform`.
///
/// The descriptor is zeroed, so releasing it twice is a no-op. Never call this
/// on a borrowed input descriptor.
///
/// # Safety
///
/// `array` must be null or point to a writable descriptor produced by this
/// library.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_array_free(array: *mut SKFfiArray) -> crate::SKFfiStatus {
    sk_ffi_guard_status(|| {
        // SAFETY: the caller guarantees a writable output descriptor.
        unsafe { sk_ffi_release_output(array) }
    })
}
