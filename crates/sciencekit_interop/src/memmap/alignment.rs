//! Alignment checks for zero-copy views over mapped bytes.
//!
//! A memory mapping is page-aligned at offset zero, but a payload can start at
//! an arbitrary offset (the `.npy` header is padded to a multiple of 64 bytes,
//! not to the element size). `ndarray` requires aligned references, so every
//! zero-copy view passes through [`sk_aligned_scalar_slice`], which rejects a
//! misaligned payload with a structured error instead of constructing an
//! invalid reference.

use std::mem::{align_of, size_of};
use std::path::Path;

use sciencekit_common::{SKError, SKFloat};

/// Check that `bytes` starts at an address aligned for `F`.
///
/// # Errors
/// Returns [`SKError::MisalignedMappedData`] naming the file and the scalar type
/// when the address is not a multiple of `align_of::<F>()`.
pub(crate) fn sk_check_alignment<F>(bytes: &[u8], path: &Path) -> Result<(), SKError> {
    if bytes.as_ptr() as usize % align_of::<F>() == 0 {
        Ok(())
    } else {
        Err(SKError::MisalignedMappedData {
            path: path.display().to_string(),
            scalar_type: std::any::type_name::<F>(),
            reason: format!(
                "the payload starts at byte {} but {}-byte elements require an address multiple of {}; rewrite the file with a padded header or open it through the casting path, which copies",
                bytes.as_ptr() as usize,
                size_of::<F>(),
                align_of::<F>()
            ),
        })
    }
}

/// Reinterpret aligned bytes as a scalar slice without copying.
///
/// The function is safe to call because every `SKFloat` implementation (`f32`,
/// `f64`) accepts every bit pattern; it validates alignment and length and
/// returns a structured error instead of constructing an invalid reference.
pub(crate) fn sk_scalar_slice<'a, F: SKFloat>(
    bytes: &'a [u8],
    path: &Path,
) -> Result<&'a [F], SKError> {
    sk_check_alignment::<F>(bytes, path)?;
    if bytes.len() % size_of::<F>() != 0 {
        return Err(SKError::MappedLengthMismatch {
            path: path.display().to_string(),
            expected_bytes: bytes.len() as u64,
            actual_bytes: (bytes.len() / size_of::<F>() * size_of::<F>()) as u64,
        });
    }
    // SAFETY: alignment and length are validated above; the mapping outlives
    // the returned slice and is never mutated while it is mapped. The sealed
    // scalar trait only covers `f32` and `f64`, so every bit pattern is valid.
    Ok(unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().cast::<F>(), bytes.len() / size_of::<F>())
    })
}
