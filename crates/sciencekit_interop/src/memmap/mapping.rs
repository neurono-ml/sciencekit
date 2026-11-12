//! Read-only mappings and the structured opening errors shared by every source.

use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;
use sciencekit_common::SKError;

/// Classify an operating-system error raised while opening a mapped file.
pub(crate) fn sk_classify_open_error(path: &Path, error: io::Error) -> SKError {
    match error.kind() {
        io::ErrorKind::NotFound => SKError::MissingMappedFile {
            path: path.display().to_string(),
        },
        io::ErrorKind::PermissionDenied => SKError::MappedFilePermissionDenied {
            path: path.display().to_string(),
            reason: error.to_string(),
        },
        _ => SKError::Io(error),
    }
}

/// Open a file for read-only mapping, classifying the failure.
pub(crate) fn sk_open_file(path: &Path) -> Result<File, SKError> {
    File::open(path).map_err(|error| sk_classify_open_error(path, error))
}

/// Map a file read-only after its length has been validated.
///
/// # Safety
/// The caller must have validated the file length against the declared layout;
/// the mapping is never written to and lives as long as the owning source.
pub(crate) fn sk_map_read_only(file: &File, path: &Path) -> Result<Mmap, SKError> {
    // SAFETY: the mapping is read-only. Callers validate the length before
    // exposing any view, so a truncated file surfaces as an opening error
    // rather than a fault on access.
    unsafe { Mmap::map(file) }.map_err(|error| sk_classify_open_error(path, error))
}
