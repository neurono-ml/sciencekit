//! The double-buffer memory guard applied to every streamed batch.

use crate::SKError;

/// Check that two resident copies of one streaming batch fit in the memory
/// available to the process.
///
/// The guard is raised when a batch is read, before it reaches the update
/// closure, so the algorithm never sees a block that would not fit twice. The
/// caller passes the batch's own [`byte_size`](crate::SKFeatureBlock::byte_size)
/// — the stored buffers for a sparse block, not its dense `rows × columns`
/// figure — so a sparse stream is accepted whenever its actual representation
/// fits even if a hypothetical dense copy would not.
pub fn sk_check_batch_buffer_guard(
    batch_byte_size: usize,
    available_memory_bytes: u64,
) -> Result<(), SKError> {
    let batch_bytes = batch_byte_size as u64;
    let required_bytes = batch_bytes.saturating_mul(2);
    if required_bytes > available_memory_bytes {
        return Err(SKError::BatchBufferOversize {
            batch_bytes,
            required_bytes,
            available_bytes: available_memory_bytes,
        });
    }
    Ok(())
}
