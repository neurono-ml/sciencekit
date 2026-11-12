//! Bounded-memory accounting for the materialized intermediates.
//!
//! Both the one-shot materialization path and the batch finalizer keep owned
//! blocks resident. They share one rule: a block that would push the resident
//! total past the configured budget is rejected with
//! [`SKError::BatchBufferOversize`] before it is stored, so the pipeline fails
//! with a structured error instead of exhausting memory.

use sciencekit_common::SKError;

/// Charge `additional` bytes against `budget`, returning the new total.
pub(super) fn sk_charge_budget(
    used_bytes: usize,
    additional_bytes: usize,
    budget_bytes: usize,
) -> Result<usize, SKError> {
    let required_bytes = used_bytes.saturating_add(additional_bytes);
    if required_bytes > budget_bytes {
        return Err(SKError::BatchBufferOversize {
            batch_bytes: additional_bytes as u64,
            required_bytes: required_bytes as u64,
            available_bytes: budget_bytes as u64,
        });
    }
    Ok(required_bytes)
}
