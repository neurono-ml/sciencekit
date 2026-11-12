//! Structured error construction for streaming pipeline failures.
//!
//! Stage failures carry the failing stage's index so a consumer can tell which
//! step rejected a batch; the central taxonomy is shared with every other
//! crate, so the wrapper preserves the original error's message inside a
//! [`SKError::Conversion`].

use sciencekit_common::SKError;

/// Wrap a stage error with the zero-based index of the failing stage.
pub(super) fn sk_stage_failure(stage_index: usize, error: SKError) -> SKError {
    SKError::Conversion(format!(
        "streaming pipeline stage {stage_index} failed: {error}"
    ))
}

/// Validate the configured epoch count.
pub(super) fn sk_validate_epochs(epochs: usize) -> Result<(), SKError> {
    if epochs == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "epochs",
            reason: "a streaming fit requires at least one epoch".to_string(),
        });
    }
    Ok(())
}

/// Validate the configured memory budget.
pub(super) fn sk_validate_memory_budget(memory_budget_bytes: usize) -> Result<(), SKError> {
    if memory_budget_bytes == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "memory_budget_bytes",
            reason: "a streaming pipeline requires a non-zero memory budget".to_string(),
        });
    }
    Ok(())
}

/// Report a one-shot provider asked for more than one epoch.
pub(super) fn sk_reject_one_shot_multiple_epochs(epochs: usize) -> SKError {
    SKError::UnsupportedOperation {
        operation: "streaming_pipeline_fit",
        reason: format!(
            "a non-reopenable source provider cannot replay {epochs} epochs; wrap the source in an SKReopenableSource to fit more than one epoch"
        ),
    }
}
