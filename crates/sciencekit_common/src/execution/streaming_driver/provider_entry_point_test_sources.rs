//! Shared fixtures for the provider entry point tests.

use crate::batching::provider::provider_test_sources::FiniteSource;
use crate::execution::{SKExecutionMode, SKExecutionPlan};

/// A finite source of one-row batches, shared with the provider conformance
/// suite so both families exercise the same fixture.
pub(super) type RecordingSource<F> = FiniteSource<F>;

pub(super) fn streaming_plan() -> SKExecutionPlan {
    SKExecutionPlan {
        mode: SKExecutionMode::OutOfCoreStreaming,
        parallelism: 2,
        batch_size: Some(1),
        buffer_depth: 1,
    }
}
