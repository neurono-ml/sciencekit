//! The execution plan shared by the test transformers.

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKExecutionPlan};

pub(crate) fn sk_test_execution_plan() -> SKExecutionPlan {
    SKExecutionPlan {
        mode: SKExecutionMode::OutOfCoreStreaming,
        parallelism: 1,
        batch_size: None,
        buffer_depth: 1,
        available_memory_bytes: u64::MAX,
        backend: SKBackendKind::Faer,
        fallback_reason: None,
    }
}

