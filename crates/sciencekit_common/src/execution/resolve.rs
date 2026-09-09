//! Resolution of an execution plan from an intent and a context.

use crate::SKError;

use super::context::SKExecutionContext;
use super::modes::{SKAccessPattern, SKExecutionMode};
use super::plan::SKExecutionPlan;

/// Resolve an execution plan from an intent and a context.
///
/// Pure and deterministic: the same `(intent, context)` always yields the same
/// plan. Automatic intent never produces an incompatibility error; an explicit
/// mode incompatible with the declared access pattern does.
pub fn sk_resolve_execution_plan(
    intent: SKExecutionMode,
    context: &SKExecutionContext,
) -> Result<SKExecutionPlan, SKError> {
    let parallelism = context.cpu_cores.max(1);
    match intent {
        SKExecutionMode::Automatic => {
            let fits_in_memory = context.dataset_size_bytes <= context.available_memory_bytes;
            let mode = if fits_in_memory {
                SKExecutionMode::InProcessSynchronous
            } else {
                match context.access_pattern {
                    SKAccessPattern::RandomAccess => SKExecutionMode::OutOfCoreMemoryMapped,
                    SKAccessPattern::Sequential | SKAccessPattern::Iterative => {
                        SKExecutionMode::OutOfCoreStreaming
                    }
                }
            };
            let batch_size = if mode == SKExecutionMode::OutOfCoreStreaming {
                context.batch_size_hint
            } else {
                None
            };
            Ok(SKExecutionPlan {
                mode,
                parallelism,
                batch_size,
            })
        }
        SKExecutionMode::InProcessSynchronous
        | SKExecutionMode::InProcessAsynchronous
        | SKExecutionMode::OutOfCoreMemoryMapped => Ok(SKExecutionPlan {
            mode: intent,
            parallelism,
            batch_size: None,
        }),
        SKExecutionMode::OutOfCoreStreaming => {
            if context.access_pattern == SKAccessPattern::RandomAccess {
                Err(SKError::ExecutionModeIncompatible {
                    mode: "out-of-core-streaming",
                    pattern: "random-access",
                })
            } else {
                Ok(SKExecutionPlan {
                    mode: intent,
                    parallelism,
                    batch_size: context.batch_size_hint,
                })
            }
        }
    }
}
