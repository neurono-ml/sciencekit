//! Resolution of an execution plan from an intent and a context.

use crate::SKError;

use super::context::SKExecutionContext;
use super::modes::{SKAccessPattern, SKExecutionMode};
use super::plan::SKExecutionPlan;

/// Resolve an execution plan from an intent and a context.
///
/// Pure and deterministic: the same `(intent, context, grain)` always yields the
/// same plan. Automatic intent never produces an incompatibility error; an
/// explicit mode incompatible with the declared access pattern does.
///
/// The resolved `parallelism` derives from the size of the work unit — the whole
/// dataset for in-memory modes, one batch for streaming, an arbitrary shard for
/// memory-mapped access — capped by the documented per-kernel `grain`
/// (minimum elements per thread) and by the core count. No machine is read here:
/// the caller injects the context.
pub fn sk_resolve_execution_plan(
    intent: SKExecutionMode,
    context: &SKExecutionContext,
) -> Result<SKExecutionPlan, SKError> {
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
                parallelism: parallelism_for(mode, context),
                batch_size,
            })
        }
        SKExecutionMode::InProcessSynchronous
        | SKExecutionMode::InProcessAsynchronous
        | SKExecutionMode::OutOfCoreMemoryMapped => Ok(SKExecutionPlan {
            mode: intent,
            parallelism: parallelism_for(intent, context),
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
                    parallelism: parallelism_for(intent, context),
                    batch_size: context.batch_size_hint,
                })
            }
        }
    }
}

/// The parallelism for a resolved mode: `min(cores, ceil(unit / grain))`, where
/// the unit is the whole dataset (in-memory), one batch (streaming) or an
/// arbitrary shard (memory-mapped → always `cores`). Sub-grain or unknown units
/// and single-core machines resolve to exactly one.
fn parallelism_for(
    mode: SKExecutionMode,
    context: &SKExecutionContext,
) -> usize {
    if context.cpu_cores <= 1 {
        return 1;
    }
    let unit_elements = match mode {
        SKExecutionMode::OutOfCoreStreaming => context.batch_size_hint.map(|batch| batch as u64),
        SKExecutionMode::OutOfCoreMemoryMapped => Some(u64::MAX),
        _ => context.dataset_elements,
    };
    match unit_elements {
        Some(unit) if unit > 0 => {
            let threads = unit.div_ceil(context.grain.max(1) as u64);
            threads.clamp(1, context.cpu_cores as u64) as usize
        }
        _ => 1,
    }
}