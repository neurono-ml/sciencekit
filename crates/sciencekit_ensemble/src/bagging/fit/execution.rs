//! Lock-free slot execution and chunked row-parallel execution for bagging.

use rayon::prelude::*;
use sciencekit_common::{SKError, SKExecutionMode};

use super::super::error::SKBaggingError;

/// Execute `number_of_estimators` independent fits into lock-free slots.
pub(in crate::bagging) fn sk_execute_estimator_slots<T, E, F>(
    number_of_estimators: usize,
    sample_size: usize,
    execution_intent: SKExecutionMode,
    fit_slot: F,
) -> Result<Vec<T>, SKBaggingError<E>>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, SKBaggingError<E>> + Sync + Send,
{
    if execution_intent == SKExecutionMode::OutOfCoreStreaming {
        return Err(SKBaggingError::Central(
            SKError::ExecutionModeIncompatible {
                mode: "out-of-core-streaming",
                pattern: "random-access",
            },
        ));
    }
    if execution_intent == SKExecutionMode::InProcessSynchronous || number_of_estimators <= 1 {
        return (0..number_of_estimators).map(fit_slot).collect();
    }
    let grain = (4096usize / sample_size.max(1)).clamp(1, number_of_estimators);
    (0..number_of_estimators)
        .into_par_iter()
        .with_min_len(grain)
        .map(fit_slot)
        .collect()
}

/// Partition `row_count` prediction rows into deterministic chunks and evaluate
/// each chunk sequentially or in parallel according to `execution_intent`.
#[allow(dead_code)]
pub(in crate::bagging) fn sk_map_row_chunks<T, E, F>(
    row_count: usize,
    estimator_count: usize,
    execution_intent: SKExecutionMode,
    evaluate_chunk: F,
) -> Result<Vec<T>, SKBaggingError<E>>
where
    T: Send,
    E: Send,
    F: Fn(usize, usize) -> Result<T, SKBaggingError<E>> + Sync + Send,
{
    if execution_intent == SKExecutionMode::OutOfCoreStreaming {
        return Err(SKBaggingError::Central(
            SKError::ExecutionModeIncompatible {
                mode: "out-of-core-streaming",
                pattern: "random-access",
            },
        ));
    }
    if execution_intent == SKExecutionMode::InProcessSynchronous || row_count <= 32 {
        return evaluate_chunk(0, row_count).map(|single| vec![single]);
    }
    let chunk_rows = (4096usize / estimator_count.max(1)).clamp(32, row_count);
    let chunk_count = row_count.div_ceil(chunk_rows);
    (0..chunk_count)
        .into_par_iter()
        .map(|chunk_index| {
            let start = chunk_index * chunk_rows;
            let end = (start + chunk_rows).min(row_count);
            evaluate_chunk(start, end)
        })
        .collect()
}
