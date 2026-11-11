//! Parallel and sequential slot execution for isolation-tree growth and row scoring.

use rayon::prelude::*;
use sciencekit_common::{SKError, SKExecutionMode};

/// Executes `number_of_trees` independent tree fits into lock-free slots.
pub(super) fn sk_execute_tree_slots<T, F>(
    number_of_trees: usize,
    sample_size: usize,
    execution_mode: SKExecutionMode,
    fit_slot: F,
) -> Result<Vec<T>, SKError>
where
    T: Send,
    F: Fn(usize) -> Result<T, SKError> + Sync + Send,
{
    if execution_mode == SKExecutionMode::OutOfCoreStreaming {
        return Err(SKError::ExecutionModeIncompatible {
            mode: "out-of-core-streaming",
            pattern: "random-access",
        });
    }
    if execution_mode == SKExecutionMode::InProcessSynchronous || number_of_trees <= 1 {
        return (0..number_of_trees).map(fit_slot).collect();
    }
    let grain = (4096usize / sample_size.max(1)).clamp(1, number_of_trees);
    (0..number_of_trees)
        .into_par_iter()
        .with_min_len(grain)
        .map(fit_slot)
        .collect()
}

/// Evaluates `row_count` independent rows in order according to `execution_mode`.
pub(super) fn sk_map_rows<T, F>(
    row_count: usize,
    tree_count: usize,
    execution_mode: SKExecutionMode,
    evaluate_row: F,
) -> Result<Vec<T>, SKError>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if execution_mode == SKExecutionMode::OutOfCoreStreaming {
        return Err(SKError::ExecutionModeIncompatible {
            mode: "out-of-core-streaming",
            pattern: "random-access",
        });
    }
    if execution_mode == SKExecutionMode::InProcessSynchronous || row_count <= 1 {
        return Ok((0..row_count).map(evaluate_row).collect());
    }
    let grain = (4096usize / tree_count.max(1)).clamp(1, row_count);
    Ok((0..row_count)
        .into_par_iter()
        .with_min_len(grain)
        .map(evaluate_row)
        .collect())
}
