//! Feature/target dimension, finiteness, execution-mode and row-parallel
//! prediction helpers for gradient boosting.

use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionMode, SKFloat, SKLabelTable, SKTargetView,
    sk_canonicalize_labels,
};

/// Validate that a fit feature view is non-empty, has at least one column and
/// uses dense or CSR storage.
pub(super) fn sk_fit_dimensions<F>(
    features: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_extract_dimensions(features)?;
    if rows == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    if columns == 0 {
        return Err(SKError::shape_mismatch_2d(rows, 1, rows, 0));
    }
    Ok((rows, columns))
}

/// Validate that a prediction feature view is non-empty, matches
/// `expected_columns`, and uses dense or CSR storage.
pub(super) fn sk_predict_dimensions<F>(
    features: &SKDataView<'_, F>,
    expected_columns: usize,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = sk_extract_dimensions(features)?;
    if rows == 0 {
        return Err(SKError::EmptyInput {
            operation: "predict",
        });
    }
    if columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows,
            expected_columns,
            rows,
            columns,
        ));
    }
    Ok((rows, columns))
}

fn sk_extract_dimensions<F>(features: &SKDataView<'_, F>) -> Result<(usize, usize), SKError> {
    match features {
        SKDataView::Dense(dense) => Ok((dense.nrows(), dense.ncols())),
        SKDataView::Sparse(sparse) => {
            if !sparse.is_csr() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR with to_csr",
                });
            }
            Ok((sparse.rows(), sparse.cols()))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

/// Reject `OutOfCoreStreaming` on in-memory stage-wise gradient boosting.
pub(super) fn sk_reject_streaming_mode(execution_intent: SKExecutionMode) -> Result<(), SKError> {
    if execution_intent == SKExecutionMode::OutOfCoreStreaming {
        return Err(SKError::ExecutionModeIncompatible {
            mode: "out-of-core-streaming",
            pattern: "random-access",
        });
    }
    Ok(())
}

/// Reject any non-finite cell in a dense or CSR view at its flattened row-major
/// coordinate.
pub(super) fn sk_validate_finite_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            let columns = dense.ncols();
            let bad = dense
                .rows()
                .into_iter()
                .enumerate()
                .find_map(|(row, values)| {
                    values
                        .iter()
                        .position(|value| !value.is_finite())
                        .map(|column| row * columns + column)
                });
            bad.map_or(Ok(()), |position| Err(SKError::NonFiniteValue { position }))
        }
        SKDataView::Sparse(sparse) => {
            let columns = sparse.cols();
            let bad = sparse
                .outer_iterator()
                .enumerate()
                .find_map(|(row, sparse_row)| {
                    sparse_row
                        .indices()
                        .iter()
                        .zip(sparse_row.data().iter())
                        .find(|(_, value)| !value.is_finite())
                        .map(|(&column, _)| row * columns + column)
                });
            bad.map_or(Ok(()), |position| Err(SKError::NonFiniteValue { position }))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

/// Canonicalize nominal or integer classification targets.
pub(super) fn sk_canonicalize_targets<F: SKFloat>(
    targets: &SKTargetView<'_, F>,
) -> Result<(Vec<usize>, SKLabelTable), SKError> {
    match targets {
        SKTargetView::Nominal(labels) => Ok(sk_canonicalize_labels(labels)),
        SKTargetView::Integer(values) => {
            let owned: Vec<String> = values.iter().map(ToString::to_string).collect();
            let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
            Ok(sk_canonicalize_labels(&borrowed))
        }
        SKTargetView::Continuous(_) => Err(SKError::UnsupportedRepresentation {
            representation: "continuous",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "pass integer or nominal class labels to a classifier",
        }),
    }
}

/// Evaluate `row_count` prediction rows sequentially or in parallel according
/// to `execution_intent`.
pub(super) fn sk_map_prediction_rows<T, F>(
    row_count: usize,
    tree_count: usize,
    execution_intent: SKExecutionMode,
    evaluate_row: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if execution_intent == SKExecutionMode::InProcessSynchronous || row_count <= 1 {
        return (0..row_count).map(evaluate_row).collect();
    }
    let grain = (4096usize / tree_count.max(1)).clamp(1, row_count);
    (0..row_count)
        .into_par_iter()
        .with_min_len(grain)
        .map(evaluate_row)
        .collect()
}

/// Select the largest class probability, breaking exact ties toward the
/// smallest canonical class index.
pub(super) fn sk_select_canonical_class<F: SKFloat>(probabilities: &[F]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .skip(1)
        .fold(0usize, |best, (class_index, &probability)| {
            if probability > probabilities[best] {
                class_index
            } else {
                best
            }
        })
}
