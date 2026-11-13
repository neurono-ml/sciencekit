//! Dimension, finiteness (missing values allowed), execution-mode and
//! row-parallel prediction helpers for histogram boosting.

use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKExecutionMode, SKFloat};

pub(crate) use crate::gradient_boosting::validation::{
    sk_canonicalize_targets, sk_fit_dimensions, sk_predict_dimensions, sk_reject_streaming_mode,
    sk_select_canonical_class,
};

/// Reject infinities while accepting `NaN` as the documented missing marker.
pub(crate) fn sk_validate_finite_or_missing_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    let position = match features {
        SKDataView::Dense(dense) => {
            let columns = dense.ncols();
            dense
                .rows()
                .into_iter()
                .enumerate()
                .find_map(|(row, values)| {
                    values
                        .iter()
                        .position(|value| value.is_infinite())
                        .map(|column| row * columns + column)
                })
        }
        SKDataView::Sparse(sparse) => {
            let columns = sparse.cols();
            sparse
                .outer_iterator()
                .enumerate()
                .find_map(|(row, sparse_row)| {
                    sparse_row
                        .indices()
                        .iter()
                        .zip(sparse_row.data().iter())
                        .find(|(_, value)| value.is_infinite())
                        .map(|(&column, _)| row * columns + column)
                })
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
        }
    };
    position.map_or(Ok(()), |position| Err(SKError::NonFiniteValue { position }))
}

/// Evaluate `row_count` prediction rows sequentially or in parallel.
pub(crate) fn sk_map_prediction_rows<T, F>(
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
