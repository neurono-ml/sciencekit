//! Input validation, the empty-column policy and the execution-plan
//! resolution shared by the batch fit (spec `simple-imputer`).

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKExecutionPlan, SKFloat};
use sprs::CsMatView;

use super::SKSimpleImputer;

/// The indices of the columns that produced no statistic, kept only when the
/// estimator retains empty features.
pub(in crate::simple_imputer) fn empty_columns<F: SKFloat>(
    statistics: &ndarray::Array1<F>,
    keep_empty_features: bool,
) -> Vec<usize> {
    if keep_empty_features {
        return Vec::new();
    }
    statistics
        .iter()
        .enumerate()
        .filter(|(_, statistic)| statistic.is_nan())
        .map(|(index, _)| index)
        .collect()
}

/// Reject empty dense inputs before any element is processed.
pub(super) fn validated_shape<F>(features: &ArrayView2<F>) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.dim();
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    Ok((rows_count, columns_count))
}

/// Reject empty sparse inputs before any element is processed.
pub(super) fn validated_sparse_shape<F>(
    features: &CsMatView<'_, F>,
) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.shape();
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    Ok((rows_count, columns_count))
}

/// Build a fresh execution context per operation, declare the reduction grain
/// and the `Sequential` access pattern, and resolve the plan.
pub(super) fn resolve_fit_plan<F: SKFloat>(
    estimator: &SKSimpleImputer<F>,
    stored_elements: u64,
) -> Result<SKExecutionPlan, SKError> {
    let mut context = sciencekit_common::SKExecutionContext::real();
    context.dataset_size_bytes = stored_elements * size_of::<F>() as u64;
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(stored_elements);
    context.access_pattern = sciencekit_common::execution::SKAccessPattern::Sequential;
    context.grain = sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
    estimator.resolve_plan(&context)
}
