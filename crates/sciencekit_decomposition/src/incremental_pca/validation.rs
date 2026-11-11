//! Input, batch-size and row-chunking validation for [`super::SKIncrementalPCA`]
//! (spec `incremental-pca`, decision 6).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::pca::{sk_validate_finiteness, sk_view_dimensions};

/// Validates full-design `fit` / `fit_transform` shape, component count,
/// `batch_size >= resolved_components`, and element finiteness.
pub(super) fn sk_validate_fit_chunking<F: SKFloat>(
    features: SKDataView<'_, F>,
    requested_components: Option<usize>,
    batch_size: usize,
    operation: &'static str,
) -> Result<(usize, usize, usize), SKError> {
    let (rows_count, columns_count) = sk_view_dimensions(&features)?;
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    let maximum_components = rows_count.min(columns_count);
    let resolved_components = requested_components.unwrap_or(maximum_components);
    if resolved_components == 0 || resolved_components > maximum_components {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_components",
            reason: format!(
                "requested {resolved_components} components, which must lie in 1..={maximum_components} for shape ({rows_count}, {columns_count})"
            ),
        });
    }
    if batch_size < resolved_components {
        return Err(SKError::InvalidHyperparameter {
            name: "batch_size",
            reason: format!(
                "batch_size ({batch_size}) must be at least number_of_components ({resolved_components})"
            ),
        });
    }
    sk_validate_finiteness(features)?;
    Ok((rows_count, columns_count, resolved_components))
}

/// Validates a single incremental `partial_fit` batch against the established
/// feature count and component count.
pub(super) fn sk_validate_partial_batch<F: SKFloat>(
    features: SKDataView<'_, F>,
    expected_columns: usize,
    required_components: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows_count, found_columns) = sk_view_dimensions(&features)?;
    if rows_count == 0 || found_columns == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if found_columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows_count,
            expected_columns,
            rows_count,
            found_columns,
        ));
    }
    if rows_count < required_components {
        return Err(SKError::InvalidHyperparameter {
            name: "batch_size",
            reason: format!(
                "batch has {rows_count} rows, which is fewer than number_of_components ({required_components})"
            ),
        });
    }
    sk_validate_finiteness(features)?;
    Ok((rows_count, found_columns))
}

/// Partitions `0..rows_count` into blocks of `batch_size`, coalescing a short
/// trailing remainder (`< minimum_rows`) into the preceding block so every
/// block satisfies the component-count minimum.
pub(super) fn sk_chunk_ranges(
    rows_count: usize,
    batch_size: usize,
    minimum_rows: usize,
) -> Vec<(usize, usize)> {
    let chunk_count = rows_count.div_ceil(batch_size);
    let mut ranges: Vec<(usize, usize)> = (0..chunk_count)
        .map(|chunk_index| {
            let start = chunk_index * batch_size;
            let end = (start + batch_size).min(rows_count);
            (start, end)
        })
        .collect();
    if ranges.len() >= 2 {
        let last_index = ranges.len() - 1;
        let (tail_start, tail_end) = ranges[last_index];
        if tail_end - tail_start < minimum_rows {
            ranges.pop();
            if let Some(previous) = ranges.last_mut() {
                previous.1 = tail_end;
            }
        }
    }
    ranges
}
