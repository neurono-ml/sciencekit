//! Column-wise concatenation of member [`SKFeatureBlock`] outputs (spec
//! `feature-union`).

use ndarray::{ArrayView2, Axis};
use sciencekit_common::{SKError, SKFloat};
use sprs::CsMat;

use crate::SKFeatureBlock;

/// Concatenate member output blocks horizontally along the column axis.
///
/// All-dense outputs produce a dense block; if any member produces a CSR sparse
/// block, the union produces a CSR sparse block with dense members converted
/// locally and sparse members shifted by their cumulative column offset.
pub(super) fn sk_concatenate_blocks<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_widths: &[usize],
    expected_total_width: usize,
) -> Result<SKFeatureBlock<F>, SKError> {
    validate_member_outputs(outputs, expected_rows, expected_widths)?;
    let any_sparse = outputs.iter().any(SKFeatureBlock::is_sparse);
    if any_sparse {
        concatenate_csr(outputs, expected_rows, expected_total_width)
    } else {
        concatenate_dense(outputs, expected_rows, expected_total_width)
    }
}

fn concatenate_dense<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_total_width: usize,
) -> Result<SKFeatureBlock<F>, SKError> {
    let dense_views: Vec<ArrayView2<'_, F>> = outputs
        .iter()
        .map(|block| match block {
            SKFeatureBlock::Dense(matrix) => Ok(matrix.view()),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "member transformers must return a dense or CSR feature block",
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let concatenated =
        ndarray::concatenate(Axis(1), &dense_views).map_err(|_| SKError::ShapeMismatch {
            expected: vec![expected_rows, expected_total_width],
            found: outputs.iter().map(SKFeatureBlock::number_of_rows).collect(),
        })?;
    Ok(SKFeatureBlock::Dense(concatenated))
}

fn concatenate_csr<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_total_width: usize,
) -> Result<SKFeatureBlock<F>, SKError> {
    let column_offsets: Vec<usize> = outputs
        .iter()
        .scan(0_usize, |running, block| {
            let current = *running;
            *running += block.number_of_columns();
            Some(current)
        })
        .collect();

    let per_row_entries: Vec<Vec<(usize, F)>> = (0..expected_rows)
        .map(|row_index| collect_csr_row(outputs, &column_offsets, row_index))
        .collect();

    let mut row_pointers = Vec::with_capacity(expected_rows + 1);
    row_pointers.push(0_usize);
    per_row_entries.iter().fold(0_usize, |accumulated, row| {
        let next = accumulated + row.len();
        row_pointers.push(next);
        next
    });

    let total_stored = row_pointers.last().copied().unwrap_or(0);
    let mut column_indices = Vec::with_capacity(total_stored);
    let mut values = Vec::with_capacity(total_stored);
    per_row_entries.into_iter().for_each(|row| {
        row.into_iter().for_each(|(column, value)| {
            column_indices.push(column);
            values.push(value);
        });
    });

    let matrix = CsMat::new_from_unsorted(
        (expected_rows, expected_total_width),
        row_pointers,
        column_indices,
        values,
    )
    .map_err(|(_, _, _, structure_error)| {
        SKError::Conversion(format!("CSR concatenation failed: {structure_error}"))
    })?;
    Ok(SKFeatureBlock::Sparse(matrix))
}

fn collect_csr_row<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    column_offsets: &[usize],
    row_index: usize,
) -> Vec<(usize, F)> {
    outputs
        .iter()
        .zip(column_offsets.iter())
        .flat_map(|(block, &offset)| match block {
            SKFeatureBlock::Dense(matrix) => matrix
                .row(row_index)
                .iter()
                .enumerate()
                .filter_map(|(column, &value)| {
                    (value != F::zero()).then_some((offset + column, value))
                })
                .collect::<Vec<_>>(),
            SKFeatureBlock::Sparse(matrix) => matrix
                .outer_view(row_index)
                .map(|row| {
                    row.iter()
                        .map(|(column, &value)| (offset + column, value))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        })
        .collect()
}

/// Validate member representations, row counts, and column widths before
/// allocating or writing any output buffer.
pub(super) fn validate_member_outputs<F>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_widths: &[usize],
) -> Result<(), SKError> {
    outputs
        .iter()
        .zip(expected_widths.iter())
        .try_for_each(|(block, &expected_width)| {
            match block {
                SKFeatureBlock::Dense(_) => {}
                SKFeatureBlock::Sparse(matrix) if !matrix.is_csr() => {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "csc",
                        suggestion: "convert sparse member output to CSR with to_csr",
                    });
                }
                SKFeatureBlock::Sparse(_) => {}
                _ => {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "unknown",
                        suggestion: "member transformers must return a dense or CSR feature block",
                    });
                }
            }
            let found_rows = block.number_of_rows();
            let found_columns = block.number_of_columns();
            if found_rows != expected_rows || found_columns != expected_width {
                return Err(SKError::ShapeMismatch {
                    expected: vec![expected_rows, expected_width],
                    found: vec![found_rows, found_columns],
                });
            }
            Ok(())
        })
}
