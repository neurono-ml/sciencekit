//! Column-wise concatenation of member [`SKFeatureBlock`] outputs (spec
//! `feature-union`).

use ndarray::{ArrayView2, Axis};
use sciencekit_common::{SKError, SKFloat};
use sprs::CsMat;

use crate::SKFeatureBlock;

use super::errors::{SKFeatureUnionMemberError, sk_validate_probe_output};

/// Concatenate member output blocks horizontally along the column axis.
///
/// All-dense outputs produce a dense block; if any member produces a CSR sparse
/// block, the union produces a CSR sparse block with dense members converted
/// locally and sparse members shifted by their cumulative column offset.
pub(crate) fn sk_concatenate_blocks<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_widths: &[usize],
    expected_total_width: usize,
) -> Result<SKFeatureBlock<F>, SKFeatureUnionMemberError> {
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
) -> Result<SKFeatureBlock<F>, SKFeatureUnionMemberError> {
    let dense_views: Vec<ArrayView2<'_, F>> = outputs
        .iter()
        .enumerate()
        .map(|(member_index, block)| match block {
            SKFeatureBlock::Dense(matrix) => Ok(matrix.view()),
            _ => Err(SKFeatureUnionMemberError::member(
                member_index,
                SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "member transformers must return a dense or CSR feature block",
                },
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let concatenated = ndarray::concatenate(Axis(1), &dense_views).map_err(|_| {
        SKFeatureUnionMemberError::union_level(SKError::ShapeMismatch {
            expected: vec![expected_rows, expected_total_width],
            found: outputs.iter().map(SKFeatureBlock::number_of_rows).collect(),
        })
    })?;
    Ok(SKFeatureBlock::Dense(concatenated))
}

fn concatenate_csr<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_total_width: usize,
) -> Result<SKFeatureBlock<F>, SKFeatureUnionMemberError> {
    let column_offsets: Vec<usize> = outputs
        .iter()
        .scan(0_usize, |running, block| {
            let current = *running;
            *running += block.number_of_columns();
            Some(current)
        })
        .collect();

    let total_stored: usize = outputs
        .iter()
        .map(|block| match block {
            SKFeatureBlock::Dense(matrix) => {
                matrix.iter().filter(|&&value| value != F::zero()).count()
            }
            SKFeatureBlock::Sparse(matrix) => matrix.nnz(),
            _ => 0,
        })
        .sum();

    let mut row_pointers = Vec::with_capacity(expected_rows + 1);
    let mut column_indices = Vec::with_capacity(total_stored);
    let mut values = Vec::with_capacity(total_stored);
    row_pointers.push(0_usize);

    (0..expected_rows).for_each(|row_index| {
        append_csr_row(
            outputs,
            &column_offsets,
            row_index,
            &mut column_indices,
            &mut values,
        );
        row_pointers.push(values.len());
    });

    let matrix = CsMat::new_from_unsorted(
        (expected_rows, expected_total_width),
        row_pointers,
        column_indices,
        values,
    )
    .map_err(|(_, _, _, structure_error)| {
        SKFeatureUnionMemberError::union_level(SKError::Conversion(format!(
            "CSR concatenation failed: {structure_error}"
        )))
    })?;
    Ok(SKFeatureBlock::Sparse(matrix))
}

fn append_csr_row<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    column_offsets: &[usize],
    row_index: usize,
    column_indices: &mut Vec<usize>,
    values: &mut Vec<F>,
) {
    outputs
        .iter()
        .zip(column_offsets.iter())
        .for_each(|(block, &offset)| match block {
            SKFeatureBlock::Dense(matrix) => {
                matrix
                    .row(row_index)
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value != F::zero())
                    .for_each(|(column, &value)| {
                        column_indices.push(offset + column);
                        values.push(value);
                    });
            }
            SKFeatureBlock::Sparse(matrix) => {
                if let Some(row) = matrix.outer_view(row_index) {
                    row.iter().for_each(|(column, &value)| {
                        column_indices.push(offset + column);
                        values.push(value);
                    });
                }
            }
            _ => {}
        });
}

/// Validate member count, representations, row counts, and column widths before
/// allocating or writing any output buffer.
fn validate_member_outputs<F>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_widths: &[usize],
) -> Result<(), SKFeatureUnionMemberError> {
    if outputs.len() != expected_widths.len() {
        return Err(SKFeatureUnionMemberError::union_level(
            SKError::ShapeMismatch {
                expected: vec![expected_widths.len()],
                found: vec![outputs.len()],
            },
        ));
    }
    outputs
        .iter()
        .zip(expected_widths.iter())
        .enumerate()
        .try_for_each(|(member_index, (block, &expected_width))| {
            let found_columns = sk_validate_probe_output(member_index, block, expected_rows)?;
            if found_columns != expected_width {
                return Err(SKFeatureUnionMemberError::member(
                    member_index,
                    SKError::ShapeMismatch {
                        expected: vec![expected_rows, expected_width],
                        found: vec![expected_rows, found_columns],
                    },
                ));
            }
            Ok(())
        })
}
