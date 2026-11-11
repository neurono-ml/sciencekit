//! Column-wise concatenation of member [`SKFeatureBlock`] outputs (spec
//! `feature-union`).

use ndarray::{ArrayView2, Axis};
use sciencekit_common::{SKError, SKFloat};

use crate::SKFeatureBlock;

/// Concatenate member output blocks horizontally along the column axis.
pub(super) fn sk_concatenate_blocks<F: SKFloat>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_widths: &[usize],
    expected_total_width: usize,
) -> Result<SKFeatureBlock<F>, SKError> {
    validate_member_shapes(outputs, expected_rows, expected_widths)?;
    let dense_views: Vec<ArrayView2<'_, F>> = outputs
        .iter()
        .map(|block| match block {
            SKFeatureBlock::Dense(matrix) => Ok(matrix.view()),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "sparse_or_unknown",
                suggestion: "use a CSR-supporting concatenation path",
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

/// Validate that every member produced the expected row count and column width
/// before writing any output buffer.
pub(super) fn validate_member_shapes<F>(
    outputs: &[SKFeatureBlock<F>],
    expected_rows: usize,
    expected_widths: &[usize],
) -> Result<(), SKError> {
    outputs
        .iter()
        .zip(expected_widths.iter())
        .try_for_each(|(block, &expected_width)| {
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
