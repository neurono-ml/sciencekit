//! Target row gathering.

use ndarray::Array1;
use sciencekit_common::{SKError, SKFloat, SKTargetView};

use super::SKTargetPartition;

/// Gathers the selected targets, preserving the target representation.
pub(crate) fn sk_gather_target_rows<F: SKFloat>(
    targets: SKTargetView<'_, F>,
    row_indices: &[usize],
) -> Result<SKTargetPartition<F>, SKError> {
    match targets {
        SKTargetView::Continuous(values) => Ok(SKTargetPartition::Continuous(
            sk_gather_continuous(values.view(), row_indices)?,
        )),
        SKTargetView::Integer(values) => Ok(SKTargetPartition::Integer(sk_gather_integer(
            values.view(),
            row_indices,
        )?)),
        SKTargetView::Nominal(values) => {
            let mut gathered = Vec::with_capacity(row_indices.len());
            for &row in row_indices {
                let label = values
                    .get(row)
                    .ok_or_else(|| sk_index_out_of_range(row, values.len()))?;
                gathered.push((*label).to_owned());
            }
            Ok(SKTargetPartition::Nominal(gathered))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the targets to continuous, integer or nominal text",
        }),
    }
}

fn sk_gather_continuous<F: SKFloat>(
    values: ndarray::ArrayView1<'_, F>,
    row_indices: &[usize],
) -> Result<Array1<F>, SKError> {
    let mut gathered = Array1::zeros(row_indices.len());
    for (position, &row) in row_indices.iter().enumerate() {
        gathered[position] = *values
            .get(row)
            .ok_or_else(|| sk_index_out_of_range(row, values.len()))?;
    }
    Ok(gathered)
}

fn sk_gather_integer(
    values: ndarray::ArrayView1<'_, i64>,
    row_indices: &[usize],
) -> Result<Array1<i64>, SKError> {
    let mut gathered = Array1::zeros(row_indices.len());
    for (position, &row) in row_indices.iter().enumerate() {
        gathered[position] = *values
            .get(row)
            .ok_or_else(|| sk_index_out_of_range(row, values.len()))?;
    }
    Ok(gathered)
}

fn sk_index_out_of_range(row: usize, length: usize) -> SKError {
    SKError::InvalidHyperparameter {
        name: "row_indices",
        reason: format!("row index {row} is outside the {length} targets of the input"),
    }
}
