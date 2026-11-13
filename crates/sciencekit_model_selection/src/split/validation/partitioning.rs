//! Row counts, split-count checks and index-partition helpers.

use sciencekit_common::{SKDataView, SKError};

use crate::SKIndexSplit;

/// The number of rows of a feature view, whichever representation it carries.
pub(crate) fn sk_number_of_rows<F>(features: &SKDataView<'_, F>) -> usize {
    match features {
        SKDataView::Dense(view) => view.nrows(),
        SKDataView::Sparse(view) => view.rows(),
        _ => 0,
    }
}

/// Rejects an operation that requires observations but received none.
pub(crate) fn sk_validate_non_empty(
    number_of_rows: usize,
    operation: &'static str,
) -> Result<(), SKError> {
    if number_of_rows == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    Ok(())
}

/// Validates an explicit split count against the row count.
pub(crate) fn sk_validate_number_of_splits(
    number_of_splits: usize,
    number_of_rows: usize,
) -> Result<(), SKError> {
    if number_of_splits < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            reason: "must be at least 2".to_owned(),
        });
    }
    if number_of_splits > number_of_rows {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            reason: format!("{number_of_splits} splits exceed the {number_of_rows} rows available"),
        });
    }
    Ok(())
}

/// Splits an ordered row sequence into consecutive test chunks.
///
/// The remainder is distributed to the first folds, so the fold sizes differ by
/// at most one and the chunks are disjoint and exhaustive.
pub(crate) fn sk_partition_into_chunks(
    order: &[usize],
    number_of_splits: usize,
) -> Vec<Vec<usize>> {
    let base = order.len() / number_of_splits;
    let remainder = order.len() % number_of_splits;
    let mut chunks = Vec::with_capacity(number_of_splits);
    let mut start = 0;
    for fold in 0..number_of_splits {
        let size = base + usize::from(fold < remainder);
        chunks.push(order[start..start + size].to_vec());
        start += size;
    }
    chunks
}

/// Builds a split whose train side is the ascending complement of `test`.
pub(crate) fn sk_split_from_test(number_of_rows: usize, test: Vec<usize>) -> SKIndexSplit {
    let mut is_test = vec![false; number_of_rows];
    for &index in &test {
        is_test[index] = true;
    }
    let train = (0..number_of_rows)
        .filter(|&index| !is_test[index])
        .collect();
    SKIndexSplit::new(train, test)
}

/// Validates that a class or group count can populate the requested splits.
pub(crate) fn sk_validate_cardinality(
    name: &'static str,
    number_of_groups: usize,
    number_of_splits: usize,
) -> Result<(), SKError> {
    if number_of_groups < number_of_splits {
        return Err(SKError::InvalidHyperparameter {
            name,
            reason: format!(
                "{number_of_groups} distinct values cannot fill {number_of_splits} splits"
            ),
        });
    }
    Ok(())
}
