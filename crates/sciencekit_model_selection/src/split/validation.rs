//! Shared split validation and fold-partition helpers.

use sciencekit_common::{SKDataView, SKError, SKFloat};

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

/// Validates that every stored feature value is finite.
///
/// Dense values are checked in logical row-major order; sparse values are
/// checked over the stored entries only, never densifying the matrix. The
/// reported position is the flattened `row * columns + column` index.
pub(crate) fn sk_validate_finite_features<F: SKFloat>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(view) => {
            for (position, value) in view.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SKError::NonFiniteValue { position });
                }
            }
        }
        SKDataView::Sparse(view) => {
            let columns = view.cols();
            for row in 0..view.rows() {
                let outer = view
                    .outer_view(row)
                    .expect("a row below the row count has an outer view");
                for (&column, value) in outer.indices().iter().zip(outer.data()) {
                    if !value.is_finite() {
                        return Err(SKError::NonFiniteValue {
                            position: row * columns + column,
                        });
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Validates that continuous targets are finite; other representations carry no
/// non-finite values.
pub(crate) fn sk_validate_finite_targets<F: SKFloat>(
    targets: Option<sciencekit_common::SKTargetView<'_, F>>,
) -> Result<(), SKError> {
    use sciencekit_common::SKTargetView;

    if let Some(SKTargetView::Continuous(values)) = targets {
        for (position, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(SKError::NonFiniteValue { position });
            }
        }
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
            reason: format!(
                "{number_of_splits} splits exceed the {number_of_rows} rows available"
            ),
        });
    }
    Ok(())
}

/// Splits an ordered row sequence into consecutive test chunks.
///
/// The remainder is distributed to the first folds, so the fold sizes differ by
/// at most one and the chunks are disjoint and exhaustive.
pub(crate) fn sk_partition_into_chunks(order: &[usize], number_of_splits: usize) -> Vec<Vec<usize>> {
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
    let train = (0..number_of_rows).filter(|&index| !is_test[index]).collect();
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

/// The rows grouped by class, in first-occurrence class order.
///
/// Continuous targets are rejected with `UnsupportedRepresentation`; integer
/// and nominal targets are canonicalized to dense class indices.
pub(crate) fn sk_class_groups<F: SKFloat>(
    targets: Option<sciencekit_common::SKTargetView<'_, F>>,
) -> Result<Vec<Vec<usize>>, SKError> {
    use sciencekit_common::{SKTargetView, sk_canonicalize_labels};

    let targets = targets.ok_or_else(|| SKError::InvalidHyperparameter {
        name: "targets",
        reason: "stratification requires targets".to_owned(),
    })?;

    let labels: Vec<usize> = match targets {
        SKTargetView::Integer(values) => {
            let collected: Vec<i64> = values.iter().copied().collect();
            sk_canonicalize_integers(&collected)
        }
        SKTargetView::Nominal(text) => sk_canonicalize_labels(text).0,
        SKTargetView::Continuous(_) => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "continuous",
                suggestion: "stratification requires categorical targets; encode them as integers or labels",
            });
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the targets to integer or nominal labels",
            });
        }
    };

    sk_groups_from_codes(&labels)
}

fn sk_canonicalize_integers(values: &[i64]) -> Vec<usize> {
    let mut mapping = std::collections::HashMap::new();
    values
        .iter()
        .map(|value| {
            let next = mapping.len();
            *mapping.entry(*value).or_insert(next)
        })
        .collect()
}

fn sk_groups_from_codes(codes: &[usize]) -> Result<Vec<Vec<usize>>, SKError> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (row, &code) in codes.iter().enumerate() {
        if code >= groups.len() {
            groups.resize_with(code + 1, Vec::new);
        }
        groups[code].push(row);
    }
    Ok(groups)
}
