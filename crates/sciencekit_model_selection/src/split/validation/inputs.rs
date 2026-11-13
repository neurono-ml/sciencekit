//! Finiteness and class-group validation of split inputs.

use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

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
    targets: Option<SKTargetView<'_, F>>,
) -> Result<(), SKError> {
    if let Some(SKTargetView::Continuous(values)) = targets {
        for (position, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(SKError::NonFiniteValue { position });
            }
        }
    }
    Ok(())
}

/// The rows grouped by class, in first-occurrence class order.
///
/// Continuous targets are rejected with `UnsupportedRepresentation`; integer
/// and nominal targets are canonicalized to dense class indices.
pub(crate) fn sk_class_groups<F: SKFloat>(
    targets: Option<SKTargetView<'_, F>>,
) -> Result<Vec<Vec<usize>>, SKError> {
    let targets = targets.ok_or_else(|| SKError::InvalidHyperparameter {
        name: "targets",
        reason: "stratification requires targets".to_owned(),
    })?;

    let labels: Vec<usize> = match targets {
        SKTargetView::Integer(values) => {
            let collected: Vec<i64> = values.iter().copied().collect();
            sk_canonicalize_integers(&collected)
        }
        SKTargetView::Nominal(text) => sciencekit_common::sk_canonicalize_labels(text).0,
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
