//! Decision values on [`SKLinearSVCModel`]
//! (spec `linear-support-vector-classifier`).

use ndarray::{Array1, Array2, ArrayView1, Axis};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsVecView;

use super::model::{SKLinearSVCHead, SKLinearSVCModel};
use super::validation::sk_validate_linear_svc_query_features;

fn sk_dense_head_decision<F: SKFloat>(head: &SKLinearSVCHead<F>, row: ArrayView1<'_, F>) -> F {
    row.dot(head.coefficients()) + head.intercept()
}

fn sk_sparse_head_decision<F: SKFloat>(head: &SKLinearSVCHead<F>, row: CsVecView<'_, F>) -> F {
    row.indices()
        .iter()
        .zip(row.data().iter())
        .fold(head.intercept(), |accumulator, (&column, &value)| {
            accumulator + head.coefficients()[column] * value
        })
}

/// One decision value per class for every query row, shape `(rows, classes)`.
///
/// A binary model exposes `[-d, d]` where `d` is the single head's margin, so
/// the larger canonical class is favored by a positive value. A multiclass
/// model exposes the one-vs-rest head of each class in canonical order.
pub(crate) fn sk_linear_svc_decision_matrix<F: SKFloat>(
    model: &SKLinearSVCModel<F>,
    features: SKDataView<'_, F>,
    operation: &'static str,
) -> Result<Array2<F>, SKError> {
    let (rows, _) =
        sk_validate_linear_svc_query_features(features, model.number_of_features, operation)?;
    let number_of_classes = model.number_of_classes();
    let single_head = model.heads.len() == 1;
    let mut decisions = Array2::zeros((rows, number_of_classes));

    match features {
        SKDataView::Dense(dense) => {
            decisions
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(row_index, mut out_row)| {
                    if single_head {
                        let margin = sk_dense_head_decision(&model.heads[0], dense.row(row_index));
                        out_row[0] = -margin;
                        out_row[1] = margin;
                    } else {
                        out_row
                            .iter_mut()
                            .zip(model.heads.iter())
                            .for_each(|(slot, head)| {
                                *slot = sk_dense_head_decision(head, dense.row(row_index));
                            });
                    }
                });
        }
        SKDataView::Sparse(sparse) => {
            decisions
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(row_index, mut out_row)| {
                    let stored = sparse
                        .outer_view(row_index)
                        .expect("row index bounded by the validated shape");
                    if single_head {
                        let margin = sk_sparse_head_decision(&model.heads[0], stored);
                        out_row[0] = -margin;
                        out_row[1] = margin;
                    } else {
                        out_row
                            .iter_mut()
                            .zip(model.heads.iter())
                            .for_each(|(slot, head)| {
                                *slot = sk_sparse_head_decision(head, stored);
                            });
                    }
                });
        }
        // `SKDataView` is `#[non_exhaustive]` in `sciencekit_common`, so the
        // wildcard is required across the crate boundary even though the
        // currently known variants are only `Dense` and `Sparse`.
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a dense matrix or a CSR matrix",
            });
        }
    }
    Ok(decisions)
}

/// Canonical winner index: the first (smallest) index holding the maximum.
pub(crate) fn sk_linear_svc_argmax<F: SKFloat>(row: ArrayView1<'_, F>) -> usize {
    row.iter().enumerate().skip(1).fold(
        0usize,
        |winner, (index, &value)| {
            if value > row[winner] { index } else { winner }
        },
    )
}

/// Predict labels (canonical class indices) from a decision matrix.
pub(crate) fn sk_linear_svc_labels_from_decisions<F: SKFloat>(
    decisions: &Array2<F>,
) -> Array1<i64> {
    Array1::from_iter(
        decisions
            .axis_iter(Axis(0))
            .map(|row| sk_linear_svc_argmax(row) as i64),
    )
}
