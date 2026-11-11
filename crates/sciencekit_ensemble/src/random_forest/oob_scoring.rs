//! Out-of-bag accuracy and R² estimation over rows absent from each tree's
//! bootstrap sample.

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;
use sciencekit_tree::SKDecisionTreeRegressorModel;

use super::classifier_model::{SKClassifierTreeMember, SKForestOobState};
use super::classifier_predict::{sk_accumulate_member_probabilities, sk_select_canonical_class};
use super::input_validation::sk_sparse_feature_value;
use super::regressor_predict::sk_regressor_tree_value;

/// Compute the out-of-bag classification accuracy from each tree's OOB rows.
pub(super) fn sk_classifier_oob_state<F: SKFloat>(
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    number_of_classes: usize,
    members: &[SKClassifierTreeMember<F>],
    oob_per_tree: Option<&[Vec<usize>]>,
) -> Result<SKForestOobState<F>, SKError> {
    let Some(oob_per_tree) = oob_per_tree else {
        return Ok(SKForestOobState::Disabled);
    };
    let rows_count = class_indices.len();
    let mut probability_sums = vec![F::zero(); rows_count * number_of_classes];
    let mut oob_counts = vec![0usize; rows_count];

    match features {
        SKDataView::Dense(dense) => {
            members
                .iter()
                .zip(oob_per_tree.iter())
                .for_each(|(member, oob_rows)| {
                    oob_rows.iter().for_each(|&row| {
                        let slice = &mut probability_sums
                            [row * number_of_classes..(row + 1) * number_of_classes];
                        sk_accumulate_member_probabilities(member, slice, |feature| {
                            dense[[row, feature]]
                        });
                        oob_counts[row] += 1;
                    });
                });
        }
        SKDataView::Sparse(sparse) => {
            members
                .iter()
                .zip(oob_per_tree.iter())
                .for_each(|(member, oob_rows)| {
                    oob_rows.iter().for_each(|&row| {
                        let sparse_row = sparse.outer_view(row).expect("row within bounds");
                        let slice = &mut probability_sums
                            [row * number_of_classes..(row + 1) * number_of_classes];
                        sk_accumulate_member_probabilities(member, slice, |feature| {
                            sk_sparse_feature_value(&sparse_row, feature)
                        });
                        oob_counts[row] += 1;
                    });
                });
        }
        _ => unreachable!("dimensions already validated"),
    }

    let mut true_labels = Vec::new();
    let mut predicted_labels = Vec::new();
    oob_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .for_each(|(row, &count)| {
            let divisor = F::from(count).expect("OOB count fits in scalar");
            let slice =
                &mut probability_sums[row * number_of_classes..(row + 1) * number_of_classes];
            slice.iter_mut().for_each(|value| *value = *value / divisor);
            true_labels.push(class_indices[row] as i64);
            predicted_labels.push(sk_select_canonical_class(slice) as i64);
        });

    if true_labels.is_empty() {
        return Ok(SKForestOobState::NoScoredRows);
    }
    let score = sk_accuracy_score::<F>(&true_labels, &predicted_labels)?;
    Ok(SKForestOobState::Scored {
        score,
        scored_rows: true_labels.len(),
    })
}

/// Compute the out-of-bag regression R² from each tree's OOB rows.
pub(super) fn sk_regressor_oob_state<F: SKFloat>(
    features: &SKDataView<'_, F>,
    responses: &ArrayView1<'_, F>,
    trees: &[SKDecisionTreeRegressorModel<F>],
    oob_per_tree: Option<&[Vec<usize>]>,
) -> Result<SKForestOobState<F>, SKError> {
    let Some(oob_per_tree) = oob_per_tree else {
        return Ok(SKForestOobState::Disabled);
    };
    let rows_count = responses.len();
    let mut response_sums = vec![F::zero(); rows_count];
    let mut oob_counts = vec![0usize; rows_count];

    match features {
        SKDataView::Dense(dense) => {
            trees
                .iter()
                .zip(oob_per_tree.iter())
                .for_each(|(tree, oob_rows)| {
                    oob_rows.iter().for_each(|&row| {
                        response_sums[row] = response_sums[row]
                            + sk_regressor_tree_value(tree, |feature| dense[[row, feature]]);
                        oob_counts[row] += 1;
                    });
                });
        }
        SKDataView::Sparse(sparse) => {
            trees
                .iter()
                .zip(oob_per_tree.iter())
                .for_each(|(tree, oob_rows)| {
                    oob_rows.iter().for_each(|&row| {
                        let sparse_row = sparse.outer_view(row).expect("row within bounds");
                        response_sums[row] = response_sums[row]
                            + sk_regressor_tree_value(tree, |feature| {
                                sk_sparse_feature_value(&sparse_row, feature)
                            });
                        oob_counts[row] += 1;
                    });
                });
        }
        _ => unreachable!("dimensions already validated"),
    }

    let mut true_targets = Vec::new();
    let mut predicted_targets = Vec::new();
    oob_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .for_each(|(row, &count)| {
            let divisor = F::from(count).expect("OOB count fits in scalar");
            true_targets.push(responses[row]);
            predicted_targets.push(response_sums[row] / divisor);
        });

    if true_targets.is_empty() {
        return Ok(SKForestOobState::NoScoredRows);
    }
    let scored_rows = true_targets.len();
    let true_array = Array1::from_vec(true_targets);
    let predicted_array = Array1::from_vec(predicted_targets);
    let score = sk_r2_score(true_array.view(), predicted_array.view())?;
    Ok(SKForestOobState::Scored { score, scored_rows })
}
