//! Out-of-bag accuracy and R² estimation for generic bagging models, routing
//! only excluded bootstrap rows through each estimator's feature subset.

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKClassifierPredictor, SKDataView, SKFloat, SKRegressorPredictor};
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::sk_r2_score;

use super::error::SKBaggingError;
use super::model::{SKBaggingClassifierMember, SKBaggingOobState};
use super::predict::{
    sk_predict_classifier_member_rows, sk_predict_regressor_rows, sk_select_canonical_bagging_class,
};

/// Compute out-of-bag classification accuracy over rows absent from each
/// estimator's bootstrap sample.
pub(super) fn sk_compute_classifier_oob_state<F, M, E>(
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    number_of_classes: usize,
    members: &[SKBaggingClassifierMember<M>],
    feature_subsets: &[Vec<usize>],
    oob_per_estimator: Option<&[Vec<usize>]>,
) -> Result<SKBaggingOobState<F>, SKBaggingError<E>>
where
    F: SKFloat,
    M: SKClassifierPredictor<F>,
    E: From<M::Error>,
{
    let Some(oob_per_estimator) = oob_per_estimator else {
        return Ok(SKBaggingOobState::Disabled);
    };
    let rows_count = class_indices.len();
    let mut probability_sums = vec![F::zero(); rows_count * number_of_classes];
    let mut out_of_bag_counts = vec![0usize; rows_count];

    members
        .iter()
        .zip(feature_subsets.iter())
        .zip(oob_per_estimator.iter())
        .try_for_each(|((member, feature_subset), oob_rows)| {
            if oob_rows.is_empty() {
                return Ok::<(), SKBaggingError<E>>(());
            }
            let predicted = sk_predict_classifier_member_rows(
                member,
                features,
                oob_rows,
                feature_subset,
                number_of_classes,
            )
            .map_err(sk_convert_bagging_error)?;
            oob_rows
                .iter()
                .enumerate()
                .for_each(|(local_row, &training_row)| {
                    let target_slice = &mut probability_sums
                        [training_row * number_of_classes..(training_row + 1) * number_of_classes];
                    target_slice
                        .iter_mut()
                        .zip(predicted.row(local_row).iter())
                        .for_each(|(accumulator, &value)| *accumulator = *accumulator + value);
                    out_of_bag_counts[training_row] += 1;
                });
            Ok(())
        })?;

    let mut true_labels = Vec::new();
    let mut predicted_labels = Vec::new();
    out_of_bag_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .for_each(|(row, &count)| {
            let divisor = F::from(count).expect("OOB count fits in scalar");
            let slice =
                &mut probability_sums[row * number_of_classes..(row + 1) * number_of_classes];
            slice.iter_mut().for_each(|value| *value = *value / divisor);
            true_labels.push(class_indices[row] as i64);
            predicted_labels.push(sk_select_canonical_bagging_class(slice) as i64);
        });

    if true_labels.is_empty() {
        return Ok(SKBaggingOobState::NoScoredRows);
    }
    let score = sk_accuracy_score::<F>(&true_labels, &predicted_labels)?;
    Ok(SKBaggingOobState::Scored {
        score,
        scored_rows: true_labels.len(),
    })
}

/// Compute out-of-bag regression R² over rows absent from each estimator's
/// bootstrap sample.
pub(super) fn sk_compute_regressor_oob_state<F, M, E>(
    features: &SKDataView<'_, F>,
    responses: &ArrayView1<'_, F>,
    estimators: &[M],
    feature_subsets: &[Vec<usize>],
    oob_per_estimator: Option<&[Vec<usize>]>,
) -> Result<SKBaggingOobState<F>, SKBaggingError<E>>
where
    F: SKFloat,
    M: SKRegressorPredictor<F>,
    E: From<M::Error>,
{
    let Some(oob_per_estimator) = oob_per_estimator else {
        return Ok(SKBaggingOobState::Disabled);
    };
    let rows_count = responses.len();
    let mut response_sums = vec![F::zero(); rows_count];
    let mut out_of_bag_counts = vec![0usize; rows_count];

    estimators
        .iter()
        .zip(feature_subsets.iter())
        .zip(oob_per_estimator.iter())
        .try_for_each(|((estimator, feature_subset), oob_rows)| {
            if oob_rows.is_empty() {
                return Ok::<(), SKBaggingError<E>>(());
            }
            let predicted =
                sk_predict_regressor_rows(estimator, features, oob_rows, feature_subset)
                    .map_err(sk_convert_bagging_error)?;
            oob_rows
                .iter()
                .zip(predicted.iter())
                .for_each(|(&training_row, &value)| {
                    response_sums[training_row] = response_sums[training_row] + value;
                    out_of_bag_counts[training_row] += 1;
                });
            Ok(())
        })?;

    let mut true_targets = Vec::new();
    let mut predicted_targets = Vec::new();
    out_of_bag_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .for_each(|(row, &count)| {
            let divisor = F::from(count).expect("OOB count fits in scalar");
            true_targets.push(responses[row]);
            predicted_targets.push(response_sums[row] / divisor);
        });

    if true_targets.is_empty() {
        return Ok(SKBaggingOobState::NoScoredRows);
    }
    let scored_rows = true_targets.len();
    let true_array = Array1::from_vec(true_targets);
    let predicted_array = Array1::from_vec(predicted_targets);
    let score = sk_r2_score(true_array.view(), predicted_array.view())?;
    Ok(SKBaggingOobState::Scored { score, scored_rows })
}

fn sk_convert_bagging_error<Source, Target>(error: SKBaggingError<Source>) -> SKBaggingError<Target>
where
    Target: From<Source>,
{
    match error {
        SKBaggingError::Central(central) => SKBaggingError::Central(central),
        SKBaggingError::Base(base) => SKBaggingError::Base(Target::from(base)),
    }
}
