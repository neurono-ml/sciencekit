//! Single-slot bootstrap fitting for [`SKBaggingClassifier`].

use ndarray::Array1;
use sciencekit_common::{SKDataView, SKFloat, SKSupervisedFit};

use crate::bootstrap::sk_out_of_bag_rows;

use super::super::classifier::SKBaggingClassifier;
use super::super::error::SKBaggingError;
use super::super::model::SKBaggingClassifierMember;
use super::super::sampling::{
    sk_bagging_feature_indices, sk_bagging_row_indices, sk_gather_dense_submatrix,
    sk_gather_sparse_submatrix,
};
use super::super::validation::sk_first_occurrence_classes;

pub(super) struct ClassifierSlot<M> {
    pub(super) member: SKBaggingClassifierMember<M>,
    pub(super) feature_subset: Vec<usize>,
    pub(super) out_of_bag_rows: Option<Vec<usize>>,
}

pub(super) fn sk_fit_classifier_slot<F, B>(
    estimator: &SKBaggingClassifier<B>,
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    two_class_pair: [usize; 2],
    (rows_count, columns_count): (usize, usize),
    estimator_index: usize,
) -> Result<ClassifierSlot<B::Model>, SKBaggingError<B::Error>>
where
    F: SKFloat,
    B: SKSupervisedFit<F> + Clone + Send + Sync,
{
    let configuration = &estimator.configuration;
    let drawn = sk_bagging_row_indices(
        rows_count,
        &configuration.maximum_samples,
        configuration.bootstrap,
        configuration.random_seed,
        estimator_index,
    );
    let feature_subset = sk_bagging_feature_indices(
        columns_count,
        &configuration.maximum_features,
        configuration.bootstrap_features,
        configuration.random_seed,
        estimator_index,
    );
    let out_of_bag_rows = configuration
        .oob_score
        .then(|| sk_out_of_bag_rows(rows_count, &drawn));
    let sampled_classes: Vec<usize> = drawn.iter().map(|&row| class_indices[row]).collect();
    let local_to_ensemble = sk_first_occurrence_classes(&sampled_classes);
    let labels = Array1::from_iter(
        sampled_classes
            .iter()
            .map(|&class_index| class_index as i64),
    );

    let member = match sk_fit_base_classifier(
        &estimator.base_estimator,
        features,
        &drawn,
        &feature_subset,
        &labels,
    ) {
        Ok(model) => SKBaggingClassifierMember::Fitted {
            model,
            local_to_ensemble,
        },
        Err(_) if local_to_ensemble.len() < 2 => {
            let probe = Array1::from_vec(vec![0_i64, 1]);
            sk_fit_base_classifier(
                &estimator.base_estimator,
                features,
                &two_class_pair,
                &feature_subset,
                &probe,
            )
            .map_err(SKBaggingError::Base)?;
            SKBaggingClassifierMember::ConstantClass {
                class_index: local_to_ensemble[0],
            }
        }
        Err(base_error) => return Err(SKBaggingError::Base(base_error)),
    };

    Ok(ClassifierSlot {
        member,
        feature_subset,
        out_of_bag_rows,
    })
}

fn sk_fit_base_classifier<F: SKFloat, B: SKSupervisedFit<F>>(
    base: &B,
    features: &SKDataView<'_, F>,
    rows: &[usize],
    subset: &[usize],
    labels: &Array1<i64>,
) -> Result<B::Model, B::Error> {
    match features {
        SKDataView::Dense(dense) => base.fit(
            sk_gather_dense_submatrix(dense, rows, subset).view(),
            labels.view(),
        ),
        SKDataView::Sparse(sparse) => base.fit(
            sk_gather_sparse_submatrix(sparse, rows, subset).view(),
            labels.view(),
        ),
        _ => unreachable!("dimensions already validated"),
    }
}
