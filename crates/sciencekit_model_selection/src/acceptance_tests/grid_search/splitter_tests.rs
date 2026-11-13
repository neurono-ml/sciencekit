//! Every splitter works through the grid search, including grouped data.

use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::super::fixtures::{contiguous_groups, cyclic_labels, dense_design};
use crate::grid_search::SKGridSearchCVClassifier;
use crate::grid_search::mock_estimators::SKMockThresholdClassifier;
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;

fn candidates() -> Vec<SKMockThresholdClassifier> {
    vec![
        SKMockThresholdClassifier::new(2.0),
        SKMockThresholdClassifier::new(6.0),
    ]
}

#[test]
fn grid_acceptance_plain_kfold_produces_one_score_per_fold() {
    let features = dense_design(30, 2);
    let labels = cyclic_labels(30, 2);
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(candidates())
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();
    assert_eq!(model.results().number_of_folds(), 3);
}

#[test]
fn grid_acceptance_stratified_kfold_produces_one_score_per_fold() {
    let features = dense_design(30, 2);
    let labels = cyclic_labels(30, 2);
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(candidates())
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();
    assert_eq!(model.results().number_of_folds(), 3);
}

#[test]
fn grid_acceptance_group_kfold_reads_the_configured_groups() {
    let features = dense_design(30, 2);
    let labels = cyclic_labels(30, 2);
    let groups = contiguous_groups(30, 3);
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(candidates())
        .splitter(SKGroupKFold::build().number_of_splits(3).build().unwrap())
        .groups(groups)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();
    assert_eq!(model.results().number_of_folds(), 3);
}
