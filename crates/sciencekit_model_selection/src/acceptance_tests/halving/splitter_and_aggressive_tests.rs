//! Stratified, grouped and plain K-fold splitters plus aggressive early
//! elimination acceptance scenarios on public halving estimators.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKError, SKSupervisedFit};

use super::super::fixtures::{contiguous_groups, cyclic_labels, dense_design};
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::halving::{SKHalvingGridSearchClassifier, SKHalvingGridSearchRegressor};
use crate::split::group::SKGroupKFold;
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;

#[test]
fn halving_acceptance_kfold_and_imbalanced_stratified_kfold_preserve_fold_contracts() {
    let features = Array2::from_shape_fn((90, 2), |(row, col)| (row + col) as f64);
    // 2:1 imbalanced binary classes (60 rows of class 0, 30 rows of class 1); threshold 60.0 is exact.
    let imbalanced_labels = Array1::from_iter((0..90).map(|row| i64::from(row >= 60)));
    let candidates = vec![
        SKMockThresholdClassifier::new(20.0),
        SKMockThresholdClassifier::new(60.0),
        SKMockThresholdClassifier::new(80.0),
    ];

    let kfold_model = SKHalvingGridSearchClassifier::build()
        .candidates(candidates.clone())
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .factor(3)
        .minimum_resource(18)
        .build()
        .unwrap()
        .fit(features.view(), imbalanced_labels.view())
        .unwrap();
    assert_eq!(
        kfold_model.elimination_trace()[0]
            .results()
            .number_of_folds(),
        3
    );

    let stratified_model = SKHalvingGridSearchClassifier::build()
        .candidates(candidates.clone())
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .shuffle(true)
                .random_seed(7)
                .build()
                .unwrap(),
        )
        .factor(3)
        .minimum_resource(18)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(features.view(), imbalanced_labels.view())
        .unwrap();
    assert_eq!(stratified_model.best_candidate_index(), 1);
    assert_eq!(stratified_model.best_score(), 1.0);

    // Nominal targets propagate the scorer's structured UnsupportedRepresentation error cleanly.
    let nominal_labels: Vec<&str> = (0..90)
        .map(|row| if row < 60 { "negative" } else { "positive" })
        .collect();
    let nominal_error = SKHalvingGridSearchClassifier::build()
        .candidates(candidates)
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .build()
                .unwrap(),
        )
        .minimum_resource(18)
        .build()
        .unwrap()
        .fit(features.view(), nominal_labels.as_slice());
    assert!(matches!(
        nominal_error,
        Err(SKError::UnsupportedRepresentation {
            representation: "nominal",
            ..
        })
    ));
}

#[test]
fn halving_acceptance_group_kfold_respects_groups_and_rejects_invalid_group_configurations() {
    let features = dense_design(60, 2);
    let labels = cyclic_labels(60, 2);
    let groups = contiguous_groups(60, 6);
    let candidates = vec![
        SKMockThresholdClassifier::new(15.0),
        SKMockThresholdClassifier::new(30.0),
        SKMockThresholdClassifier::new(45.0),
    ];

    let grouped_model = SKHalvingGridSearchClassifier::build()
        .candidates(candidates.clone())
        .splitter(SKGroupKFold::build().number_of_splits(3).build().unwrap())
        .groups(groups)
        .factor(3)
        .minimum_resource(18)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(
        grouped_model.elimination_trace()[0]
            .results()
            .number_of_folds(),
        3
    );

    let missing_groups = SKHalvingGridSearchClassifier::build()
        .candidates(candidates.clone())
        .splitter(SKGroupKFold::build().number_of_splits(3).build().unwrap())
        .minimum_resource(18)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());
    assert!(matches!(
        missing_groups,
        Err(SKError::InvalidHyperparameter { name: "groups", .. })
    ));

    let wrong_length = SKHalvingGridSearchClassifier::build()
        .candidates(candidates.clone())
        .splitter(SKGroupKFold::build().number_of_splits(3).build().unwrap())
        .groups(Array1::zeros(10))
        .minimum_resource(18)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());
    assert!(matches!(wrong_length, Err(SKError::ShapeMismatch { .. })));

    let single_group = SKHalvingGridSearchClassifier::build()
        .candidates(candidates)
        .splitter(SKGroupKFold::build().number_of_splits(3).build().unwrap())
        .groups(Array1::zeros(60))
        .minimum_resource(18)
        .build()
        .unwrap()
        .fit(features.view(), labels.view());
    assert!(matches!(
        single_group,
        Err(SKError::InvalidHyperparameter {
            name: "number_of_splits",
            ..
        })
    ));
}

#[test]
fn halving_acceptance_aggressive_mode_prunes_early_on_public_estimator_surface() {
    let features = dense_design(40, 2);
    let targets = Array1::from_elem(40, 11.0);
    let candidates: Vec<SKMockConstantRegressor> = (0..27)
        .map(|index| SKMockConstantRegressor::new(index as f64))
        .collect();

    let non_aggressive = SKHalvingGridSearchRegressor::build()
        .candidates(candidates.clone())
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .factor(3)
        .minimum_resource(12)
        .maximum_resource(36)
        .aggressive_elimination(false)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(non_aggressive.candidate_count_trajectory(), vec![27, 9, 3]);

    let aggressive = SKHalvingGridSearchRegressor::build()
        .candidates(candidates)
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .factor(3)
        .minimum_resource(12)
        .maximum_resource(36)
        .aggressive_elimination(true)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(aggressive.resource_schedule(), vec![12, 36]);
    assert_eq!(aggressive.candidate_count_trajectory(), vec![27, 3, 1]);
    assert_eq!(
        aggressive.elimination_trace()[0]
            .aggressively_pruned_candidate_indices()
            .len(),
        6
    );
    assert_eq!(aggressive.best_candidate_index(), 11);
}
