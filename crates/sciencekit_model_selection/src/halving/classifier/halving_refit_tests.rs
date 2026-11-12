//! TDD tests for classifier halving refit and prediction delegation
//! (spec `halving-grid-search`, requirement `Refit and prediction`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKExecutionMode, SKSupervisedFit,
};

use crate::grid_search::SKClassifierScoring;
use crate::grid_search::mock_estimators::SKMockThresholdClassifier;
use crate::halving::SKHalvingGridSearchClassifier;
use crate::split::stratified::SKStratifiedKFold;

#[test]
fn halving_refit_classifier_trains_winner_on_all_rows_and_delegates_predictions() {
    // 90 rows with feature 0 = row as f64, label = 1 iff row >= 45.
    // Threshold 45.0 is exact (accuracy = 1.0); 9 candidates at [5, 15, ..., 85] have index 4 = 45.0.
    let features = Array2::from_shape_fn((90, 2), |(row, col)| (row + col) as f64);
    let labels = Array1::from_iter((0..90).map(|row| i64::from(row >= 45)));
    let candidates: Vec<SKMockThresholdClassifier> = (0..9)
        .map(|index| SKMockThresholdClassifier::new((index * 10 + 5) as f64))
        .collect();

    let estimator = SKHalvingGridSearchClassifier::build()
        .candidates(candidates)
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .shuffle(true)
                .random_seed(11)
                .build()
                .unwrap(),
        )
        .factor(3)
        .minimum_resource(18)
        .random_seed(23)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    let model = estimator.fit(features.view(), labels.view()).unwrap();
    assert!(model.is_refitted());
    assert_eq!(model.best_candidate_index(), 4);
    assert_eq!(model.best_parameters().threshold(), 45.0);
    assert_eq!(model.best_score(), 1.0);
    assert_eq!(model.candidate_count_trajectory(), vec![9, 3, 1]);
    assert_eq!(model.resource_schedule(), vec![18, 54]);
    assert_eq!(model.best_model().unwrap().training_rows(), 90);
    assert_eq!(model.scoring(), SKClassifierScoring::Accuracy);
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let predicted_labels = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted_labels, labels);

    let probabilities = model.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.dim(), (90, 2));
    assert_eq!(probabilities[[0, 0]], 1.0);
    assert_eq!(probabilities[[89, 1]], 1.0);
}
