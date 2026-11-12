//! Classifier prediction delegation (spec `grid-search`, requirement
//! `Classifier and regressor prediction`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::SKGridSearchCVClassifier;
use crate::grid_search::mock_estimators::SKMockThresholdClassifier;
use crate::split::kfold::SKKFold;

fn fitted() -> (
    super::SKGridSearchCVClassifierModel<f64, SKMockThresholdClassifier>,
    Array2<f64>,
) {
    let features = Array2::from_shape_fn((12, 2), |(row, column)| (row + column) as f64);
    let labels = Array1::from_iter((0..12).map(|row| i64::from(row >= 6)));
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(2.0),
            SKMockThresholdClassifier::new(6.0),
            SKMockThresholdClassifier::new(10.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), labels.view()).unwrap();
    (model, features)
}

#[test]
fn grid_predict_classifier_labels_equal_the_best_model() {
    let (model, features) = fitted();
    let expected = model
        .best_model()
        .unwrap()
        .predict_labels(features.view())
        .unwrap();
    assert_eq!(model.predict_labels(features.view()).unwrap(), expected);
}

#[test]
fn grid_predict_classifier_probabilities_equal_the_best_model() {
    let (model, features) = fitted();
    let expected = model
        .best_model()
        .unwrap()
        .predict_probabilities(features.view())
        .unwrap();
    assert_eq!(
        model.predict_probabilities(features.view()).unwrap(),
        expected
    );
}
