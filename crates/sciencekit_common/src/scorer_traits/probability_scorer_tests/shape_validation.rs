//! Shape validation of the probability matrix.

use ndarray::{Array2, array};

use super::fixtures::{ArgmaxAccuracy, WrongShapeClassifier};
use crate::SKProbabilityScorer;
use crate::scorer_traits::sk_validate_probability_shape;
use crate::{SKDataView, SKError, SKTargetView};

/// A probability matrix with the wrong row count is a shape mismatch.
#[test]
fn probability_shape_row_mismatch_is_rejected() {
    let model = WrongShapeClassifier {
        rows: 2,
        columns: 2,
    };
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![0_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let error = ArgmaxAccuracy
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect_err("the probability rows disagree with the targets");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A probability matrix with no class column is a shape mismatch.
#[test]
fn probability_shape_empty_columns_are_rejected() {
    let model = WrongShapeClassifier {
        rows: 3,
        columns: 0,
    };
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![0_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let error = ArgmaxAccuracy
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect_err("a probability matrix needs at least one class column");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The shared shape validator rejects empty targets.
#[test]
fn probability_shape_empty_targets_are_rejected() {
    let probabilities = Array2::<f64>::zeros((0, 2));
    let error = sk_validate_probability_shape(0, probabilities.view())
        .expect_err("empty targets are rejected");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// The shared shape validator accepts a matching matrix.
#[test]
fn probability_shape_accepts_matching_matrix() {
    let probabilities = Array2::<f64>::zeros((3, 2));
    sk_validate_probability_shape(3, probabilities.view()).expect("matching shape");
}
