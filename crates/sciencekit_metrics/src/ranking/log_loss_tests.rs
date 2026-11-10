//! TDD tests for binary and multiclass log loss (spec `ranking-metrics`).
//!
//! Clipping policy: every probability is clipped to
//! `[F::epsilon(), 1 − F::epsilon()]` before the logarithm, so a probability of
//! exactly zero or one yields a finite loss. This matches scikit-learn's
//! per-dtype `np.finfo(dtype).eps` clipping. Probabilities outside `[0, 1]`
//! are rejected.

use ndarray::array;
use sciencekit_common::SKError;

use super::{sk_binary_log_loss, sk_log_loss};

/// A confident correct prediction has a near-zero loss.
#[test]
fn log_loss_confident_correct_prediction_is_low() {
    let probabilities = array![[0.99_f64, 0.01]];
    let loss = sk_log_loss(&[0_i64], probabilities.view(), None).expect("valid input");
    assert!((loss + 0.99_f64.ln()).abs() < 1e-12);
    assert!(loss > 0.0 && loss < 0.02);
}

/// A probability of exactly zero is clipped, never infinite.
#[test]
fn log_loss_zero_probability_is_finite_through_clipping() {
    let probabilities = array![[0.0_f64, 1.0]];
    let loss = sk_log_loss(&[0_i64], probabilities.view(), None).expect("valid input");
    assert!(loss.is_finite());
    assert!((loss + f64::EPSILON.ln()).abs() < 1e-12);
}

/// A multiclass hand-computed fixture: −(ln 0.8 + ln 0.7) / 2.
#[test]
fn log_loss_multiclass_hand_computed() {
    let labels = [0_i64, 1];
    let probabilities = array![[0.8_f64, 0.1, 0.1], [0.2, 0.7, 0.1]];
    let loss = sk_log_loss(&labels, probabilities.view(), None).expect("valid input");
    let expected = -(0.8_f64.ln() + 0.7_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1e-12);
}

/// An explicit class-label list maps columns to arbitrary labels.
#[test]
fn log_loss_explicit_class_labels_are_resolved() {
    let labels = [10_i64, 20];
    let probabilities = array![[0.9_f64, 0.1], [0.4, 0.6]];
    let loss = sk_log_loss(&labels, probabilities.view(), Some(&[10, 20])).expect("valid input");
    let expected = -(0.9_f64.ln() + 0.6_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1e-12);
}

/// A label outside the class list is rejected.
#[test]
fn log_loss_unknown_label_is_rejected() {
    let probabilities = array![[0.9_f64, 0.1]];
    let error = sk_log_loss(&[7_i64], probabilities.view(), Some(&[10, 20]))
        .expect_err("label 7 is not in the class list");
    assert!(matches!(error, SKError::UnknownLabel { label: 7 }));
}

/// A probability above one is rejected at its flattened position.
#[test]
fn log_loss_probability_above_one_is_rejected() {
    let probabilities = array![[1.5_f64, -0.5]];
    let error =
        sk_log_loss(&[0_i64], probabilities.view(), None).expect_err("probability outside [0, 1]");
    assert!(matches!(
        error,
        SKError::ProbabilityOutOfRange { position: 0 }
    ));
}

/// A non-finite probability is rejected at its flattened position.
#[test]
fn log_loss_non_finite_probability_is_rejected() {
    let probabilities = array![[f64::NAN, 0.5]];
    let error =
        sk_log_loss(&[0_i64], probabilities.view(), None).expect_err("non-finite probability");
    assert!(matches!(error, SKError::NonFiniteValue { position: 0 }));
}

/// Empty input is rejected.
#[test]
fn log_loss_empty_input_is_rejected() {
    let probabilities = ndarray::Array2::<f64>::zeros((0, 2));
    let error = sk_log_loss(&[], probabilities.view(), None).expect_err("empty input");
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A row/label mismatch is a shape mismatch.
#[test]
fn log_loss_row_mismatch_is_a_shape_mismatch() {
    let probabilities = ndarray::Array2::<f64>::zeros((2, 2));
    let error = sk_log_loss(&[0_i64], probabilities.view(), None).expect_err("mismatch");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A class list whose length disagrees with the columns is a shape mismatch.
#[test]
fn log_loss_class_label_length_mismatch_is_a_shape_mismatch() {
    let probabilities = ndarray::Array2::<f64>::zeros((1, 3));
    let error = sk_log_loss(&[0_i64], probabilities.view(), Some(&[0, 1]))
        .expect_err("two labels for three columns");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The binary convenience function equals the two-column function.
#[test]
fn binary_log_loss_equals_two_column_log_loss() {
    let labels = [1_i64, 0, 1];
    let positive_probabilities = array![0.8_f64, 0.3, 0.6];
    let binary =
        sk_binary_log_loss(&labels, positive_probabilities.view(), 1).expect("valid input");
    let columns = array![[0.2_f64, 0.8], [0.7, 0.3], [0.4, 0.6]];
    let two_column = sk_log_loss(&labels, columns.view(), None).expect("valid input");
    assert!((binary - two_column).abs() < 1e-12);
}

/// The binary form clips a zero positive probability.
#[test]
fn binary_log_loss_zero_probability_is_finite() {
    let positive_probabilities = array![0.0_f64];
    let loss = sk_binary_log_loss(&[1_i64], positive_probabilities.view(), 1).expect("valid input");
    assert!(loss.is_finite());
}
