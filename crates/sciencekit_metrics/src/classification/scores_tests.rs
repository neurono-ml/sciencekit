//! TDD tests for the classification score functions (spec `evaluation-metrics`).

use super::{SKAveraging, sk_f1_score, sk_precision_score, sk_recall_score};

/// The shared unbalanced example: truth [0,0,0,0,1], predictions [0,0,0,1,1].
const TRUE_LABELS: [i64; 5] = [0, 0, 0, 0, 1];
const PREDICTED_LABELS: [i64; 5] = [0, 0, 0, 1, 1];

/// Precision matches the hand-computed value for every averaging mode.
#[test]
fn precision_matches_hand_computed_values() {
    let cases = [
        (SKAveraging::Macro, 0.75),
        (SKAveraging::Micro, 0.8),
        (SKAveraging::Weighted, 0.9),
        (SKAveraging::Binary { positive_label: 1 }, 0.5),
    ];
    for (averaging, expected) in cases {
        let score = sk_precision_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS, None, averaging)
            .expect("valid input");
        assert!(
            (score - expected).abs() < 1e-12,
            "precision {averaging:?}: {score} vs {expected}"
        );
    }
}

/// Recall matches the hand-computed value for every averaging mode.
#[test]
fn recall_matches_hand_computed_values() {
    let cases = [
        (SKAveraging::Macro, 0.875),
        (SKAveraging::Micro, 0.8),
        (SKAveraging::Weighted, 0.8),
        (SKAveraging::Binary { positive_label: 1 }, 1.0),
    ];
    for (averaging, expected) in cases {
        let score = sk_recall_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS, None, averaging)
            .expect("valid input");
        assert!(
            (score - expected).abs() < 1e-12,
            "recall {averaging:?}: {score} vs {expected}"
        );
    }
}

/// F1 matches the hand-computed value for every averaging mode.
#[test]
fn f1_matches_hand_computed_values() {
    let cases = [
        (SKAveraging::Macro, 16.0 / 21.0),
        (SKAveraging::Micro, 0.8),
        (SKAveraging::Weighted, 86.0 / 105.0),
        (SKAveraging::Binary { positive_label: 1 }, 2.0 / 3.0),
    ];
    for (averaging, expected) in cases {
        let score = sk_f1_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS, None, averaging)
            .expect("valid input");
        assert!(
            (score - expected).abs() < 1e-12,
            "f1 {averaging:?}: {score} vs {expected}"
        );
    }
}

/// An explicit label space is honoured by the score functions.
#[test]
fn explicit_labels_are_honoured() {
    // With label 2 resolved but unobserved, macro averaging includes its zeros.
    let score =
        sk_precision_score::<f64>(&[0, 0, 1], &[0, 1, 0], Some(&[0, 1, 2]), SKAveraging::Macro)
            .expect("valid input");
    // Class 0: 1/2, class 1: 0, class 2: 0 -> mean 1/6.
    assert!((score - 1.0 / 6.0).abs() < 1e-12);
}
