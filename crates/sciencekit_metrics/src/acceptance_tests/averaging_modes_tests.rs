//! Every averaging mode produces defined scores on the same evaluation.

use crate::classification::{
    SKAveraging, sk_accuracy_score, sk_f1_score, sk_precision_score, sk_recall_score,
};

const TRUE_LABELS: [i64; 6] = [0, 0, 0, 0, 1, 2];
const PREDICTED_LABELS: [i64; 6] = [0, 0, 1, 1, 1, 2];

/// Every mode yields finite scores inside the unit interval.
#[test]
fn every_averaging_mode_is_defined() {
    let modes = [
        SKAveraging::Macro,
        SKAveraging::Micro,
        SKAveraging::Weighted,
        SKAveraging::Binary { positive_label: 1 },
    ];
    for averaging in modes {
        for score in [
            sk_precision_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS, None, averaging)
                .expect("valid input"),
            sk_recall_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS, None, averaging)
                .expect("valid input"),
            sk_f1_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS, None, averaging)
                .expect("valid input"),
        ] {
            assert!(score.is_finite(), "score must be finite for {averaging:?}");
            assert!(
                (0.0..=1.0).contains(&score),
                "score {score} out of range for {averaging:?}"
            );
        }
    }
}

/// Micro-averaged F1 equals accuracy for single-label classification.
#[test]
fn micro_f1_equals_accuracy() {
    let accuracy =
        sk_accuracy_score::<f64>(&TRUE_LABELS, &PREDICTED_LABELS).expect("valid input");
    let micro = sk_f1_score::<f64>(
        &TRUE_LABELS,
        &PREDICTED_LABELS,
        None,
        SKAveraging::Micro,
    )
    .expect("valid input");
    assert!((micro - accuracy).abs() < 1e-12);
}
