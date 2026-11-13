//! Companion tests for binary and multiclass log-loss (task 2.3).

use ndarray::array;

use super::{SKBinaryLogLoss, SKGradientLoss, SKMulticlassLogLoss};

/// Binary log-loss initializes at log-odds `ln(p / (1 - p))`, maps raw score `0`
/// to probability `0.5` without overflow at extreme scores, and produces
/// negative gradient `y - p` and Hessian `p * (1 - p)`.
#[test]
fn binary_log_loss_matches_hand_computed_log_odds_gradients_and_hessians() {
    let loss = SKBinaryLogLoss::new();
    // Positive prior = 3/4 = 0.75 -> log-odds = ln(0.75 / 0.25) = ln(3).
    let targets = [0.0_f64, 1.0, 1.0, 1.0];
    let initial = loss.initial_prediction(&targets);
    assert!((initial - 3.0_f64.ln()).abs() < 1e-12);
    assert!((loss.inverse_link(initial) - 0.75).abs() < 1e-12);
    assert!((loss.link(0.75) - initial).abs() < 1e-12);

    // Scenario: Binary probabilities are sigmoid of the score -> score 0.0 => 0.5.
    assert!((loss.inverse_link(0.0_f64) - 0.5).abs() < 1e-12);
    assert!(loss.inverse_link(500.0_f64).is_finite());
    assert!(loss.inverse_link(-500.0_f64).is_finite());

    // Scenario: Log-loss gradient -> at p = 0.25 (raw = ln(1/3)),
    // negative_gradient = y - p and hessian = p(1 - p) = 0.1875.
    let raw_for_quarter = (1.0_f64 / 3.0).ln();
    assert!((loss.negative_gradient(1.0, raw_for_quarter) - 0.75).abs() < 1e-12);
    assert!((loss.negative_gradient(0.0, raw_for_quarter) - (-0.25)).abs() < 1e-12);
    assert!((loss.hessian(1.0, raw_for_quarter) - 0.1875).abs() < 1e-12);
    assert!((loss.hessian(0.0, raw_for_quarter) - 0.1875).abs() < 1e-12);

    // Pointwise cross-entropy at raw = 0 (p = 0.5) is ln(2) for both y = 0 and y = 1.
    assert!((loss.loss(0.0_f64, 1.0) - 2.0_f64.ln()).abs() < 1e-12);
    assert!((loss.loss(0.0_f64, 0.0) - 2.0_f64.ln()).abs() < 1e-12);
}

/// Multiclass log-loss computes class log-priors, shift-invariant softmax
/// probabilities summing to one, and `C` per-class negative-gradient and
/// diagonal-Hessian vectors per stage.
#[test]
fn multiclass_log_loss_matches_hand_computed_softmax_and_class_vectors() {
    let loss = SKMulticlassLogLoss::build()
        .number_of_classes(3)
        .build()
        .expect("3-class log-loss builds");

    // Counts: [1, 2, 1] over 4 rows -> priors [0.25, 0.5, 0.25].
    let class_indices = [0usize, 1, 1, 2];
    let initials: Vec<f64> = loss.initial_predictions(&class_indices);
    assert!((initials[0] - 0.25_f64.ln()).abs() < 1e-12);
    assert!((initials[1] - 0.5_f64.ln()).abs() < 1e-12);
    assert!((initials[2] - 0.25_f64.ln()).abs() < 1e-12);

    let probabilities = loss.softmax(&initials);
    assert!((probabilities[0] - 0.25).abs() < 1e-12);
    assert!((probabilities[1] - 0.50).abs() < 1e-12);
    assert!((probabilities[2] - 0.25).abs() < 1e-12);
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Shift invariance and overflow safety on large raw scores.
    let shifted = loss.softmax(&[1000.0_f64, 1000.0 + 2.0_f64.ln(), 1000.0]);
    assert!((shifted[0] - 0.25).abs() < 1e-12);
    assert!((shifted[1] - 0.50).abs() < 1e-12);
    assert!((shifted[2] - 0.25).abs() < 1e-12);

    // Scenario: Multiclass gradients -> 3 classes produce 3 gradient/Hessian vectors.
    let raw_scores = array![
        [0.0_f64, 0.0, 0.0],
        [0.25_f64.ln(), 0.5_f64.ln(), 0.25_f64.ln()],
    ];
    let (negative_gradients, hessians) =
        loss.stage_gradients_and_hessians(&[0usize, 1], &raw_scores);
    assert_eq!(negative_gradients.len(), 3);
    assert_eq!(hessians.len(), 3);

    // Row 0 has uniform p = 1/3 and true class 0 -> neg_grad = [2/3, -1/3, -1/3], h = 2/9.
    assert!((negative_gradients[0][0] - (2.0 / 3.0)).abs() < 1e-12);
    assert!((negative_gradients[1][0] - (-1.0 / 3.0)).abs() < 1e-12);
    assert!((negative_gradients[2][0] - (-1.0 / 3.0)).abs() < 1e-12);
    assert!((hessians[0][0] - (2.0 / 9.0)).abs() < 1e-12);

    // Row 1 has p = [0.25, 0.5, 0.25] and true class 1 -> neg_grad = [-0.25, 0.5, -0.25],
    // hessians = [0.1875, 0.25, 0.1875], loss = -ln(0.5) = ln(2).
    assert!((negative_gradients[0][1] - (-0.25)).abs() < 1e-12);
    assert!((negative_gradients[1][1] - 0.5).abs() < 1e-12);
    assert!((hessians[0][1] - 0.1875).abs() < 1e-12);
    assert!((hessians[1][1] - 0.25).abs() < 1e-12);
    assert!(
        (loss.loss_for_row(raw_scores.row(1).as_slice().expect("slice"), 1) - 2.0_f64.ln()).abs()
            < 1e-12
    );
}
