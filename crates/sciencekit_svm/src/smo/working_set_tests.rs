//! Companion TDD tests for SMO gradient state and second-order working-set
//! selection (spec `support-vector-machines`).

use super::{SKSmoState, sk_select_first_working_index, sk_select_working_pair};

#[test]
fn smo_working_set_selects_hand_computed_toy_pair_and_updates_gradients() {
    // 3 points in 1-D: x = [-2.0, -1.0, 2.0], y = [-1, -1, +1], linear kernel:
    // K = [[4, 2, -4], [2, 1, -2], [-4, -2, 4]], diag = [4, 1, 4].
    // At alpha = [0, 0, 0], G = [-1, -1, -1]:
    // -y G = [-1, -1, +1]; I_up = {2} (score +1), I_low = {0, 1} (score -1).
    let state = SKSmoState::new(vec![-1, -1, 1], vec![1.0_f64, 1.0, 1.0], vec![-1.0, -1.0, -1.0]);
    let (first, first_score) = sk_select_first_working_index(&state).unwrap();
    assert_eq!(first, 2);
    assert!((first_score - 1.0).abs() < 1e-12);

    // Pair with first = 2:
    // t = 0: b = 2, a = 4 + 4 - 2*(-4) = 16 => gain = 4 / 32 = 0.125
    // t = 1: b = 2, a = 4 + 1 - 2*(-2) = 9  => gain = 4 / 18 = 2/9 ≈ 0.2222
    // So second-order selection picks t = 1 over t = 0!
    let diagonal = [4.0_f64, 1.0, 4.0];
    let row_2 = [-4.0_f64, -2.0, 4.0];
    let pair = sk_select_working_pair(&state, first, first_score, &diagonal, &row_2, 3).unwrap();
    assert_eq!(pair.first, 2);
    assert_eq!(pair.second, 1);
    assert!((pair.violation - 2.0).abs() < 1e-12);
    assert!((pair.second_order_gain - (2.0 / 9.0)).abs() < 1e-12);
}

#[test]
fn smo_working_set_maximizes_second_order_gain_among_all_lower_candidates() {
    // 4 variables: first = 0 (y = +1, G = -1.5 => -yG = +1.5),
    // candidates 1, 2, 3 with y = -1 and varying gradients and curvatures.
    let mut state = SKSmoState::new(
        vec![1, -1, -1, -1],
        vec![2.0_f64; 4],
        vec![-1.5, -0.5, -1.2, -0.9],
    );
    let (first, first_score) = sk_select_first_working_index(&state).unwrap();
    assert_eq!(first, 0);

    let diagonal = [2.0_f64, 2.0, 1.2, 5.0];
    let row_0 = [2.0_f64, 0.5, 0.9, -1.0];
    // Candidate gains b^2 / (2 a) where -y_t G_t = G_t (since y_t = -1):
    // t=1: b = 1.5 - (-0.5) = 2.0, a = 2 + 2 - 1 = 3.0   => gain = 4 / 6 = 0.6667
    // t=2: b = 1.5 - (-1.2) = 2.7, a = 2 + 1.2 - 1.8 = 1.4 => gain = 7.29 / 2.8 = 2.60357
    // t=3: b = 1.5 - (-0.9) = 2.4, a = 2 + 5 + 2 = 9.0   => gain = 5.76 / 18 = 0.32
    let pair = sk_select_working_pair(&state, first, first_score, &diagonal, &row_0, 4).unwrap();
    assert_eq!(pair.second, 2);

    for candidate in [1usize, 2, 3] {
        let b = first_score - state.negative_signed_gradient(candidate);
        let a = diagonal[0] + diagonal[candidate] - 2.0 * row_0[candidate];
        let gain = (b * b) / (2.0 * a);
        assert!(pair.second_order_gain + 1e-12 >= gain);
    }

    // Also verify gradient update on a two-variable step.
    let row_2 = [0.9_f64, 0.1, 1.2, -0.3];
    state.update_pair(0, 2, 0.5, 0.5, &row_0, &row_2, 4);
    assert!((state.alphas[0] - 0.5).abs() < 1e-12);
    assert!((state.alphas[2] - 0.5).abs() < 1e-12);
    // For t=0 (y=+1): G_0 = -1.5 + (+1)*(+1*0.5*2.0 + (-1)*0.5*0.9) = -1.5 + 1.0 - 0.45 = -0.95
    assert!((state.gradients[0] - (-0.95)).abs() < 1e-12);
}
