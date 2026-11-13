//! Companion TDD tests for class-weight edge cases in the linear SVC
//! (spec `linear-support-vector-classifier`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::{SKLinearSVC, SKLinearSVCLoss};
use crate::svc::SKClassWeight;

#[test]
fn explicit_class_weights_reach_the_solver_in_their_own_class_role() {
    // No intercept, squared hinge, C = 1, rows x = -1 (class 0) and x = +2
    // (class 1), weights a and b. Both dual coordinates stay interior, so
    // stationarity gives
    //     0.5 w² + a (1 - w)² + b (1 - 2w)²
    //     ⇒ w = (2a + 4b) / (1 + 2a + 8b).
    // Swapping the roles of `a` and `b` would move `w` elsewhere, so matching
    // the closed form proves each class receives its own weight.
    let features = array![[-1.0_f64], [2.0]];
    let labels = array![0_i64, 1];

    [(0.1_f64, 1.0_f64), (0.1, 2.0)]
        .into_iter()
        .for_each(|(class_zero_weight, class_one_weight)| {
            let model = SKLinearSVC::<f64>::build()
                .loss(SKLinearSVCLoss::SquaredHinge)
                .include_intercept(false)
                .tolerance(1e-12)
                .class_weight(SKClassWeight::Explicit(vec![
                    class_zero_weight,
                    class_one_weight,
                ]))
                .build()
                .unwrap()
                .fit(features.view(), labels.view())
                .unwrap();

            let expected = (2.0 * class_zero_weight + 4.0 * class_one_weight)
                / (1.0 + 2.0 * class_zero_weight + 8.0 * class_one_weight);
            let fitted = model.heads()[0].coefficients()[0];
            assert!(
                (fitted - expected).abs() < 1e-9,
                "weights ({class_zero_weight}, {class_one_weight}) produced {fitted}, expected {expected}"
            );
        });
}

#[test]
fn explicit_zero_weight_class_is_ignored_without_breaking_convergence() {
    // `Explicit` accepts a zero weight (the validator only requires finite,
    // non-negative values with at least one positive entry). A zero-weighted
    // class contributes nothing to the objective, so the solver must drop its
    // rows instead of reporting their permanently violating gradient.
    let features = array![[-2.0_f64], [-1.0], [-0.5], [0.2], [0.4], [0.8]];
    let labels = array![0_i64, 0, 0, 0, 1, 1];

    [SKLinearSVCLoss::Hinge, SKLinearSVCLoss::SquaredHinge]
        .into_iter()
        .for_each(|loss| {
            let model = SKLinearSVC::<f64>::build()
                .loss(loss)
                .class_weight(SKClassWeight::Explicit(vec![0.0, 1.0]))
                .build()
                .unwrap()
                .fit(features.view(), labels.view())
                .unwrap_or_else(|error| {
                    panic!("zero-weight class must not break the {loss:?} fit: {error:?}")
                });

            assert!(
                model.heads()[0]
                    .coefficients()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(model.heads()[0].intercept().is_finite());
        });
}
