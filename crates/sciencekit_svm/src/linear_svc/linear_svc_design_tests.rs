//! Companion TDD tests for the linear SVC design abstraction
//! (spec `linear-support-vector-classifier`).

use ndarray::array;

use super::design::{SKDenseLinearSvcDesign, SKLinearSvcDesignAccess};

#[test]
fn dense_design_row_dot_matches_the_hand_computed_augmented_row() {
    let features = array![[1.0_f64, 2.0], [3.0, -1.0]];
    let design = SKDenseLinearSvcDesign::new(features.view(), true);

    assert_eq!(design.number_of_rows(), 2);
    assert_eq!(design.number_of_columns(), 2);
    assert!(design.has_intercept());

    // coefficients = [w_0, w_1, bias]; row 0 = [1, 2, 1], row 1 = [3, -1, 1].
    let coefficients = [0.5_f64, -1.5, 2.0];
    assert!((design.row_dot(0, &coefficients) - (0.5 - 3.0 + 2.0)).abs() < 1e-12);
    assert!((design.row_dot(1, &coefficients) - (1.5 + 1.5 + 2.0)).abs() < 1e-12);

    assert!((design.row_norm_squared(0) - (1.0 + 4.0 + 1.0)).abs() < 1e-12);
    assert!((design.row_norm_squared(1) - (9.0 + 1.0 + 1.0)).abs() < 1e-12);
}

#[test]
fn dense_design_adds_scaled_augmented_rows_into_the_target() {
    let features = array![[1.0_f64, 2.0], [3.0, -1.0]];
    let design = SKDenseLinearSvcDesign::new(features.view(), true);

    let mut target = [0.0_f64; 3];
    design.add_scaled_row(0, 2.0, &mut target);
    design.add_scaled_row(1, -1.0, &mut target);

    assert!((target[0] - (2.0 - 3.0)).abs() < 1e-12);
    assert!((target[1] - (4.0 + 1.0)).abs() < 1e-12);
    assert!((target[2] - (2.0 - 1.0)).abs() < 1e-12);
}

#[test]
fn dense_design_without_intercept_uses_the_stored_columns_only() {
    let features = array![[1.0_f64, 2.0]];
    let design = SKDenseLinearSvcDesign::new(features.view(), false);
    let coefficients = [2.0_f64, 3.0];
    assert!((design.row_dot(0, &coefficients) - 8.0).abs() < 1e-12);
    assert!((design.row_norm_squared(0) - 5.0).abs() < 1e-12);
}
