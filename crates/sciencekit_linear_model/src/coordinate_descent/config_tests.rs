//! Configuration hyperparameter tests for the coordinate-descent substrate
//! (spec `coordinate-descent-solver`).

use ndarray::{Array1, array};
use sciencekit_common::SKError;

use super::config::{
    SKCoordinateDescentConfig, SKIntoAlphas, SKSelection, sk_validate_coordinate_descent_hyperparameters,
    sk_validate_l1_ratio,
};

/// A scalar penalty converts to a one-element sequence.
#[test]
fn scalar_alpha_converts_to_one_element() {
    assert_eq!(2.5_f64.into_alphas(), vec![2.5]);
}

/// Every accepted sequence form converts to the same owned vector, in order.
#[test]
fn sequence_forms_convert_in_order() {
    assert_eq!(vec![1.0_f64, 2.0, 3.0].into_alphas(), vec![1.0, 2.0, 3.0]);
    assert_eq!(
        array![1.0_f64, 2.0, 3.0].into_alphas(),
        vec![1.0, 2.0, 3.0]
    );
    let slice: &[f64] = &[1.0, 2.0, 3.0];
    assert_eq!(slice.into_alphas(), vec![1.0, 2.0, 3.0]);
    assert_eq!([1.0_f64, 2.0, 3.0].into_alphas(), vec![1.0, 2.0, 3.0]);
}

/// The default configuration is documented.
#[test]
fn default_configuration_is_documented() {
    let config = SKCoordinateDescentConfig::<f64>::default();
    assert_eq!(config.tolerance, 1e-4);
    assert_eq!(config.maximum_number_of_iterations, 1000);
    assert!(!config.positive);
    assert_eq!(config.selection, SKSelection::Cyclic);
    assert!(config.screening);
}

/// A valid configuration validates without error.
#[test]
fn valid_hyperparameters_validate() {
    assert!(
        sk_validate_coordinate_descent_hyperparameters(&[1.0_f64, 2.0], 1e-4, 1000).is_ok()
    );
    assert!(sk_validate_l1_ratio(0.5_f64).is_ok());
    assert!(sk_validate_l1_ratio(1.0_f64).is_ok());
    assert!(sk_validate_l1_ratio(0.0_f64).is_ok());
}

/// An empty penalty sequence is rejected.
#[test]
fn empty_alpha_sequence_is_rejected() {
    let error =
        sk_validate_coordinate_descent_hyperparameters(&[] as &[f64], 1e-4, 1000).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// A negative penalty is rejected.
#[test]
fn negative_alpha_is_rejected() {
    let error =
        sk_validate_coordinate_descent_hyperparameters(&[1.0, -1.0], 1e-4, 1000).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "alpha", .. }
    ));
}

/// A non-positive tolerance is rejected.
#[test]
fn non_positive_tolerance_is_rejected() {
    let error =
        sk_validate_coordinate_descent_hyperparameters(&[1.0], 0.0, 1000).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
}

/// A zero iteration budget is rejected.
#[test]
fn zero_iteration_budget_is_rejected() {
    let error =
        sk_validate_coordinate_descent_hyperparameters(&[1.0], 1e-4, 0).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_iterations",
            ..
        }
    ));
}

/// An out-of-range `l1_ratio` is rejected.
#[test]
fn out_of_range_l1_ratio_is_rejected() {
    for value in [-0.1_f64, 1.1] {
        let error = sk_validate_l1_ratio(value).unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "l1_ratio",
                ..
            }
        ));
    }
}

/// The `Array1` conversion is exercised through the trait method on a value.
#[test]
fn array_conversion_uses_trait_method() {
    let values: Array1<f64> = Array1::from_vec(vec![4.0, 5.0]);
    assert_eq!(values.into_alphas(), vec![4.0, 5.0]);
}
