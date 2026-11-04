//! Tests for `SKPolynomialFeatures` (spec `polynomial-features`).
//!
//! This sibling owns the shared default expander and the expansion and
//! validation tests.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError};

use super::core_implementation::SKPolynomialFeatures;

/// The default-configured expander every test starts from.
pub(super) fn default_expander() -> SKPolynomialFeatures {
    SKPolynomialFeatures::build()
        .build()
        .expect("default expander builds")
}

/// The hand-checkable degree-two expansion of one row.
#[test]
fn hand_checkable_expansion() {
    let model = default_expander()
        .fit(
            Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0])
                .unwrap()
                .view(),
        )
        .unwrap();
    let output: Array2<f64> = model
        .transform_dense(array![[2.0_f64, 3.0]].view())
        .unwrap();
    assert_eq!(output, array![[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]);
    assert_eq!(model.number_of_output_features(), 6);
    assert_eq!(model.number_of_input_features(), 2);
}

/// Interaction-only drops the pure power terms.
#[test]
fn interaction_only_drops_powers() {
    let features = Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0]).unwrap();
    let model = SKPolynomialFeatures::build()
        .interaction_only(true)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    // bias + x0 + x1 + x0*x1 = 4 features.
    assert_eq!(model.number_of_output_features(), 4);
    let output: Array2<f64> = model.transform_dense(features.view()).unwrap();
    assert_eq!(output, array![[1.0, 2.0, 3.0, 6.0]]);
}

/// Excluding the bias drops the constant column.
#[test]
fn include_bias_false_drops_constant() {
    let features = Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0]).unwrap();
    let model = SKPolynomialFeatures::build()
        .include_bias(false)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    assert_eq!(model.number_of_output_features(), 5);
    let output: Array2<f64> = model.transform_dense(features.view()).unwrap();
    assert_eq!(output, array![[2.0, 3.0, 4.0, 6.0, 9.0]]);
}

/// The documented column order is stable and deterministic.
#[test]
fn documented_column_order_is_stable() {
    let features = Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0]).unwrap();
    let first = default_expander().fit(features.view()).unwrap();
    let second = default_expander().fit(features.view()).unwrap();
    assert_eq!(first.terms(), second.terms());
    assert_eq!(
        first.terms(),
        &[vec![], vec![0], vec![1], vec![0, 0], vec![0, 1], vec![1, 1]]
    );
}

/// A zero degree is rejected at build time.
#[test]
fn zero_degree_is_rejected() {
    let result = SKPolynomialFeatures::build().degree(0).build();
    assert!(matches!(result, Err(SKError::InvalidHyperparameter { .. })));
}

/// Empty input is rejected before computing anything.
#[test]
fn empty_input_errors() {
    let result = default_expander().fit(Array2::<f64>::zeros((0, 2)).view());
    assert!(result.is_err());
}

/// Sparse input is rejected at the dense-only input boundary.
#[test]
fn sparse_input_rejected() {
    let sparse = sprs::CsMat::new_csc((1, 2), vec![0, 0, 0], vec![], Vec::<f64>::new());
    let result = default_expander().fit(sparse.view());
    assert!(matches!(
        result,
        Err(SKError::UnsupportedRepresentation { .. })
    ));
}

/// A transform with an unexpected column count fails.
#[test]
fn transform_column_mismatch_errors() {
    let features = Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0]).unwrap();
    let model = default_expander().fit(features.view()).unwrap();
    let result: Result<Array2<f64>, _> = model.transform_dense(
        Array2::from_shape_vec((1, 3), vec![1.0_f64, 2.0, 3.0])
            .unwrap()
            .view(),
    );
    assert!(result.is_err());
}

/// The backend defaults to Faer and flows into the fitted model.
#[test]
fn backend_default_and_override() {
    use sciencekit_common::SKBackendKind;

    let features = Array2::from_shape_vec((1, 2), vec![2.0_f64, 3.0]).unwrap();
    let default_model = default_expander().fit(features.view()).unwrap();
    assert_eq!(default_model.backend(), SKBackendKind::Faer);

    let custom = SKPolynomialFeatures::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    assert_eq!(custom.backend(), SKBackendKind::MatrixMultiply);
}
