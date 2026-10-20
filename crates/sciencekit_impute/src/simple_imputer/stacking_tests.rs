//! Companion tests for the `add_indicator` stacking rule (spec
//! `simple-imputer`).
//!
//! The stacked output is the imputed features followed, column by column, by
//! the missingness mask of the features that carried a missing cell at fit
//! time.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::core_implementation::SKSimpleImputer;
use super::output::SKImputedOutput;

/// Build a CSR matrix from triplets for the sparse cases.
fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

/// The dense payload of an imputation, panicking on a sparse one.
fn dense_imputation(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => {
            panic!("expected a dense imputation, found a csr one: {matrix:?}")
        }
    }
}

#[test]
fn indicator_is_disabled_by_default() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN, 3.0], [4.0, f64::NAN, 6.0]])
        .unwrap();
    assert!(model.indicator().is_none());
    assert_eq!(model.number_of_output_features(), 2);
    let output = dense_imputation(
        model
            .transform(&array![[f64::NAN, f64::NAN, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(output.ncols(), 2, "no mask column without add_indicator");
}

#[test]
fn stacking_appends_one_column_per_missing_feature() {
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN, 3.0], [4.0, f64::NAN, 6.0]])
        .unwrap();
    let indicator = model.indicator().expect("the indicator is fitted");
    assert_eq!(indicator.scope(), &[1]);
    assert_eq!(indicator.number_of_input_columns(), 3);
    let output = dense_imputation(
        model
            .transform(&array![[f64::NAN, f64::NAN, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(
        output,
        array![[2.5, 4.5, 1.0]],
        "the imputed features come first, the mask last"
    );
    assert_eq!(
        output.ncols(),
        model.number_of_output_features() + indicator.number_of_output_columns()
    );
}

#[test]
fn a_retained_empty_column_keeps_its_indicator_column() {
    let model = SKSimpleImputer::<f64>::build()
        .keep_empty_features(true)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN, 3.0], [4.0, f64::NAN, 6.0]])
        .unwrap();
    assert_eq!(model.number_of_output_features(), 3);
    assert_eq!(model.indicator().unwrap().scope(), &[1]);
    let output = dense_imputation(
        model
            .transform(&array![[f64::NAN, f64::NAN, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(output, array![[2.5, 0.0, 4.5, 1.0]]);
}

#[test]
fn new_missingness_is_imputed_but_not_masked() {
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    assert_eq!(
        model.indicator().unwrap().number_of_output_columns(),
        0,
        "no feature was missing at fit time"
    );
    let output = dense_imputation(model.transform(&array![[f64::NAN, 2.0]]).unwrap());
    assert_eq!(output, array![[2.0, 2.0]]);
}

#[test]
fn sparse_stacking_appends_the_mask_columns() {
    let fitted = csr(
        (2, 2),
        &[(0, 0, 1.0), (0, 1, f64::NAN), (1, 0, 3.0), (1, 1, 5.0)],
    );
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(fitted.view())
        .unwrap();
    assert_eq!(model.indicator().unwrap().scope(), &[1]);
    let features = csr(
        (2, 2),
        &[(0, 0, 1.0), (0, 1, f64::NAN), (1, 0, 3.0), (1, 1, 5.0)],
    );
    match model.transform(features.view()).unwrap() {
        SKImputedOutput::Sparse(matrix) => {
            assert_eq!(matrix.shape(), (2, 3), "two features plus one mask");
            assert_eq!(matrix.get(0, 1), Some(&5.0), "the missing cell is filled");
            assert_eq!(
                matrix.get(0, 2),
                Some(&1.0),
                "the mask follows the features"
            );
            assert_eq!(matrix.get(1, 2), None, "an observed cell is not masked");
        }
        SKImputedOutput::Dense(matrix) => panic!("expected a csr result, found {matrix:?}"),
    }
}
