//! Companion tests for stacking on dense inputs (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::dense_imputation;
use crate::simple_imputer::core_implementation::SKSimpleImputer;

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
fn incremental_batches_extend_the_indicator_scope_like_the_whole() {
    let first = array![[1.0, 2.0], [3.0, 4.0]];
    let second = array![[5.0, f64::NAN], [7.0, 8.0]];
    let whole = array![[1.0, 2.0], [3.0, 4.0], [5.0, f64::NAN], [7.0, 8.0]];

    let concatenated = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&whole)
        .unwrap();
    let mut incremental = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&first)
        .unwrap();
    incremental.partial_fit(&second).unwrap();

    assert_eq!(
        incremental.indicator().unwrap().scope(),
        concatenated.indicator().unwrap().scope(),
        "the second batch contributes the missing column"
    );
    assert_eq!(incremental.statistics(), concatenated.statistics());
    let probe = array![[f64::NAN, f64::NAN]];
    assert_eq!(
        dense_imputation(incremental.transform(&probe).unwrap()),
        dense_imputation(concatenated.transform(&probe).unwrap())
    );
}
