//! Companion tests for the dense transform (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::dense;
use crate::simple_imputer::core_implementation::SKSimpleImputer;
use crate::simple_imputer::strategy::SKImputationStrategy;

#[test]
fn dense_fills_only_the_missing_cells() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [3.0, 8.0], [5.0, 10.0]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, 8.0], [3.0, f64::NAN], [7.0, 12.0]])
            .unwrap(),
    );
    assert_eq!(
        imputed,
        array![[3.0, 8.0], [3.0, 9.0], [7.0, 12.0]],
        "the statistic fills the missing cells and leaves the observed ones"
    );
}

#[test]
fn observed_cells_stay_bit_identical() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[0.1, 1.0], [0.1, 1.0 / 3.0]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[0.1, 1.0 / 3.0], [f64::NAN, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(
        imputed[[0, 0]].to_bits(),
        0.1f64.to_bits(),
        "an observed cell is copied, never recomputed"
    );
    assert_eq!(imputed[[0, 1]].to_bits(), (1.0f64 / 3.0).to_bits());
}

#[test]
fn column_count_mismatch_is_rejected() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0]])
        .unwrap();
    let error = model
        .transform(&array![[1.0, 2.0, 3.0]])
        .expect_err("three columns cannot be transformed by a two-column model");
    assert!(
        error.to_string().contains("shape mismatch"),
        "the error reports the shape contract: {error}"
    );
}

#[test]
fn dense_input_produces_dense_output() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0], [f64::NAN]])
        .unwrap();
    let output = model.transform(&array![[f64::NAN], [2.0]]).unwrap();
    assert_eq!(output.representation(), "dense");
    assert_eq!((output.nrows(), output.ncols()), (2, 1));
}

#[test]
fn all_missing_column_is_dropped_by_default() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [2.0, f64::NAN]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, f64::NAN], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(imputed, array![[1.5], [2.0]]);
}

#[test]
fn keep_empty_features_retains_the_column_as_zero() {
    let model = SKSimpleImputer::<f64>::build()
        .keep_empty_features(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [2.0, f64::NAN]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, f64::NAN], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(imputed, array![[1.5, 0.0], [2.0, 0.0]]);
}

#[test]
fn constant_strategy_fills_every_missing_cell_and_keeps_the_columns() {
    let model = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Constant)
        .fill_value(Some(-1.0))
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [f64::NAN, f64::NAN]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, f64::NAN], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(imputed, array![[-1.0, -1.0], [2.0, -1.0]]);
}
