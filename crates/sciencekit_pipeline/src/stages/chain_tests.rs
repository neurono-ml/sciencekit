//! Tests for the nested-tuple stage chain (spec `typed-pipeline`).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKDataView, SKFeatureBlock};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::SKConfiguredStages;

fn scaler() -> SKStandardScaler {
    SKStandardScaler::build().build().unwrap()
}

fn block() -> SKFeatureBlock<f64> {
    SKFeatureBlock::from(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap())
}

#[test]
fn the_empty_chain_has_no_stages_and_passes_the_block_through() {
    let chain = ();
    assert_eq!(SKConfiguredStages::<f64>::number_of_stages(&chain), 0);
    let (fitted, output) = chain.fit_stages(&block()).unwrap();
    let output = chain.transform_stages(&fitted, &output).unwrap();
    assert!(!output.is_sparse());
}

#[test]
fn a_chain_counts_every_appended_stage() {
    let chain = ((), scaler());
    assert_eq!(SKConfiguredStages::<f64>::number_of_stages(&chain), 1);
    let chain = ((chain, scaler()), scaler());
    assert_eq!(SKConfiguredStages::<f64>::number_of_stages(&chain), 3);
}

#[test]
fn the_chain_threads_the_output_of_one_stage_into_the_next() {
    let chain = ((), scaler());
    let (fitted, output) = chain.fit_stages(&block()).unwrap();
    // The first scaler was fitted on the input and applied; the result is dense.
    assert!(!output.is_sparse());
    assert_eq!(output.number_of_rows(), 2);
    let transformed = chain.transform_stages(&fitted, &block()).unwrap();
    match transformed.view() {
        SKDataView::Dense(view) => assert!((view[[0, 0]] + 1.0).abs() < 1e-12),
        _ => panic!("the scaler produces a dense block"),
    }
}
