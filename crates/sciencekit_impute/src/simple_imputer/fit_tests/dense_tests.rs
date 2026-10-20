//! Companion tests for the dense batch fit (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKUnsupervisedFit};

use crate::simple_imputer::core_implementation::SKSimpleImputer;
use crate::simple_imputer::strategy::SKImputationStrategy;

#[test]
fn mean_over_the_observed_values() {
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let model = imputer.fit(&array![[7.0, f64::NAN], [10.0, 1.0]]).unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[8.5, 1.0]);
    assert!(model.dropped_columns().is_empty());
    assert_eq!(model.number_of_input_columns(), 2);
    assert_eq!(model.number_of_output_features(), 2);
}

#[test]
fn median_of_an_odd_observed_count() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[1.0], [f64::NAN], [3.0], [5.0]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[3.0]);
}

#[test]
fn median_of_an_even_observed_count_averages_the_middle_pair() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[1.0], [3.0], [f64::NAN], [9.0], [5.0], [f64::NAN]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[4.0]);
}

#[test]
fn most_frequent_tie_breaks_toward_the_smallest() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap();
    // Column 0 has a strict mode (2 twice, 5 once); column 1 ties at two
    // occurrences each, so the smallest of the tied values must win.
    let model = imputer
        .fit(&array![[2.0, 5.0], [5.0, 5.0], [2.0, 1.0], [f64::NAN, 1.0]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[2.0, 1.0]);
}

#[test]
fn constant_fills_every_column_with_the_configured_value() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Constant)
        .fill_value(Some(-1.0))
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[1.0, f64::NAN], [f64::NAN, f64::NAN]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[-1.0, -1.0]);
    assert!(
        model.dropped_columns().is_empty(),
        "a constant strategy has no empty columns"
    );
}

#[test]
fn all_missing_column_gets_a_nan_statistic_and_is_reported() {
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let model = imputer
        .fit(&array![[1.0, f64::NAN], [2.0, f64::NAN]])
        .unwrap();
    let statistics = model.statistics();
    assert!(statistics[1].is_nan());
    assert_eq!(model.dropped_columns(), &[1usize]);
}

#[test]
fn the_model_records_the_carried_backend_kind() {
    let imputer = SKSimpleImputer::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = imputer.fit(&array![[1.0]]).unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

#[test]
fn fit_leaves_the_estimator_reusable_and_deterministic() {
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let data = array![[1.0, f64::NAN], [3.0, 8.0], [5.0, 10.0]];
    let first = imputer.fit(&data).unwrap();
    let second = imputer.fit(&data).unwrap();
    assert_eq!(
        first.statistics().as_slice().unwrap(),
        second.statistics().as_slice().unwrap()
    );
}

#[test]
fn a_nan_value_under_a_zero_marker_is_observed_so_the_column_stays() {
    // Under a `0.0` marker a `NaN` cell is an observed value, so column 0 holds
    // data even though its mean is `NaN`: it must stay in the output.
    let features = array![[f64::NAN, 2.0], [1.0, 3.0]];
    let model = SKSimpleImputer::<f64>::build()
        .missing_values(0.0)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();

    assert!(model.statistics()[0].is_nan());
    assert!(model.dropped_columns().is_empty());
    assert_eq!(model.number_of_output_features(), 2);
}
