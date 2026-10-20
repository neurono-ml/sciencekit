//! Acceptance on little data: the hand-checkable matrix and the export
//! boundary.

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::dense_imputation;
use crate::missing_indicator::SKMissingFeatureScope;
use crate::simple_imputer::{SKImputationStrategy, SKSimpleImputer};

#[test]
fn the_tiny_matrix_is_hand_checkable_end_to_end() {
    let features = array![
        [1.0, f64::NAN, 10.0],
        [3.0, 4.0, f64::NAN],
        [5.0, 6.0, 30.0]
    ];

    // Column means: (1+3+5)/3 = 3, (4+6)/2 = 5, (10+30)/2 = 20.
    let mean = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    assert_eq!(
        dense_imputation(mean.transform(&features).unwrap()),
        array![[1.0, 5.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]]
    );

    // Column medians: 3, 5 and 20 on the same observed sets.
    let median = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    assert_eq!(
        dense_imputation(median.transform(&features).unwrap()),
        array![[1.0, 5.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]]
    );
}

#[test]
fn the_fitted_state_is_reachable_as_the_export_boundary() {
    let features = array![[1.0, f64::NAN, 10.0], [3.0, f64::NAN, 30.0]];
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();

    assert_eq!(model.number_of_input_columns(), 3);
    assert_eq!(model.number_of_output_features(), 2);
    assert_eq!(model.dropped_columns(), &[1]);
    assert_eq!(model.strategy(), SKImputationStrategy::Mean);
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    let statistics = model.statistics();
    assert_eq!(statistics.len(), 3);
    assert!((statistics[0] - 2.0).abs() < 1e-12);
    assert!(
        statistics[1].is_nan(),
        "an all-missing column has no statistic"
    );
    assert!((statistics[2] - 20.0).abs() < 1e-12);

    let indicator = model.indicator().expect("the indicator is fitted");
    assert_eq!(indicator.scope(), &[1]);
    assert_eq!(indicator.number_of_input_columns(), 3);
    assert_eq!(indicator.number_of_output_columns(), 1);
    assert_eq!(indicator.features(), SKMissingFeatureScope::MissingOnly);
    assert_eq!(indicator.backend(), SKBackendKind::MatrixMultiply);
}
