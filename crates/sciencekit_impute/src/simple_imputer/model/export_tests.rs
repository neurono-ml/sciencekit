//! Companion tests: imputers and indicators round-trip through the contract.

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKFeatureTransformer, SKUnsupervisedFit, sk_assert_export_round_trip,
};

use crate::missing_indicator::SKMissingFeatureScope;
use crate::missing_indicator::SKMissingIndicator;
use crate::simple_imputer::SKImputationStrategy;
use crate::simple_imputer::SKSimpleImputer;
use crate::simple_imputer::output::SKImputedOutput;

fn dense(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => matrix.to_dense(),
    }
}

#[test]
fn simple_imputer_round_trips_strategy_and_statistics() {
    let training = array![[1.0, f64::NAN], [3.0, 8.0], [5.0, 10.0]];
    let model = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&training)
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.strategy(), SKImputationStrategy::Median);
    let probe = array![[f64::NAN, 8.0], [3.0, f64::NAN]];
    assert_eq!(
        dense(model.transform(&probe).unwrap()),
        dense(imported.transform(&probe).unwrap())
    );
}

#[test]
fn simple_imputer_round_trips_the_composed_indicator() {
    let training = array![[1.0, f64::NAN], [3.0, 8.0], [5.0, 10.0]];
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .keep_empty_features(true)
        .build()
        .unwrap()
        .fit(&training)
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    let probe = array![[f64::NAN, 8.0], [3.0, f64::NAN], [7.0, 12.0]];
    assert_eq!(
        dense(model.transform(&probe).unwrap()),
        dense(imported.transform(&probe).unwrap())
    );
}

#[test]
fn missing_indicator_round_trips_scope_and_scope_policy() {
    let training = array![[1.0, f64::NAN, 3.0], [3.0, 4.0, 5.0]];
    let model = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .sparse_output(false)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&training)
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.scope(), model.scope());
    assert_eq!(imported.features(), SKMissingFeatureScope::All);
    let probe = array![[f64::NAN, 1.0, 3.0], [2.0, f64::NAN, 5.0]];
    assert_eq!(
        model.transform(&probe).unwrap(),
        imported.transform(&probe).unwrap()
    );
}
