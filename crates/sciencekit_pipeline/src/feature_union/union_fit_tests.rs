//! Fit and transform tests over tuple unions (spec `feature-union`, Task 1.2).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataView, SKFeatureBlock};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::feature_union::SKFeatureUnion;

fn dense_matrix(block: &SKFeatureBlock<f64>) -> Array2<f64> {
    match block.view() {
        SKDataView::Dense(view) => view.to_owned(),
        _ => panic!("expected dense feature block"),
    }
}

#[test]
fn hand_checkable_two_member_concatenation_preserves_member_order() {
    let standard = SKStandardScaler::build().build().unwrap();
    let minmax = SKMinMaxScaler::<f64>::build().build().unwrap();
    let union = SKFeatureUnion::build()
        .stages((standard, minmax))
        .build()
        .unwrap();

    let training = SKFeatureBlock::from(array![[1.0_f64, 10.0], [3.0, 20.0]]);
    let fitted = union.fit(&training).unwrap();
    let output = fitted.transform(&training).unwrap();

    let expected = array![[-1.0_f64, -1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]];
    let actual = dense_matrix(&output);
    assert_eq!(actual.dim(), (2, 4));
    actual
        .iter()
        .zip(expected.iter())
        .for_each(|(found, target)| assert!((found - target).abs() < 1e-12));
}

#[test]
fn every_member_sees_the_same_input_rows_and_columns() {
    // In a linear pipeline, a second stage sees the *scaled* data (mean 0).
    // In a feature union, both members see the *raw* input block independently.
    let standard = SKStandardScaler::build().build().unwrap();
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let union = SKFeatureUnion::build()
        .stages((standard, imputer))
        .build()
        .unwrap();

    let raw = SKFeatureBlock::from(array![[2.0_f64, 10.0], [4.0, 20.0], [6.0, 30.0]]);
    let fitted = union.fit(&raw).unwrap();
    let (scaler_model, imputer_model) = fitted.fitted_stages();

    assert!((scaler_model.mean()[0] - 4.0).abs() < 1e-12);
    assert!((scaler_model.mean()[1] - 20.0).abs() < 1e-12);
    assert!((imputer_model.statistics()[0] - 4.0).abs() < 1e-12);
    assert!((imputer_model.statistics()[1] - 20.0).abs() < 1e-12);
}

#[test]
fn transform_is_deterministic_across_repeated_calls() {
    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();

    let training = SKFeatureBlock::from(array![[1.0_f64, 2.0], [3.0, 6.0], [5.0, 10.0]]);
    let fitted = union.fit(&training).unwrap();

    let query = SKFeatureBlock::from(array![[2.0_f64, 4.0], [4.0, 8.0]]);
    let first = dense_matrix(&fitted.transform(&query).unwrap());
    let second = dense_matrix(&fitted.transform(&query).unwrap());
    assert_eq!(first, second);
}
