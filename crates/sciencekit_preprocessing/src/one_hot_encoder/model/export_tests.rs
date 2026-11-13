//! Companion tests: encoders and expanders round-trip through the contract.

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExportable, SKImportable, sk_assert_export_round_trip,
};

use crate::one_hot_encoder::SKOneHotEncoder;
use crate::polynomial_features::SKPolynomialFeatures;

fn nominal_rows() -> Vec<Vec<&'static str>> {
    vec![
        vec!["alpha", "north"],
        vec!["beta", "south"],
        vec!["alpha", "south"],
    ]
}

#[test]
fn one_hot_encoder_round_trips_indicator_output() {
    let owned = nominal_rows();
    let view: Vec<&[&str]> = owned.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(view.as_slice())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_output_features(), 4);
    assert_eq!(
        model.transform_dense::<f64, _>(view.as_slice()).unwrap(),
        imported.transform_dense::<f64, _>(view.as_slice()).unwrap()
    );
}

#[test]
fn one_hot_encoder_round_trips_integer_category_codes() {
    let codes: Array2<i64> = array![[3, 7], [4, 7], [3, 8]];
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(&codes)
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.number_of_output_features(), 4);
    assert_eq!(
        model.transform_dense::<f64, _>(&codes).unwrap(),
        imported.transform_dense::<f64, _>(&codes).unwrap()
    );
}

#[test]
fn one_hot_encoder_rejects_a_wrong_category_count() {
    let owned = nominal_rows();
    let view: Vec<&[&str]> = owned.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(view.as_slice())
        .unwrap();
    let state = SKExportable::<f64>::export_state(&model).unwrap();
    let mut tensors: Vec<_> = state.tensors().values().cloned().collect();
    tensors[0] =
        sciencekit_common::SKTensor::integer("one_hot.category_counts", vec![1], vec![2]).unwrap();
    let broken = sciencekit_common::SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(crate::one_hot_encoder::model::SKOneHotEncoderModel::import_state(&broken).is_err());
}

#[test]
fn polynomial_features_round_trips_expansion() {
    let features: Array2<f64> = array![[2.0, 3.0], [4.0, 5.0]];
    let model = SKPolynomialFeatures::build()
        .degree(2)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_output_features(), 6);
    assert_eq!(
        model.transform_dense(features.view()).unwrap(),
        imported.transform_dense(features.view()).unwrap()
    );
}
