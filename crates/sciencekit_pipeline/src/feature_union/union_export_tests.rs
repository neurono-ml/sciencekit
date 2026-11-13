//! Companion tests: a fitted feature union nests every member state.

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKExportable, SKFeatureBlock, SKImportable};
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::feature_union::{
    SK_FEATURE_UNION_ESTIMATOR_NAME, SKFeatureUnion, sk_feature_union_member_count,
};

#[test]
fn feature_union_export_nests_each_member_and_reports_the_backend() {
    let input: SKFeatureBlock<f64> = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]].into();
    let fitted = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&input)
        .unwrap();
    let state = fitted.export_state().unwrap();
    assert_eq!(
        state.metadata().estimator_name(),
        SK_FEATURE_UNION_ESTIMATOR_NAME
    );
    assert_eq!(state.metadata().feature_count(), 2);
    assert_eq!(sk_feature_union_member_count(&state).unwrap(), 2);
    assert_eq!(
        state
            .metadata()
            .parameter("backend")
            .and_then(sciencekit_common::SKParameterValue::as_string),
        Some("matrixmultiply")
    );
    assert!(
        state
            .tensor("feature_union.member.0.tensor.standard_scaler.mean")
            .is_some()
    );
    assert!(
        state
            .tensor("feature_union.member.1.tensor.minmax_scaler.minimum")
            .is_some()
    );
    assert_eq!(
        state.tensor("feature_union.member_widths").unwrap().shape(),
        &[2]
    );
}

#[test]
fn feature_union_import_returns_a_typed_unsupported_operation() {
    let input: SKFeatureBlock<f64> = array![[1.0_f64, 2.0], [3.0, 4.0]].into();
    let fitted = SKFeatureUnion::build()
        .stages((SKStandardScaler::build().build().unwrap(),))
        .build()
        .unwrap()
        .fit(&input)
        .unwrap();
    assert!(matches!(
        crate::feature_union::SKFeatureUnionModel::<f64, (SKStandardScaler,)>::import_state(
            &fitted.export_state().unwrap()
        ),
        Err(sciencekit_common::SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));
}
