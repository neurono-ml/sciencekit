//! Companion tests: PCA and truncated SVD round-trip their projections.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExportable, SKImportable, SKModelState, SKTensor,
    SKUnsupervisedFit, sk_assert_export_round_trip,
};

use crate::pca::SKPCAModel;
use crate::truncated_svd::SKTruncatedSVDModel;

use crate::pca::SKPCA;
use crate::truncated_svd::SKTruncatedSVD;

#[test]
fn pca_round_trips_components_mean_and_whitening() {
    let training = array![
        [1.0_f64, 2.0, 0.5, -1.0],
        [2.5, 4.1, 1.2, -0.5],
        [4.0, 6.2, 2.1, 0.2],
        [5.5, 8.0, 2.9, 1.0],
    ];
    let model = SKPCA::<f64>::build()
        .number_of_components(2)
        .whiten(true)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.components(), model.components());
    assert_eq!(imported.mean(), model.mean());
    assert_eq!(
        model.transform(training.view()).unwrap(),
        imported.transform(training.view()).unwrap()
    );
}

#[test]
fn truncated_svd_round_trips_components_and_projection() {
    let training = array![
        [1.0_f64, 0.0, 2.0],
        [0.0, 3.0, 1.0],
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 3.0],
    ];
    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .random_seed(11)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.singular_values(), model.singular_values());
    assert_eq!(
        model.transform(training.view()).unwrap(),
        imported.transform(training.view()).unwrap()
    );
}

#[test]
fn pca_import_rejects_a_mean_with_the_wrong_feature_count() {
    let training = array![[1.0_f64, 2.0, 0.5], [2.5, 4.1, 1.2], [4.0, 6.2, 2.1]];
    let model = SKPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let state = model.export_state().unwrap();
    let mut tensors: Vec<_> = state
        .tensors()
        .values()
        .filter(|tensor| tensor.name() != "pca.mean")
        .cloned()
        .collect();
    tensors.push(SKTensor::floating("pca.mean", vec![1], vec![0.0]).unwrap());
    let broken = SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(matches!(
        SKPCAModel::<f64>::import_state(&broken),
        Err(sciencekit_common::SKError::ShapeMismatch { .. })
    ));
}

#[test]
fn truncated_svd_import_rejects_singular_values_with_the_wrong_length() {
    let training = array![[1.0_f64, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 0.0]];
    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let state = model.export_state().unwrap();
    let mut tensors: Vec<_> = state
        .tensors()
        .values()
        .filter(|tensor| tensor.name() != "truncated_svd.singular_values")
        .cloned()
        .collect();
    tensors.push(SKTensor::floating("truncated_svd.singular_values", vec![1], vec![1.0]).unwrap());
    let broken = SKModelState::new(state.metadata().clone(), tensors).unwrap();
    assert!(matches!(
        SKTruncatedSVDModel::<f64>::import_state(&broken),
        Err(sciencekit_common::SKError::ShapeMismatch { .. })
    ));
}
