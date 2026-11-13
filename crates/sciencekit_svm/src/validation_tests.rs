//! Companion validation and error-taxonomy tests for `SKSVC` and `SKSVR`
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKError, SKExecutionMode,
    SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::{SKKernel, SKKernelGamma};
use sprs::TriMat;

use crate::{SKSVC, SKSVR};

#[test]
fn validation_rejects_empty_input_and_single_class_targets() {
    let empty_features = Array2::<f64>::zeros((0, 3));
    let empty_labels = Array1::<i64>::zeros(0);
    let empty_targets = Array1::<f64>::zeros(0);

    assert!(matches!(
        SKSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(empty_features.view(), empty_labels.view())
            .unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));
    assert!(matches!(
        SKSVR::<f64>::build()
            .build()
            .unwrap()
            .fit(empty_features.view(), empty_targets.view())
            .unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));

    let single_class_x = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let single_class_y = array![7_i64, 7, 7];
    assert!(matches!(
        SKSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(single_class_x.view(), single_class_y.view())
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "classes",
            ..
        }
    ));

    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .build()
        .unwrap()
        .fit(
            array![[-1.0_f64, 0.0], [1.0, 0.0]].view(),
            array![0_i64, 1].view(),
        )
        .unwrap();
    let empty_query = Array2::<f64>::zeros((0, 2));
    assert!(matches!(
        model.predict_labels(empty_query.view()).unwrap_err(),
        SKError::EmptyInput {
            operation: "predict"
        }
    ));
}

#[test]
fn validation_rejects_shape_mismatch_and_non_finite_values() {
    let features = array![[-1.0_f64, 0.0], [1.0, 0.0], [2.0, 1.0]];
    let labels = array![0_i64, 1, 1];
    let targets = array![-1.0_f64, 1.0, 2.0];

    let svc = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let svr = SKSVR::<f64>::build()
        .kernel(SKKernel::Linear)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    let wrong_columns = array![[1.0_f64, 2.0, 3.0]];
    assert!(matches!(
        svc.predict_labels(wrong_columns.view()).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));
    assert!(matches!(
        svr.predict(wrong_columns.view()).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));

    let nan_dense = array![[-1.0_f64, 0.0], [1.0, f64::NAN]];
    assert!(matches!(
        SKSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(nan_dense.view(), array![0_i64, 1].view())
            .unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));
    assert!(matches!(
        svc.predict_labels(nan_dense.view()).unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));

    let mut triplets = TriMat::new((2, 4));
    triplets.add_triplet(0, 1, 1.0_f64);
    triplets.add_triplet(1, 2, f64::INFINITY);
    let inf_csr = triplets.to_csr();
    assert!(matches!(
        SKSVR::<f64>::build()
            .build()
            .unwrap()
            .fit(inf_csr.view(), array![0.0_f64, 1.0].view())
            .unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));

    assert!(matches!(
        SKSVR::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), array![1.0_f64, f64::NAN, 2.0].view())
            .unwrap_err(),
        SKError::NonFiniteValue { position: 1 }
    ));
}

#[test]
fn validation_reports_not_converged_invalid_kernels_and_transfers_backend() {
    let features = array![
        [-3.0_f64, 0.2],
        [-2.0, -0.3],
        [-0.5, 0.4],
        [0.4, -0.5],
        [2.0, 0.3],
        [3.0, -0.2]
    ];
    let labels = array![0_i64, 0, 1, 0, 1, 1];
    let err = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(10.0)
        .tolerance(1e-12)
        .maximum_number_of_iterations(1)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(err, SKError::NotConverged { iterations: 1 }));

    // Constant features make Scale gamma undefined at fit time.
    let constant = array![[1.0_f64, 1.0], [1.0, 1.0]];
    assert!(matches!(
        SKSVC::<f64>::build()
            .gamma(SKKernelGamma::Scale)
            .build()
            .unwrap()
            .fit(constant.view(), array![0_i64, 1].view())
            .unwrap_err(),
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));

    // Backend and execution mode transfer to both fitted models.
    let fitted_svc = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(fitted_svc.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        fitted_svc.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
    let _ = fitted_svc.predict_labels(features.view()).unwrap();
}
