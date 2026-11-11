//! Tiny-data and full-kernel-catalogue acceptance tests
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::{SKKernel, SKKernelGamma, sk_linear_kernel_matrix};
use sciencekit_metrics::classification::sk_accuracy_score;

use crate::{SKSVC, SKSVR};

#[test]
fn acceptance_tiny_data_and_precomputed_kernel_work_for_svc_and_svr() {
    let tiny_x = array![[-1.0_f64, 0.0], [1.0, 0.0]];
    let tiny_labels = array![0_i64, 1];
    let tiny_svc = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .build()
        .unwrap()
        .fit(tiny_x.view(), tiny_labels.view())
        .unwrap();
    assert_eq!(tiny_svc.predict_labels(tiny_x.view()).unwrap(), tiny_labels);

    let gram = sk_linear_kernel_matrix(
        SKDataView::Dense(tiny_x.view()),
        SKDataView::Dense(tiny_x.view()),
    )
    .unwrap();
    let precomputed_svc = SKSVC::<f64>::build()
        .kernel(SKKernel::Precomputed)
        .build()
        .unwrap()
        .fit(gram.view(), tiny_labels.view())
        .unwrap();
    assert_eq!(
        precomputed_svc.predict_labels(gram.view()).unwrap(),
        tiny_labels
    );

    let reg_x = array![[0.0_f64], [1.0], [2.0]];
    let reg_y = array![1.0_f64, 3.0, 5.0];
    let reg_gram = sk_linear_kernel_matrix(
        SKDataView::Dense(reg_x.view()),
        SKDataView::Dense(reg_x.view()),
    )
    .unwrap();
    let precomputed_svr = SKSVR::<f64>::build()
        .kernel(SKKernel::Precomputed)
        .C(20.0)
        .epsilon(0.01)
        .build()
        .unwrap()
        .fit(reg_gram.view(), reg_y.view())
        .unwrap();
    let preds = precomputed_svr.predict(reg_gram.view()).unwrap();
    preds
        .iter()
        .zip(reg_y.iter())
        .for_each(|(&pred, &target)| assert!((pred - target).abs() < 0.05));
}

#[test]
fn acceptance_nonlinear_xor_fixture_separates_with_rbf_better_than_linear() {
    let xor_x = array![
        [-1.0_f64, -1.0],
        [-1.1, -0.9],
        [1.0, 1.0],
        [0.9, 1.1],
        [-1.0, 1.0],
        [-0.9, 1.1],
        [1.0, -1.0],
        [1.1, -0.9]
    ];
    let xor_y = array![0_i64, 0, 0, 0, 1, 1, 1, 1];
    let truth = xor_y.to_vec();

    let linear_model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(2.0)
        .build()
        .unwrap()
        .fit(xor_x.view(), xor_y.view())
        .unwrap();
    let rbf_model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(1.0))
        .C(5.0)
        .build()
        .unwrap()
        .fit(xor_x.view(), xor_y.view())
        .unwrap();
    let poly_model = SKSVC::<f64>::build()
        .kernel(SKKernel::Polynomial)
        .degree(2)
        .gamma(SKKernelGamma::Value(1.0))
        .coef0(1.0)
        .C(5.0)
        .build()
        .unwrap()
        .fit(xor_x.view(), xor_y.view())
        .unwrap();

    let linear_acc: f64 =
        sk_accuracy_score(&truth, &linear_model.predict_labels(xor_x.view()).unwrap().to_vec())
            .unwrap();
    let rbf_acc: f64 =
        sk_accuracy_score(&truth, &rbf_model.predict_labels(xor_x.view()).unwrap().to_vec())
            .unwrap();
    let poly_acc: f64 =
        sk_accuracy_score(&truth, &poly_model.predict_labels(xor_x.view()).unwrap().to_vec())
            .unwrap();

    assert!(rbf_acc >= linear_acc && (rbf_acc - 1.0).abs() < 1e-12);
    assert!((poly_acc - 1.0).abs() < 1e-12);
}
