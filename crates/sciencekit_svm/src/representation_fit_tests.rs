//! Companion tests for dense and CSR fit equivalence and non-densification
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::{SKKernel, SKKernelGamma};
use sprs::{CsMat, TriMat};

use crate::{SKSVC, SKSVR};

fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new(dense.dim());
    dense
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, col), &val)| triplets.add_triplet(row, col, val));
    triplets.to_csr()
}

#[test]
fn representation_fit_dense_and_csr_agree_for_svc_and_svr() {
    let svc_dense = array![
        [-3.0_f64, 0.0, 1.5, 0.0],
        [-2.5, 0.5, 0.0, 1.0],
        [-2.0, 0.0, 1.0, 0.5],
        [0.0, 3.0, 0.0, -1.5],
        [0.5, 2.5, -1.0, 0.0],
        [-0.5, 2.0, 0.0, -1.0],
        [3.0, 0.0, -1.5, 0.0],
        [2.5, -0.5, 0.0, 1.5],
        [2.0, 0.5, -1.0, 0.0]
    ];
    let svc_labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];
    let svc_csr = to_csr(&svc_dense);

    for kernel in [
        SKKernel::Linear,
        SKKernel::Polynomial,
        SKKernel::RadialBasisFunction,
        SKKernel::Sigmoid,
    ] {
        let dense_model = SKSVC::<f64>::build()
            .kernel(kernel)
            .gamma(SKKernelGamma::Scale)
            .C(1.5)
            .probability(true)
            .random_seed(5)
            .build()
            .unwrap()
            .fit(svc_dense.view(), svc_labels.view())
            .unwrap();

        let sparse_model = SKSVC::<f64>::build()
            .kernel(kernel)
            .gamma(SKKernelGamma::Scale)
            .C(1.5)
            .probability(true)
            .random_seed(5)
            .build()
            .unwrap()
            .fit(svc_csr.view(), svc_labels.view())
            .unwrap();

        assert!(
            sparse_model
                .pairwise_models()
                .iter()
                .all(|m| m.support_vectors().is_sparse())
        );
        assert_eq!(
            dense_model.predict_labels(svc_dense.view()).unwrap(),
            sparse_model.predict_labels(svc_csr.view()).unwrap()
        );

        let d_dense = dense_model.decision_function(svc_dense.view()).unwrap();
        let d_sparse = sparse_model.decision_function(svc_csr.view()).unwrap();
        d_dense
            .iter()
            .zip(d_sparse.iter())
            .for_each(|(&a, &b)| assert!((a - b).abs() < 1e-9));

        let p_dense = dense_model.predict_probabilities(svc_dense.view()).unwrap();
        let p_sparse = sparse_model.predict_probabilities(svc_csr.view()).unwrap();
        p_dense
            .iter()
            .zip(p_sparse.iter())
            .for_each(|(&a, &b)| assert!((a - b).abs() < 1e-8));
    }

    let svr_targets = Array1::from_vec(vec![-2.0_f64, -1.6, -1.3, 0.2, 0.5, 0.4, 2.1, 1.9, 1.7]);
    let svr_dense = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(5.0)
        .epsilon(0.05)
        .build()
        .unwrap()
        .fit(svc_dense.view(), svr_targets.view())
        .unwrap();
    let svr_sparse = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(5.0)
        .epsilon(0.05)
        .build()
        .unwrap()
        .fit(svc_csr.view(), svr_targets.view())
        .unwrap();

    assert!(svr_sparse.support_vectors().is_sparse());
    let pred_dense = svr_dense.predict(svc_dense.view()).unwrap();
    let pred_sparse = svr_sparse.predict(svc_csr.view()).unwrap();
    pred_dense
        .iter()
        .zip(pred_sparse.iter())
        .for_each(|(&a, &b)| assert!((a - b).abs() < 1e-9));
}

#[test]
fn representation_fit_ultra_wide_csr_succeeds_without_densifying() {
    // 6 rows by 50,000,000 columns: densifying would allocate 2.4 GB.
    let ultra_csr = CsMat::new(
        (6, 50_000_000),
        vec![0, 2, 4, 6, 8, 10, 12],
        vec![
            5, 49_999_990, 5, 49_999_990, 5, 49_999_990, 100, 25_000_000, 100, 25_000_000, 100,
            25_000_000,
        ],
        vec![
            -3.0_f64, -2.0, -2.5, -1.5, -2.0, -2.5, 3.0, 2.0, 2.5, 1.5, 2.0, 2.5,
        ],
    );
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let targets = array![-1.0_f64, -1.1, -0.9, 1.0, 1.1, 0.9];

    let svc = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Scale)
        .build()
        .unwrap()
        .fit(ultra_csr.view(), labels.view())
        .unwrap();
    assert_eq!(svc.predict_labels(ultra_csr.view()).unwrap(), labels);

    let svr = SKSVR::<f64>::build()
        .kernel(SKKernel::Linear)
        .build()
        .unwrap()
        .fit(ultra_csr.view(), targets.view())
        .unwrap();
    let preds = svr.predict(ultra_csr.view()).unwrap();
    assert!(preds[0] < 0.0 && preds[3] > 0.0);
}
