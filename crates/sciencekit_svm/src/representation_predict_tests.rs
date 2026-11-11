//! Companion tests for mixed dense/CSR prediction against models fitted in
//! either representation (spec `support-vector-machines`).

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::SKKernel;
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
fn representation_predict_supports_all_four_fit_and_query_combinations_for_svc() {
    let train_dense = array![
        [-2.5_f64, 0.0, 1.0],
        [-2.0, 0.5, 0.0],
        [-1.8, -0.4, 0.5],
        [2.5, 0.0, -1.0],
        [2.0, -0.5, 0.0],
        [1.8, 0.4, -0.5]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let query_dense = array![
        [-2.2_f64, 0.1, 0.8],
        [0.0, 0.2, 0.0],
        [2.2, -0.1, -0.8]
    ];
    let train_csr = to_csr(&train_dense);
    let query_csr = to_csr(&query_dense);

    for kernel in [
        SKKernel::Linear,
        SKKernel::Polynomial,
        SKKernel::RadialBasisFunction,
        SKKernel::Sigmoid,
    ] {
        let dense_fitted = SKSVC::<f64>::build()
            .kernel(kernel)
            .probability(true)
            .random_seed(3)
            .build()
            .unwrap()
            .fit(train_dense.view(), labels.view())
            .unwrap();
        let sparse_fitted = SKSVC::<f64>::build()
            .kernel(kernel)
            .probability(true)
            .random_seed(3)
            .build()
            .unwrap()
            .fit(train_csr.view(), labels.view())
            .unwrap();

        let dd_dec = dense_fitted.decision_function(query_dense.view()).unwrap();
        let ds_dec = dense_fitted.decision_function(query_csr.view()).unwrap();
        let sd_dec = sparse_fitted.decision_function(query_dense.view()).unwrap();
        let ss_dec = sparse_fitted.decision_function(query_csr.view()).unwrap();

        for candidate in [&ds_dec, &sd_dec, &ss_dec] {
            candidate
                .iter()
                .zip(dd_dec.iter())
                .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-9));
        }

        let dd_prob = dense_fitted.predict_probabilities(query_dense.view()).unwrap();
        let ds_prob = dense_fitted.predict_probabilities(query_csr.view()).unwrap();
        let sd_prob = sparse_fitted.predict_probabilities(query_dense.view()).unwrap();
        let ss_prob = sparse_fitted.predict_probabilities(query_csr.view()).unwrap();

        for candidate in [&ds_prob, &sd_prob, &ss_prob] {
            candidate
                .iter()
                .zip(dd_prob.iter())
                .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-8));
        }

        let ref_labels = dense_fitted.predict_labels(query_dense.view()).unwrap();
        assert_eq!(dense_fitted.predict_labels(query_csr.view()).unwrap(), ref_labels);
        assert_eq!(sparse_fitted.predict_labels(query_dense.view()).unwrap(), ref_labels);
        assert_eq!(sparse_fitted.predict_labels(query_csr.view()).unwrap(), ref_labels);
    }
}

#[test]
fn representation_predict_supports_all_four_fit_and_query_combinations_for_svr() {
    let train_dense = array![
        [-2.0_f64, 0.0, 1.0],
        [-1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, -0.5, 0.0],
        [2.0, 0.0, -1.0]
    ];
    let targets = array![-3.0_f64, -1.2, 0.2, 1.3, 3.1];
    let query_dense = array![[-1.5_f64, 0.2, 0.5], [0.5, -0.2, 0.0], [1.8, 0.0, -0.8]];
    let train_csr = to_csr(&train_dense);
    let query_csr = to_csr(&query_dense);

    for kernel in [
        SKKernel::Linear,
        SKKernel::Polynomial,
        SKKernel::RadialBasisFunction,
        SKKernel::Sigmoid,
    ] {
        let dense_fitted = SKSVR::<f64>::build()
            .kernel(kernel)
            .C(4.0)
            .epsilon(0.05)
            .tolerance(1e-8)
            .build()
            .unwrap()
            .fit(train_dense.view(), targets.view())
            .unwrap();
        let sparse_fitted = SKSVR::<f64>::build()
            .kernel(kernel)
            .C(4.0)
            .epsilon(0.05)
            .tolerance(1e-8)
            .build()
            .unwrap()
            .fit(train_csr.view(), targets.view())
            .unwrap();

        let dd = dense_fitted.predict(query_dense.view()).unwrap();
        let ds = dense_fitted.predict(query_csr.view()).unwrap();
        let sd = sparse_fitted.predict(query_dense.view()).unwrap();
        let ss = sparse_fitted.predict(query_csr.view()).unwrap();

        ds.iter()
            .zip(dd.iter())
            .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-12));
        ss.iter()
            .zip(sd.iter())
            .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-12));
        for candidate in [&sd, &ss] {
            candidate
                .iter()
                .zip(dd.iter())
                .for_each(|(&actual, &reference)| {
                    assert!(
                        (actual - reference).abs() < 1e-6,
                        "kernel {kernel:?}: actual={actual}, reference={reference}"
                    )
                });
        }
    }
}
