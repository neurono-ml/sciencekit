//! Large-data (>= 10,000 rows) dense and CSR acceptance tests
//! (PRD §8.7, spec `support-vector-machines`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_math::{SKKernel, SKKernelGamma};
use sprs::CsMat;

use crate::{SKSVC, SKSVR};

const LARGE_ROWS: usize = 10_500;

#[test]
fn acceptance_large_dense_and_csr_svc_fits_and_predicts_over_ten_thousand_rows() {
    let dense = Array2::from_shape_fn((LARGE_ROWS, 4), |(row, col)| {
        let cluster = match row % 3 {
            0 => -4.0,
            1 => 0.0,
            _ => 4.0,
        };
        if col == (row % 3) {
            cluster + ((row % 17) as f64) * 0.02
        } else {
            0.0
        }
    });
    let labels = Array1::from_shape_fn(LARGE_ROWS, |row| (row % 3) as i64);

    // Build CSR directly in O(rows) without densifying.
    let mut indptr = Vec::with_capacity(LARGE_ROWS + 1);
    let mut indices = Vec::with_capacity(LARGE_ROWS);
    let mut data = Vec::with_capacity(LARGE_ROWS);
    indptr.push(0);
    for row in 0..LARGE_ROWS {
        let col = row % 3;
        let value = dense[[row, col]];
        if value != 0.0 {
            indices.push(col);
            data.push(value);
        }
        indptr.push(indices.len());
    }
    let csr = CsMat::new((LARGE_ROWS, 4), indptr, indices, data);

    let dense_svc = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Auto)
        .C(2.0)
        .cache_size(4)
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();

    let predicted_dense = dense_svc.predict_labels(dense.view()).unwrap();
    assert_eq!(predicted_dense.len(), LARGE_ROWS);
    let matches = predicted_dense
        .iter()
        .zip(labels.iter())
        .filter(|(a, b)| a == b)
        .count();
    assert_eq!(matches, LARGE_ROWS);

    let sparse_svc = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(2.0)
        .cache_size(4)
        .build()
        .unwrap()
        .fit(csr.view(), labels.view())
        .unwrap();
    let predicted_sparse = sparse_svc.predict_labels(csr.view()).unwrap();
    assert_eq!(predicted_sparse, labels);
}

#[test]
fn acceptance_large_svr_fits_and_predicts_over_ten_thousand_rows() {
    let features = Array2::from_shape_fn((LARGE_ROWS, 2), |(row, col)| {
        let t = -2.0 + 4.0 * (row as f64) / ((LARGE_ROWS - 1) as f64);
        if col == 0 { t } else { 0.5 * t }
    });
    let targets = Array1::from_shape_fn(LARGE_ROWS, |row| {
        2.0 * features[[row, 0]] - features[[row, 1]] + 0.75
    });

    let model = SKSVR::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(10.0)
        .epsilon(0.05)
        .cache_size(4)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    let predictions = model.predict(features.view()).unwrap();
    assert_eq!(predictions.len(), LARGE_ROWS);
    predictions
        .iter()
        .zip(targets.iter())
        .for_each(|(&pred, &target)| assert!((pred - target).abs() <= 0.08));
}
