//! Companion TDD tests for dense/CSR parity and the non-densifying sparse path
//! (spec `linear-support-vector-classifier`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKLinearSVC;

fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new(dense.dim());
    dense
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, column), &value)| triplets.add_triplet(row, column, value));
    triplets.to_csr()
}

fn fixture() -> (Array2<f64>, ndarray::Array1<i64>) {
    let features = array![
        [-3.0_f64, 0.0, 1.5],
        [-2.5, 0.5, 0.0],
        [-2.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [0.5, 2.5, -1.0],
        [0.0, 2.0, 0.0],
        [3.0, 0.0, -1.5],
        [2.5, -0.5, 0.0],
        [2.0, 0.0, -1.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];
    (features, labels)
}

#[test]
fn dense_and_csr_fits_agree_without_densifying() {
    let (features, labels) = fixture();
    let csr = to_csr(&features);

    let dense_model = SKLinearSVC::<f64>::build()
        .C(1.5)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let sparse_model = SKLinearSVC::<f64>::build()
        .C(1.5)
        .build()
        .unwrap()
        .fit(csr.view(), labels.view())
        .unwrap();

    assert_eq!(dense_model.heads().len(), sparse_model.heads().len());
    dense_model
        .heads()
        .iter()
        .zip(sparse_model.heads().iter())
        .for_each(|(dense_head, sparse_head)| {
            dense_head
                .coefficients()
                .iter()
                .zip(sparse_head.coefficients().iter())
                .for_each(|(left, right)| assert!((left - right).abs() < 1e-9));
            assert!((dense_head.intercept() - sparse_head.intercept()).abs() < 1e-9);
        });
}

#[test]
fn sparse_prediction_matches_dense_prediction_and_reads_stored_entries() {
    let (features, labels) = fixture();
    let csr = to_csr(&features);

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let dense_decisions = model.decision_function(features.view()).unwrap();
    let sparse_decisions = model.decision_function(csr.view()).unwrap();
    dense_decisions
        .iter()
        .zip(sparse_decisions.iter())
        .for_each(|(left, right)| assert!((left - right).abs() < 1e-12));

    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        model.predict_labels(csr.view()).unwrap()
    );
    assert_eq!(model.predict_labels(csr.view()).unwrap(), labels);
}

#[test]
fn sparse_fit_scans_stored_entries_only_on_a_matrix_too_wide_to_densify() {
    // 40 rows x 4,000,000 columns: densifying would need 1.28 GB, while the
    // stored entries fit well below a megabyte.
    let rows = 40;
    let columns = 4_000_000;
    let mut triplets = TriMat::new((rows, columns));
    (0..rows).for_each(|row| {
        let signal = if row % 2 == 0 { -1.0_f64 } else { 1.0 };
        triplets.add_triplet(row, row * 1000, signal);
        triplets.add_triplet(row, columns - 1 - row, 0.5 * signal);
    });
    let csr = triplets.to_csr();
    let labels = ndarray::Array1::from_iter((0..rows).map(|row| (row % 2) as i64));

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(csr.view(), labels.view())
        .unwrap();

    assert_eq!(model.number_of_features(), columns);
    assert_eq!(model.predict_labels(csr.view()).unwrap(), labels);
}
