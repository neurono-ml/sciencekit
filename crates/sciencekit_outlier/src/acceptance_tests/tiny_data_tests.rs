//! Tiny-data acceptance tests (1 row, 2 rows, identical rows, small f32/f64 dense and CSR).

use ndarray::{Array2, array};
use sciencekit_common::{SKOutlierLabel, SKOutlierPredictor};
use sprs::TriMat;

use crate::{SKContamination, SKIsolationForest};

#[test]
fn acceptance_tiny_single_row_two_rows_and_constant_rows_behave_cleanly() {
    let single = array![[4.2_f64, -1.7, 0.0]];
    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(16)
        .random_seed(11)
        .build()
        .unwrap();
    let single_model = detector.fit(&single).unwrap();
    assert_eq!(single_model.subsample_size(), 1);
    let single_scores = single_model.score_samples(&single).unwrap();
    let single_labels = single_model.predict(&single).unwrap();
    assert_eq!(single_scores.len(), 1);
    assert!((single_scores[0] - (-0.5)).abs() < 1e-15);
    assert_eq!(single_labels[0], SKOutlierLabel::Inlier);

    let pair = array![[0.0_f64, 0.0], [10.0, -10.0]];
    let pair_fraction = SKIsolationForest::<f64>::build()
        .number_of_trees(32)
        .contamination(SKContamination::Fraction(0.5))
        .random_seed(11)
        .build()
        .unwrap();
    let pair_labels = pair_fraction.fit_predict(&pair).unwrap();
    assert_eq!(pair_labels.len(), 2);

    // Identical rows cannot be split and must all score identically as inliers under Auto.
    let constant = Array2::from_elem((6, 3), 3.0_f32);
    let constant_model = SKIsolationForest::<f32>::build()
        .number_of_trees(20)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(&constant)
        .unwrap();
    let constant_anomalies = constant_model.anomaly_scores(&constant).unwrap();
    let constant_labels = constant_model.predict(&constant).unwrap();
    assert!(
        constant_anomalies
            .iter()
            .all(|&score| (score - 0.5).abs() < 1e-6)
    );
    assert!(constant_labels.iter().all(|label| label.is_inlier()));
}

#[test]
fn acceptance_tiny_dense_sparse_and_interop_round_trip() {
    let dense = array![
        [0.0_f64, 0.1, 0.0],
        [0.1, 0.0, -0.1],
        [-0.1, 0.1, 0.0],
        [0.0, -0.1, 0.1],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, -0.05],
        [-0.05, 0.0, 0.05],
        [25.0, -30.0, 18.0]
    ];
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, values)| {
            values
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .for_each(|(column_index, &value)| {
                    triplets.add_triplet(row_index, column_index, value);
                });
        });
    let csr = triplets.to_csr();

    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(64)
        .contamination(SKContamination::Fraction(0.125))
        .random_seed(2026)
        .build()
        .unwrap();

    let dense_model = detector.fit(&dense).unwrap();
    let sparse_model = detector.fit(csr.view()).unwrap();
    let labels_dense = SKOutlierPredictor::predict(&dense_model, &dense).unwrap();
    let labels_sparse = SKOutlierPredictor::predict(&sparse_model, csr.view()).unwrap();
    assert_eq!(labels_dense, labels_sparse);
    assert_eq!(labels_dense[7], SKOutlierLabel::Outlier);

    let external_codes: Vec<i64> = labels_dense.iter().map(|label| label.to_i64()).collect();
    assert_eq!(external_codes[7], -1);
    assert!(external_codes.iter().take(7).all(|&code| code == 1));
    let restored: Vec<SKOutlierLabel> = external_codes
        .into_iter()
        .map(|code| SKOutlierLabel::from_i64(code).unwrap())
        .collect();
    assert_eq!(restored.as_slice(), labels_dense.as_slice().unwrap());
}
