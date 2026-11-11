//! Large-data acceptance tests (`>= 20 000` rows): dense/sparse parity, Auto and explicit
//! contamination, thread-count determinism, concurrent serving and `sciencekit_metrics` evaluation.

use std::sync::Arc;

use ndarray::Array2;
use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKOutlierLabel, SKOutlierPredictor};
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};
use sciencekit_metrics::ranking::{sk_average_precision_score, sk_roc_auc_score};
use sprs::CsMat;

use crate::{SKContamination, SKIsolationForest};

const TOTAL_ROWS: usize = 20_000;
const OUTLIER_ROWS: usize = 400;
const FEATURE_COUNT: usize = 6;

fn build_large_dataset() -> (Array2<f64>, CsMat<f64>, Vec<i64>) {
    let inlier_rows = TOTAL_ROWS - OUTLIER_ROWS;
    let dense = Array2::from_shape_fn((TOTAL_ROWS, FEATURE_COUNT), |(row, column)| {
        if row < inlier_rows {
            // Sparse-friendly clustered normal rows with ~50% exact zeros.
            if (row + column) % 2 == 0 {
                0.0
            } else {
                let centered = ((row * 19 + column * 37) % 101) as f64 / 50.0 - 1.0;
                0.35 * centered
            }
        } else {
            // Well-separated multi-coordinate anomalies.
            let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
            let magnitude = 12.0 + ((row * 7 + column * 13) % 15) as f64;
            sign * magnitude
        }
    });

    let mut row_pointers = Vec::with_capacity(TOTAL_ROWS + 1);
    let mut column_indices = Vec::new();
    let mut stored_values = Vec::new();
    row_pointers.push(0);
    dense.rows().into_iter().for_each(|row| {
        row.iter()
            .enumerate()
            .filter(|&(_, &value)| value != 0.0)
            .for_each(|(column, &value)| {
                column_indices.push(column);
                stored_values.push(value);
            });
        row_pointers.push(column_indices.len());
    });
    let csr = CsMat::new(
        (TOTAL_ROWS, FEATURE_COUNT),
        row_pointers,
        column_indices,
        stored_values,
    );

    let ground_truth: Vec<i64> = (0..TOTAL_ROWS)
        .map(|row| {
            if row < inlier_rows {
                SKOutlierLabel::Inlier.to_i64()
            } else {
                SKOutlierLabel::Outlier.to_i64()
            }
        })
        .collect();

    (dense, csr, ground_truth)
}

#[test]
fn acceptance_large_20k_dense_sparse_auto_and_fraction_produce_high_metrics() {
    let (dense, csr, ground_truth) = build_large_dataset();
    let true_contamination = OUTLIER_ROWS as f64 / TOTAL_ROWS as f64;

    let fraction_detector = SKIsolationForest::<f64>::build()
        .number_of_trees(100)
        .contamination(SKContamination::Fraction(true_contamination))
        .random_seed(2026)
        .build()
        .unwrap();
    let auto_detector = SKIsolationForest::<f64>::build()
        .number_of_trees(100)
        .contamination(SKContamination::Auto)
        .random_seed(2026)
        .build()
        .unwrap();

    let dense_model = fraction_detector.fit(&dense).unwrap();
    let sparse_model = fraction_detector.fit(csr.view()).unwrap();
    let auto_model = auto_detector.fit(&dense).unwrap();

    assert_eq!(dense_model, sparse_model);
    assert_eq!(auto_model.offset(), -0.5);

    let anomalies_dense = dense_model.anomaly_scores(&dense).unwrap();
    let anomalies_sparse = sparse_model.anomaly_scores(csr.view()).unwrap();
    assert_eq!(anomalies_dense, anomalies_sparse);

    let labels_fraction = dense_model.predict(&dense).unwrap();
    let labels_sparse = sparse_model.predict(csr.view()).unwrap();
    let labels_auto = auto_model.predict(&dense).unwrap();
    assert_eq!(labels_fraction, labels_sparse);

    let predicted_fraction_i64: Vec<i64> = labels_fraction.iter().map(|l| l.to_i64()).collect();
    let predicted_auto_i64: Vec<i64> = labels_auto.iter().map(|l| l.to_i64()).collect();

    let roc_auc = sk_roc_auc_score(anomalies_dense.as_slice().unwrap(), &ground_truth, -1).unwrap();
    let average_precision =
        sk_average_precision_score(anomalies_dense.as_slice().unwrap(), &ground_truth, -1).unwrap();
    let accuracy_fraction: f64 = sk_accuracy_score(&ground_truth, &predicted_fraction_i64).unwrap();
    let f1_outlier_fraction: f64 = sk_f1_score(
        &ground_truth,
        &predicted_fraction_i64,
        None,
        SKAveraging::Binary { positive_label: -1 },
    )
    .unwrap();
    let accuracy_auto: f64 = sk_accuracy_score(&ground_truth, &predicted_auto_i64).unwrap();

    assert!(roc_auc > 0.99, "roc_auc = {roc_auc}");
    assert!(
        average_precision > 0.95,
        "average_precision = {average_precision}"
    );
    assert!(
        accuracy_fraction > 0.99,
        "accuracy_fraction = {accuracy_fraction}"
    );
    assert!(
        f1_outlier_fraction > 0.90,
        "f1_outlier_fraction = {f1_outlier_fraction}"
    );
    assert!(accuracy_auto > 0.95, "accuracy_auto = {accuracy_auto}");
}

#[test]
fn acceptance_large_20k_thread_invariance_and_concurrent_prediction() {
    let (dense, csr, _) = build_large_dataset();
    let dense = Arc::new(dense);
    let csr = Arc::new(csr);

    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(32)
        .contamination(SKContamination::Fraction(0.02))
        .random_seed(909)
        .build()
        .unwrap();

    let run_in_pool = |threads: usize| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let model = detector.fit(dense.view()).unwrap();
                let scores = model.score_samples(dense.view()).unwrap();
                (model, scores)
            })
    };

    let (single_thread_model, single_thread_scores) = run_in_pool(1);
    let (four_thread_model, four_thread_scores) = run_in_pool(4);
    assert_eq!(single_thread_model, four_thread_model);
    assert_eq!(single_thread_scores, four_thread_scores);

    let shared_model = Arc::new(single_thread_model);
    let expected_labels = shared_model.predict(dense.view()).unwrap();

    let workers: Vec<_> = (0..4)
        .map(|worker| {
            let model = Arc::clone(&shared_model);
            let dense = Arc::clone(&dense);
            let csr = Arc::clone(&csr);
            std::thread::spawn(move || {
                if worker % 2 == 0 {
                    SKOutlierPredictor::predict(&*model, dense.view()).unwrap()
                } else {
                    SKOutlierPredictor::predict(&*model, csr.view()).unwrap()
                }
            })
        })
        .collect();

    workers
        .into_iter()
        .for_each(|worker| assert_eq!(worker.join().unwrap(), expected_labels));
}
