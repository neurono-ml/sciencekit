//! Companion tests for `score_samples`, `anomaly_scores`, `decision_function`, `predict` and `fit_predict`.

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKOutlierLabel, SKOutlierPredictor};
use sprs::{CsMat, TriMat};

use crate::{SK_ISOLATION_AUTO_OFFSET, SKContamination, SKIsolationForest};

fn clustered_with_outliers_f64() -> Array2<f64> {
    let mut values = Vec::with_capacity(40 * 2);
    (0..38).for_each(|index| {
        let angle = (index as f64) * 0.17;
        values.push(0.15 * angle.cos());
        values.push(0.15 * angle.sin());
    });
    // Two distant anomalies at rows 38 and 39.
    values.extend_from_slice(&[25.0, -25.0, -30.0, 28.0]);
    Array2::from_shape_vec((40, 2), values).unwrap()
}

fn dense_to_csr(matrix: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((matrix.nrows(), matrix.ncols()));
    matrix
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, row_values)| {
            row_values
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .for_each(|(column_index, &value)| {
                    triplets.add_triplet(row_index, column_index, value);
                });
        });
    triplets.to_csr()
}

#[test]
fn isolation_score_auto_offset_monotone_scores_and_label_agreement() {
    let data = clustered_with_outliers_f64();
    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(100)
        .random_seed(42)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();

    let model = detector.fit(&data).unwrap();
    assert!((SKOutlierPredictor::offset(&model) - SK_ISOLATION_AUTO_OFFSET).abs() < 1e-15);

    let anomalies = model.anomaly_scores(&data).unwrap();
    let scores = SKOutlierPredictor::score_samples(&model, &data).unwrap();
    let decisions = SKOutlierPredictor::decision_function(&model, &data).unwrap();
    let labels = SKOutlierPredictor::predict(&model, &data).unwrap();
    let fit_predicted = detector.fit_predict(&data).unwrap();

    assert_eq!(anomalies.len(), 40);
    assert_eq!(scores.len(), 40);
    assert_eq!(decisions.len(), 40);
    assert_eq!(labels.len(), 40);
    assert_eq!(labels, fit_predicted);

    anomalies
        .iter()
        .zip(scores.iter())
        .zip(decisions.iter())
        .zip(labels.iter())
        .for_each(|(((&anomaly, &score), &decision), &label)| {
            assert!(!anomaly.is_nan() && (0.0..=1.0).contains(&anomaly));
            assert!((score + anomaly).abs() < 1e-15);
            assert!((decision - (score - model.offset())).abs() < 1e-15);
            let expected = if decision < 0.0 {
                SKOutlierLabel::Outlier
            } else {
                SKOutlierLabel::Inlier
            };
            assert_eq!(label, expected);
        });

    let worst_inlier_anomaly = anomalies.iter().take(38).copied().fold(0.0_f64, f64::max);
    assert!(anomalies[38] > worst_inlier_anomaly);
    assert!(anomalies[39] > worst_inlier_anomaly);
    assert_eq!(labels[38], SKOutlierLabel::Outlier);
    assert_eq!(labels[39], SKOutlierLabel::Outlier);
    assert!(labels.iter().take(38).all(|label| label.is_inlier()));
}

#[test]
fn isolation_score_explicit_contamination_sets_exact_quantile_offset() {
    let data = clustered_with_outliers_f64();
    let contamination = 0.05_f64;
    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(100)
        .contamination(SKContamination::Fraction(contamination))
        .random_seed(42)
        .build()
        .unwrap();

    let model = detector.fit(&data).unwrap();
    let scores = model.score_samples(&data).unwrap();
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let position = contamination * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let expected_offset =
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64);

    assert!((model.offset() - expected_offset).abs() < 1e-15);
    let labels = model.predict(&data).unwrap();
    let outlier_count = labels.iter().filter(|label| label.is_outlier()).count();
    assert_eq!(outlier_count, 2);
    assert_eq!(labels[38], SKOutlierLabel::Outlier);
    assert_eq!(labels[39], SKOutlierLabel::Outlier);
}

#[test]
fn isolation_score_scalar_invariance_and_four_way_representation_parity() {
    let data_f64 = array![
        [0.0_f64, 0.25, 0.0],
        [0.125, 0.0, -0.125],
        [-0.125, 0.125, 0.0],
        [0.0, -0.25, 0.125],
        [0.25, 0.125, 0.0],
        [0.0, 0.0, -0.25],
        [16.0, -20.0, 12.0]
    ];
    let data_f32 = data_f64.mapv(|value| value as f32);
    let csr_f64 = dense_to_csr(&data_f64);

    let labels_f64 = SKIsolationForest::<f64>::build()
        .number_of_trees(64)
        .random_seed(99)
        .build()
        .unwrap()
        .fit_predict(&data_f64)
        .unwrap();
    let labels_f32 = SKIsolationForest::<f32>::build()
        .number_of_trees(64)
        .random_seed(99)
        .build()
        .unwrap()
        .fit_predict(&data_f32)
        .unwrap();
    assert_eq!(labels_f64, labels_f32);
    assert_eq!(labels_f64[6], SKOutlierLabel::Outlier);

    let detector = SKIsolationForest::<f64>::build()
        .number_of_trees(64)
        .contamination(SKContamination::Fraction(1.0 / 7.0))
        .random_seed(99)
        .build()
        .unwrap();
    let dense_model = detector.fit(&data_f64).unwrap();
    let sparse_model = detector.fit(csr_f64.view()).unwrap();

    let scores_dd = dense_model.score_samples(&data_f64).unwrap();
    let scores_ds = dense_model.score_samples(csr_f64.view()).unwrap();
    let scores_sd = sparse_model.score_samples(&data_f64).unwrap();
    let scores_ss = sparse_model.score_samples(csr_f64.view()).unwrap();

    scores_dd
        .iter()
        .zip(scores_ds.iter())
        .zip(scores_sd.iter())
        .zip(scores_ss.iter())
        .for_each(|(((&dd, &ds), &sd), &ss)| {
            assert!((dd - ds).abs() < 1e-15);
            assert!((dd - sd).abs() < 1e-15);
            assert!((dd - ss).abs() < 1e-15);
        });

    assert_eq!(
        dense_model.predict(&data_f64).unwrap(),
        sparse_model.predict(csr_f64.view()).unwrap()
    );
    assert_eq!(
        detector.fit_predict(&data_f64).unwrap(),
        detector.fit_predict(csr_f64.view()).unwrap()
    );
}
