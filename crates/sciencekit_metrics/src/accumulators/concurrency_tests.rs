//! TDD tests for concurrent accumulation (spec `evaluation-metrics`).

use rayon::prelude::*;

use super::{SKClassificationAccumulator, SKRegressionAccumulator};

const OBSERVATIONS: usize = 10_000;

/// Both accumulator types are safe to share across threads.
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn accumulators_are_send_and_sync() {
    assert_send_sync::<SKClassificationAccumulator>();
    assert_send_sync::<SKRegressionAccumulator<f64>>();
}

/// Parallel classification partitions merge to the sequential result.
#[test]
fn classification_parallel_partitions_match_sequential() {
    let true_labels: Vec<i64> = (0..OBSERVATIONS).map(|index| (index % 4) as i64).collect();
    let predicted_labels: Vec<i64> = (0..OBSERVATIONS)
        .map(|index| ((index * 3) % 5) as i64)
        .collect();

    let mut sequential = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        sequential.observe(*true_label, *predicted_label);
    }

    let merged = (0..8)
        .into_par_iter()
        .map(|partition| {
            let mut accumulator = SKClassificationAccumulator::new();
            let start = partition * OBSERVATIONS / 8;
            let end = (partition + 1) * OBSERVATIONS / 8;
            for index in start..end {
                accumulator.observe(true_labels[index], predicted_labels[index]);
            }
            accumulator
        })
        .reduce(SKClassificationAccumulator::new, |mut left, right| {
            left.merge(&right);
            left
        });

    let sequential_matrix = sequential
        .confusion_matrix(None)
        .expect("non-empty accumulator");
    let parallel_matrix = merged
        .confusion_matrix(None)
        .expect("non-empty accumulator");
    assert_eq!(sequential_matrix.labels(), parallel_matrix.labels());
    assert_eq!(sequential_matrix.counts(), parallel_matrix.counts());
    assert_eq!(merged.number_of_observations(), OBSERVATIONS);
}

/// Parallel regression partitions merge to the sequential result.
#[test]
fn regression_parallel_partitions_match_sequential() {
    let targets: Vec<f64> = (0..OBSERVATIONS)
        .map(|index| (index % 17) as f64 * 0.5)
        .collect();
    let predictions: Vec<f64> = (0..OBSERVATIONS)
        .map(|index| (index % 13) as f64 * 0.25)
        .collect();

    let mut sequential = SKRegressionAccumulator::new();
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        sequential.observe(*target, *prediction).expect("finite");
    }

    let merged = (0..8)
        .into_par_iter()
        .map(|partition| {
            let mut accumulator = SKRegressionAccumulator::new();
            let start = partition * OBSERVATIONS / 8;
            let end = (partition + 1) * OBSERVATIONS / 8;
            for index in start..end {
                accumulator
                    .observe(targets[index], predictions[index])
                    .expect("finite");
            }
            accumulator
        })
        .reduce(SKRegressionAccumulator::new, |mut left, right| {
            left.merge(&right);
            left
        });

    assert!(
        (sequential.mean_squared_error().unwrap() - merged.mean_squared_error().unwrap()).abs()
            < 1e-12
    );
    assert!(
        (sequential.mean_absolute_error().unwrap() - merged.mean_absolute_error().unwrap()).abs()
            < 1e-12
    );
    assert!((sequential.r2_score().unwrap() - merged.r2_score().unwrap()).abs() < 1e-12);
    assert_eq!(merged.number_of_observations(), OBSERVATIONS);
}
