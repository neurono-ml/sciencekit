//! Concurrent accumulation of a large classification dataset.

use rayon::prelude::*;

use crate::accumulators::SKClassificationAccumulator;

const OBSERVATIONS: usize = 100_000;
const PARTITIONS: usize = 8;

/// Parallel partitions merge to the sequential confusion matrix.
#[test]
fn parallel_classification_accumulation_matches_sequential() {
    let true_labels: Vec<i64> = (0..OBSERVATIONS).map(|index| (index % 5) as i64).collect();
    let predicted_labels: Vec<i64> = (0..OBSERVATIONS)
        .map(|index| ((index * 7) % 6) as i64)
        .collect();

    let mut sequential = SKClassificationAccumulator::new();
    for (true_label, predicted_label) in true_labels.iter().zip(predicted_labels.iter()) {
        sequential.observe(*true_label, *predicted_label);
    }

    let merged = (0..PARTITIONS)
        .into_par_iter()
        .map(|partition| {
            let mut accumulator = SKClassificationAccumulator::new();
            let start = partition * OBSERVATIONS / PARTITIONS;
            let end = (partition + 1) * OBSERVATIONS / PARTITIONS;
            for index in start..end {
                accumulator.observe(true_labels[index], predicted_labels[index]);
            }
            accumulator
        })
        .reduce(SKClassificationAccumulator::new, |mut left, right| {
            left.merge(&right);
            left
        });

    let sequential_matrix = sequential.confusion_matrix(None).expect("non-empty");
    let parallel_matrix = merged.confusion_matrix(None).expect("non-empty");
    assert_eq!(sequential_matrix.labels(), parallel_matrix.labels());
    assert_eq!(sequential_matrix.counts(), parallel_matrix.counts());
    assert_eq!(merged.number_of_observations(), OBSERVATIONS);
}
