//! Deterministic fixtures shared by the acceptance tests.

use ndarray::{Array1, Array2};
use sciencekit_common::SKClusterLabel;
use sprs::{CsMat, TriMat};

/// A deterministic planted design: each row belongs to `row % clusters` and
/// sits at a well-separated base with a tiny integer offset.
pub(crate) fn planted_design(rows: usize, clusters: usize) -> (Array2<f64>, Array1<i64>) {
    let mut features = Array2::zeros((rows, 3));
    let mut truth = Array1::zeros(rows);
    for row in 0..rows {
        let group = row % clusters;
        truth[row] = group as i64;
        for feature in 0..3 {
            let offset = ((row * 7 + feature * 13) % 17) as f64 * 0.05;
            features[[row, feature]] = group as f64 * 1_000.0 + offset;
        }
    }
    (features, truth)
}

/// CSR copy of a dense fixture.
pub(crate) fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

/// Majority mapping of cluster ids to truth groups, then `sk_accuracy_score`.
pub(crate) fn majority_accuracy(
    truth: &Array1<i64>,
    labels: &Array1<SKClusterLabel>,
    number_of_clusters: usize,
) -> f64 {
    let mut mapping = vec![0_i64; number_of_clusters];
    for cluster in 0..number_of_clusters {
        let mut counts = std::collections::BTreeMap::<i64, usize>::new();
        for (row, label) in labels.iter().enumerate() {
            if label.cluster_index() == Some(cluster as u32) {
                *counts.entry(truth[row]).or_default() += 1;
            }
        }
        mapping[cluster] = counts
            .into_iter()
            .max_by_key(|(group, count)| (*count, std::cmp::Reverse(*group)))
            .map(|(group, _)| group)
            .unwrap_or(0);
    }
    let predicted: Vec<i64> = labels
        .iter()
        .map(|label| mapping[label.cluster_index().expect("KMeans never emits noise") as usize])
        .collect();
    let true_labels: Vec<i64> = truth.iter().copied().collect();
    sciencekit_metrics::classification::sk_accuracy_score::<f64>(&true_labels, &predicted)
        .expect("aligned non-empty assignments")
}
