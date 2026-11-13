//! Typed cluster assignments grouped into compact cluster indices.

use std::collections::BTreeMap;

use ndarray::ArrayView1;
use sciencekit_common::SKClusterLabel;

/// Rows mapped to compact cluster indices, plus the per-cluster sizes.
///
/// Noise rows map to `None`; every cluster index lies in
/// `0..number_of_clusters` and noise is never counted as a cluster.
pub(super) struct SKClusterGrouping {
    /// One entry per feature row.
    pub row_cluster: Vec<Option<usize>>,
    /// Number of rows in each compact cluster.
    pub cluster_sizes: Vec<usize>,
}

impl SKClusterGrouping {
    /// Number of clusters that carry at least one row.
    pub fn number_of_clusters(&self) -> usize {
        self.cluster_sizes.len()
    }
}

/// Group the typed assignments into compact cluster indices.
///
/// Labels are canonicalized through a `BTreeMap`, so the compact order follows
/// the cluster identifier and every metric sees a stable cluster ordering.
pub(super) fn sk_group_clusters(assignments: ArrayView1<'_, SKClusterLabel>) -> SKClusterGrouping {
    let mut indices = BTreeMap::new();
    let mut row_cluster = Vec::with_capacity(assignments.len());
    let mut cluster_sizes = Vec::new();
    for assignment in assignments.iter() {
        match assignment.cluster_index() {
            Some(index) => {
                let next = indices.len();
                let compact = *indices.entry(index).or_insert(next);
                if compact == cluster_sizes.len() {
                    cluster_sizes.push(0);
                }
                cluster_sizes[compact] += 1;
                row_cluster.push(Some(compact));
            }
            None => row_cluster.push(None),
        }
    }
    SKClusterGrouping {
        row_cluster,
        cluster_sizes,
    }
}
