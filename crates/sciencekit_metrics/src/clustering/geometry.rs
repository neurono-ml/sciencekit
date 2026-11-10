//! Cluster centroids and dispersion shared by the centroid scores.

use ndarray::{Array2, ArrayView1};
use sciencekit_common::{SKClusterLabel, SKDataView, SKFloat};

use super::grouping::{SKClusterGrouping, sk_group_clusters};
use super::inputs::sk_view_dimensions;

/// Dense centroids plus the within-cluster dispersion of each cluster.
pub(super) struct SKClusterGeometry<F> {
    /// One dense centroid per compact cluster (`number_of_clusters × features`).
    pub centroids: Array2<F>,
    /// Mean Euclidean distance from each row to its centroid.
    pub mean_scatter: Vec<F>,
}

/// Compute the geometry without densifying the feature input.
///
/// Centroids are accumulated row by row (`O(number_of_clusters × features)`),
/// and each row's distance to its centroid is derived from its stored sparse
/// entries plus the implicit-zero contribution, so the input representation is
/// read directly.
pub(super) fn sk_cluster_geometry<F: SKFloat>(
    features: SKDataView<'_, F>,
    assignments: ArrayView1<'_, SKClusterLabel>,
) -> SKClusterGeometry<F> {
    let grouping = sk_group_clusters(assignments);
    let (_, number_of_features) = sk_view_dimensions(&features);
    let number_of_clusters = grouping.number_of_clusters();
    let mut centroids = Array2::<F>::zeros((number_of_clusters, number_of_features));

    match features {
        SKDataView::Dense(dense) => {
            for (row, cluster) in grouping.row_cluster.iter().enumerate() {
                if let Some(cluster) = cluster {
                    let mut centroid = centroids.row_mut(*cluster);
                    centroid.zip_mut_with(&dense.row(row), |sum, value| *sum = *sum + *value);
                }
            }
        }
        SKDataView::Sparse(sparse) => {
            for (row, cluster) in grouping.row_cluster.iter().enumerate() {
                if let Some(cluster) = cluster {
                    let outer = sparse.outer_view(row).expect("row within bounds");
                    for (column, value) in outer.iter() {
                        let current = centroids[[*cluster, column]];
                        centroids[[*cluster, column]] = current + *value;
                    }
                }
            }
        }
        _ => unreachable!("SKDataView only exposes dense and csr"),
    }

    for (cluster, mut centroid) in centroids.rows_mut().into_iter().enumerate() {
        let size = F::from(grouping.cluster_sizes[cluster]).expect("a cluster size fits");
        for value in centroid.iter_mut() {
            *value = *value / size;
        }
    }

    let (_squared_scatter, mut mean_scatter) = match features {
        SKDataView::Dense(dense) => sk_dense_scatter(&dense, &grouping, &centroids),
        SKDataView::Sparse(sparse) => sk_sparse_scatter(&sparse, &grouping, &centroids),
        _ => unreachable!("SKDataView only exposes dense and csr"),
    };
    for (cluster, scatter) in mean_scatter.iter_mut().enumerate() {
        let size = F::from(grouping.cluster_sizes[cluster]).expect("a cluster size fits");
        *scatter = *scatter / size;
    }

    SKClusterGeometry {
        centroids,
        mean_scatter,
    }
}

/// Squared centroid distance between two dense centroid rows.
pub(super) fn sk_centroid_squared_distance<F: SKFloat>(
    left: ArrayView1<'_, F>,
    right: ArrayView1<'_, F>,
) -> F {
    left.iter()
        .zip(right.iter())
        .fold(F::zero(), |total, (left, right)| {
            let deviation = *left - *right;
            total + deviation * deviation
        })
}

/// Per-cluster scatter over a dense matrix.
fn sk_dense_scatter<F: SKFloat>(
    dense: &ndarray::ArrayView2<'_, F>,
    grouping: &SKClusterGrouping,
    centroids: &Array2<F>,
) -> (Vec<F>, Vec<F>) {
    let number_of_clusters = grouping.number_of_clusters();
    let mut squared = vec![F::zero(); number_of_clusters];
    let mut distances = vec![F::zero(); number_of_clusters];
    for (row, cluster) in grouping.row_cluster.iter().enumerate() {
        let Some(cluster) = cluster else {
            continue;
        };
        let centroid = centroids.row(*cluster);
        let mut squared_distance = F::zero();
        for (value, center) in dense.row(row).iter().zip(centroid.iter()) {
            let deviation = *value - *center;
            squared_distance = squared_distance + deviation * deviation;
        }
        squared[*cluster] = squared[*cluster] + squared_distance;
        distances[*cluster] = distances[*cluster] + squared_distance.sqrt();
    }
    (squared, distances)
}

/// Per-cluster scatter over a CSR matrix without densifying it.
fn sk_sparse_scatter<F: SKFloat>(
    sparse: &sprs::CsMatView<'_, F>,
    grouping: &SKClusterGrouping,
    centroids: &Array2<F>,
) -> (Vec<F>, Vec<F>) {
    let number_of_clusters = grouping.number_of_clusters();
    let centroid_norms: Vec<F> = centroids
        .rows()
        .into_iter()
        .map(|centroid| {
            centroid
                .iter()
                .fold(F::zero(), |total, value| total + *value * *value)
        })
        .collect();
    let mut squared = vec![F::zero(); number_of_clusters];
    let mut distances = vec![F::zero(); number_of_clusters];
    for (row, cluster) in grouping.row_cluster.iter().enumerate() {
        let Some(cluster) = cluster else {
            continue;
        };
        let centroid = centroids.row(*cluster);
        let mut stored = F::zero();
        let mut stored_centroid_squared = F::zero();
        for (column, value) in sparse.outer_view(row).expect("row within bounds").iter() {
            let deviation = *value - centroid[column];
            stored = stored + deviation * deviation;
            stored_centroid_squared = stored_centroid_squared + centroid[column] * centroid[column];
        }
        let implicit = centroid_norms[*cluster] - stored_centroid_squared;
        let squared_distance = stored + implicit;
        let squared_distance = if squared_distance < F::zero() {
            F::zero()
        } else {
            squared_distance
        };
        squared[*cluster] = squared[*cluster] + squared_distance;
        distances[*cluster] = distances[*cluster] + squared_distance.sqrt();
    }
    (squared, distances)
}
