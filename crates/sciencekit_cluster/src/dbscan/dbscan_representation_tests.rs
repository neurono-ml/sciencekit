//! Companion tests for DBSCAN representation parity: dense trees vs brute
//! force, sparse CSR vs dense, and zero-densification guarantees
//! (spec `dbscan`, decisions 1 and 5).

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKClusterLabel, SKDataView, SKExecutionMode};
use sciencekit_neighbors::SKTrainingData;
use sprs::{CsMat, TriMat};

use super::config::SKDBSCANConfiguration;
use super::expansion::{sk_extract_core_samples, sk_run_dbscan_partition};
use super::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

fn two_blob_plus_noise_dense() -> Array2<f64> {
    Array2::from_shape_fn((45, 3), |(row, column)| {
        if row < 20 {
            // Blob 0 near (1.0, 0.0, 2.0) with deliberate zeros in column 1.
            let base = if column == 1 {
                0.0
            } else {
                (column + 1) as f64
            };
            let jitter = if column == 1 {
                0.0
            } else {
                ((row * 5 + column * 3) % 7) as f64 * 0.03
            };
            base + jitter
        } else if row < 40 {
            // Blob 1 near (10.0, 10.0, 0.0) with deliberate zeros in column 2.
            if column == 2 {
                0.0
            } else {
                10.0 + ((row * 3 + column * 7) % 5) as f64 * 0.04
            }
        } else {
            // 5 isolated noise rows far from both blobs.
            50.0 + (row - 40) as f64 * 8.0 + column as f64
        }
    })
}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

fn build_configuration(
    eps: f64,
    minimum_samples: usize,
    metric: SKDBSCANMetric,
    algorithm: SKDBSCANAlgorithm,
) -> SKDBSCANConfiguration<f64> {
    SKDBSCANConfiguration {
        eps,
        minimum_samples,
        metric,
        algorithm,
        leaf_size: 8,
        backend: SKBackendKind::Faer,
        execution_mode: SKExecutionMode::Automatic,
    }
}

#[test]
fn dbscan_representation_dense_trees_and_brute_force_agree() {
    let dense = two_blob_plus_noise_dense();
    let training = SKTrainingData::from_view(SKDataView::Dense(dense.view()));

    for metric in [
        SKDBSCANMetric::Euclidean,
        SKDBSCANMetric::Manhattan,
        SKDBSCANMetric::Chebyshev,
    ] {
        let kd = sk_run_dbscan_partition(
            &build_configuration(0.5, 4, metric, SKDBSCANAlgorithm::KdTree),
            &training,
        )
        .unwrap();
        let ball = sk_run_dbscan_partition(
            &build_configuration(0.5, 4, metric, SKDBSCANAlgorithm::BallTree),
            &training,
        )
        .unwrap();
        let brute = sk_run_dbscan_partition(
            &build_configuration(0.5, 4, metric, SKDBSCANAlgorithm::BruteForce),
            &training,
        )
        .unwrap();

        assert_eq!(kd.strategy, SKDBSCANSearchStrategy::KdTree);
        assert_eq!(ball.strategy, SKDBSCANSearchStrategy::BallTree);
        assert_eq!(brute.strategy, SKDBSCANSearchStrategy::BruteForce);
        assert_eq!(kd.number_of_clusters, 2);
        assert_eq!(kd.labels, brute.labels);
        assert_eq!(ball.labels, brute.labels);
        assert_eq!(kd.core_sample_indices, brute.core_sample_indices);
        assert_eq!(ball.core_sample_indices, brute.core_sample_indices);
    }
}

#[test]
fn dbscan_representation_sparse_and_dense_fits_agree() {
    let dense = two_blob_plus_noise_dense();
    let sparse = csr_from_dense(&dense);
    let dense_training = SKTrainingData::from_view(SKDataView::Dense(dense.view()));
    let sparse_training = SKTrainingData::from_view(SKDataView::Sparse(sparse.view()));

    for (metric, eps) in [
        (SKDBSCANMetric::Euclidean, 0.5),
        (SKDBSCANMetric::Manhattan, 0.8),
        (SKDBSCANMetric::Chebyshev, 0.4),
        (SKDBSCANMetric::Cosine, 0.01),
    ] {
        let configuration = build_configuration(eps, 4, metric, SKDBSCANAlgorithm::Auto);
        let dense_result = sk_run_dbscan_partition(&configuration, &dense_training).unwrap();
        let sparse_result = sk_run_dbscan_partition(&configuration, &sparse_training).unwrap();

        assert_eq!(sparse_result.strategy, SKDBSCANSearchStrategy::BruteForce);
        if metric.is_tree_compatible() {
            assert_eq!(dense_result.strategy, SKDBSCANSearchStrategy::KdTree);
        } else {
            assert_eq!(dense_result.strategy, SKDBSCANSearchStrategy::BruteForce);
        }
        assert_eq!(
            dense_result.number_of_clusters,
            sparse_result.number_of_clusters
        );
        assert_eq!(
            dense_result.core_sample_indices,
            sparse_result.core_sample_indices
        );
        assert_eq!(dense_result.labels, sparse_result.labels);
    }
}

#[test]
fn dbscan_representation_wide_sparse_design_and_core_extraction_never_densify() {
    // 6 rows x 10_000_000 columns: densifying would allocate 480 MB for a 6-row matrix.
    // Built directly in CSR: rows 0..3 cluster near col 0 = 1.0, rows 3..5 are sparse
    // pair (< minimum_samples = 3 -> noise), row 5 is far at col 9_999_999 = 50.0 (noise).
    let width = 10_000_000;
    let indptr = vec![0, 1, 2, 3, 4, 5, 6];
    let indices = vec![0, 0, 0, 5_000_000, 5_000_000, width - 1];
    let values = vec![1.0_f64, 1.05, 1.10, 10.0, 10.05, 50.0];
    let csr = CsMat::new((6, width), indptr, indices, values);
    let non_zeros_before = csr.nnz();

    let training = SKTrainingData::from_view(SKDataView::Sparse(csr.view()));
    assert_eq!(training.representation(), "csr");

    let configuration =
        build_configuration(0.2, 3, SKDBSCANMetric::Euclidean, SKDBSCANAlgorithm::Auto);
    let result = sk_run_dbscan_partition(&configuration, &training).unwrap();
    assert_eq!(result.strategy, SKDBSCANSearchStrategy::BruteForce);
    assert_eq!(result.number_of_clusters, 1);
    assert_eq!(result.core_sample_indices.to_vec(), vec![0, 1, 2]);
    assert_eq!(
        result.labels.to_vec(),
        vec![
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::Noise,
            SKClusterLabel::Noise,
            SKClusterLabel::Noise,
        ]
    );

    let core_samples =
        sk_extract_core_samples(&training, result.core_sample_indices.as_slice().unwrap());
    assert_eq!(core_samples.representation(), "csr");
    assert_eq!(core_samples.rows(), 3);
    assert_eq!(core_samples.columns(), width);
    assert_eq!(csr.nnz(), non_zeros_before);
}
