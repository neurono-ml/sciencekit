//! End-to-end acceptance tests for `SKIncrementalPCA` on tiny `f32`/`f64`
//! designs, large dense matrices with batch-size variation and wide sparse CSR
//! designs without densification (PRD §8.7 / spec `incremental-pca`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::{planted_rank_matrix, planted_wide_csr};
use crate::SKIncrementalPCA;

#[test]
fn acceptance_incremental_pca_tiny_f32_large_dense_and_wide_sparse() {
    // 1 x 3 boundary.
    let single_row = array![[4.0_f64, -2.0, 7.0]];
    let single_model = SKIncrementalPCA::<f64>::build()
        .build()
        .unwrap()
        .fit(&single_row)
        .unwrap();
    assert_eq!(single_model.number_of_components(), 1);
    assert_eq!(single_model.explained_variance()[0], 0.0);

    // f32 end-to-end with whitening and round-trip reconstruction.
    let small_f32 = array![
        [1.0_f32, 2.0, 0.5],
        [2.0, 4.2, 1.1],
        [3.0, 5.8, 1.7],
        [4.0, 8.1, 2.0],
        [5.0, 10.0, 2.6]
    ];
    let f32_model = SKIncrementalPCA::<f32>::build()
        .number_of_components(2)
        .batch_size(3)
        .whiten(true)
        .build()
        .unwrap()
        .fit(&small_f32)
        .unwrap();
    let f32_projected = f32_model.transform(&small_f32).unwrap();
    assert_eq!(f32_projected.dim(), (5, 2));
    assert!(f32_projected.iter().all(|value| value.is_finite()));

    // Large 3_000 x 24 dense design across multiple batch sizes.
    let large_dense = planted_rank_matrix(3_000, 24, 3, 1e-4);
    let baseline = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .batch_size(1_024)
        .build()
        .unwrap()
        .fit(&large_dense)
        .unwrap();
    let fine_batches = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .batch_size(128)
        .build()
        .unwrap()
        .fit(&large_dense)
        .unwrap();
    let ratio_sum: f64 = baseline.explained_variance_ratio().iter().sum();
    assert!(ratio_sum > 0.999);
    for (left, right) in baseline
        .components()
        .iter()
        .zip(fine_batches.components().iter())
    {
        assert!((left - right).abs() < 1e-8);
    }

    // Wide 1_200 x 160 sparse CSR design without densification.
    let wide_sparse = planted_wide_csr(1_200, 160);
    let sparse_model = SKIncrementalPCA::<f64>::build()
        .number_of_components(4)
        .batch_size(256)
        .build()
        .unwrap()
        .fit(wide_sparse.view())
        .unwrap();
    assert_eq!(sparse_model.components().dim(), (4, 160));
    let sparse_projected = sparse_model.transform(wide_sparse.view()).unwrap();
    assert_eq!(sparse_projected.dim(), (1_200, 4));
    assert!(sparse_model.explained_variance_ratio()[0] > 0.5);
}
