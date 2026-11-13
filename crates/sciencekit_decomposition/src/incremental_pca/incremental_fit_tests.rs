//! Companion tests for `SKIncrementalPCA::fit`, chunking, `partial_fit` and
//! `SKIncrementalPCAModel` accessors.

use ndarray::{Array2, Axis};
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKIncrementalPCA;
use crate::pca::test_tracing::PcaCaptureGuard;
use crate::{SKPCA, SKPCASolver};

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn incremental_fit_chunks_dense_and_sparse_and_matches_pca() {
    let dense = Array2::from_shape_fn((50, 6), |(row, column)| {
        if (row * 5 + column * 3) % 4 == 0 {
            let angle = (row as f64 + 1.0) * 0.23 + (column as f64) * 0.61;
            angle.sin() * (8.0 - column as f64) + ((row * 7 + column * 11) % 9) as f64 * 0.4
        } else {
            0.0
        }
    });
    let sparse = csr_from_dense(&dense);

    // 50 rows with batch_size 12 and 3 components exercises trailing-chunk coalescing
    // (chunks 0..12, 12..24, 24..36, 36..50).
    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .batch_size(12)
        .whiten(true)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    let dense_model = estimator.fit(&dense).unwrap();
    let sparse_model = estimator.fit(sparse.view()).unwrap();
    let full_pca = SKPCA::<f64>::build()
        .number_of_components(3)
        .svd_solver(SKPCASolver::Full)
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();

    assert_eq!(dense_model.samples_seen(), 50);
    assert_eq!(dense_model.epochs_consumed(), 1);
    assert_eq!(dense_model.number_of_components(), 3);
    assert_eq!(dense_model.number_of_features(), 6);
    assert_eq!(dense_model.batch_size(), 12);
    assert!(dense_model.whiten());
    assert_eq!(dense_model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        dense_model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let default_model = SKIncrementalPCA::<f64>::build()
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();
    assert_eq!(default_model.number_of_components(), 6);

    for (left, right) in dense_model
        .components()
        .iter()
        .zip(full_pca.components().iter())
    {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
    for (left, right) in sparse_model
        .components()
        .iter()
        .zip(dense_model.components().iter())
    {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
    for (left, right) in sparse_model
        .explained_variance_ratio()
        .iter()
        .zip(full_pca.explained_variance_ratio().iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
    for (left, right) in dense_model
        .singular_values()
        .iter()
        .zip(full_pca.singular_values().iter())
    {
        assert!((left - right).abs() < 1e-9);
    }
}

#[test]
fn incremental_fit_partial_fit_chains_and_records_backend_spans() {
    let data = Array2::from_shape_fn((36, 4), |(row, column)| {
        let angle = (row as f64 + 1.0) * 0.19;
        angle.sin() * (column as f64 + 2.0) + ((row * 3 + column * 5) % 7) as f64 * 0.35
    });
    let batch_one = data.slice_axis(Axis(0), ndarray::Slice::from(0..12));
    let batch_two = data
        .slice_axis(Axis(0), ndarray::Slice::from(12..24))
        .to_owned();
    let batch_three = data.slice_axis(Axis(0), ndarray::Slice::from(24..36));
    let batch_two_csr = csr_from_dense(&batch_two);

    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();

    let single_pass = estimator.fit(&data).unwrap();

    let capture = PcaCaptureGuard::begin();
    let mut chained = estimator.fit(batch_one).unwrap();
    chained
        .partial_fit(batch_two_csr.view())
        .unwrap()
        .partial_fit(batch_three)
        .unwrap();
    let spans = capture.spans();
    assert!(
        spans.iter().any(|(name, fields)| {
            name == "sciencekit.operation"
                && fields.get("operation").map(String::as_str) == Some("fit")
                && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
        }),
        "expected fit span with matrixmultiply backend, got: {spans:?}"
    );
    assert!(
        spans.iter().any(|(name, fields)| {
            name == "sciencekit.operation"
                && fields.get("operation").map(String::as_str) == Some("partial_fit")
                && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
        }),
        "expected partial_fit span with matrixmultiply backend, got: {spans:?}"
    );

    assert_eq!(chained.samples_seen(), 36);
    for (left, right) in chained.mean().iter().zip(single_pass.mean().iter()) {
        assert!((left - right).abs() < 1e-11);
    }
    for (left, right) in chained
        .scatter_matrix()
        .iter()
        .zip(single_pass.scatter_matrix().iter())
    {
        assert!((left - right).abs() < 1e-9);
    }
    for (left, right) in chained
        .components()
        .iter()
        .zip(single_pass.components().iter())
    {
        assert!((left - right).abs() < 1e-9);
    }
}
