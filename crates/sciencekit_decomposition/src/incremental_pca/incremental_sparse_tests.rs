//! Companion tests for sparse CSR Gram-based scatter accumulation without
//! densification.

use ndarray::{Array2, Axis};
use sciencekit_common::SKBackendKind;
use sciencekit_math::backend::sk_math_backend_for_kind;
use sprs::{CsMat, TriMat};

use super::accumulator::SKCovarianceAccumulator;
use super::merge::sk_update_dense;
use super::sparse::sk_update_sparse;

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
fn incremental_sparse_matches_dense_across_batches_without_densification() {
    let dense = Array2::from_shape_fn((48, 7), |(row, column)| {
        if (row * 3 + column * 5) % 4 == 0 {
            (row as f64 + 1.0) * 0.85 - (column as f64) * 1.4
        } else {
            0.0
        }
    });
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let mut dense_accumulator = SKCovarianceAccumulator::<f64>::new(7);
    let mut sparse_accumulator = SKCovarianceAccumulator::<f64>::new(7);

    for chunk in dense.axis_chunks_iter(Axis(0), 12) {
        sk_update_dense(&mut dense_accumulator, backend.as_ref(), chunk, 1);
        let owned_chunk = chunk.to_owned();
        let csr_chunk = csr_from_dense(&owned_chunk);
        sk_update_sparse(&mut sparse_accumulator, csr_chunk.view());
    }

    assert_eq!(sparse_accumulator.samples_seen(), 48);
    for (left, right) in sparse_accumulator
        .mean()
        .iter()
        .zip(dense_accumulator.mean().iter())
    {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }
    for (left, right) in sparse_accumulator
        .scatter()
        .iter()
        .zip(dense_accumulator.scatter().iter())
    {
        assert!((left - right).abs() < 1e-10, "left={left}, right={right}");
    }
}

#[test]
fn incremental_sparse_mixed_with_dense_batches_matches_single_dense_pass() {
    let dense = Array2::from_shape_fn((30, 6), |(row, column)| {
        if (row + column) % 3 == 0 {
            100.0 + (row as f64) * 0.5 - (column as f64) * 0.9
        } else {
            0.0
        }
    });
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let mut reference = SKCovarianceAccumulator::<f64>::new(6);
    sk_update_dense(&mut reference, backend.as_ref(), dense.view(), 1);

    let first = dense.slice_axis(Axis(0), ndarray::Slice::from(0..10)).to_owned();
    let second = dense.slice_axis(Axis(0), ndarray::Slice::from(10..22)).to_owned();
    let third = dense.slice_axis(Axis(0), ndarray::Slice::from(22..30));
    let first_csr = csr_from_dense(&first);
    let second_csr = csr_from_dense(&second);

    let mut mixed = SKCovarianceAccumulator::<f64>::new(6);
    sk_update_sparse(&mut mixed, first_csr.view());
    sk_update_dense(&mut mixed, backend.as_ref(), third, 1);
    sk_update_sparse(&mut mixed, second_csr.view());

    assert_eq!(mixed.samples_seen(), 30);
    for (left, right) in mixed.mean().iter().zip(reference.mean().iter()) {
        assert!((left - right).abs() < 1e-11);
    }
    for (left, right) in mixed.scatter().iter().zip(reference.scatter().iter()) {
        assert!((left - right).abs() < 1e-9);
    }
}
