//! Companion tests for covariance finalization and equality with `SKPCA`.

use ndarray::{Array2, Axis, array};
use sciencekit_common::{SKBackendKind, SKBuilder, SKUnsupervisedFit};
use sciencekit_math::backend::sk_math_backend_for_kind;

use super::accumulator::SKCovarianceAccumulator;
use super::finalize::sk_finalize_accumulator;
use super::merge::sk_update_dense;
use crate::{SKPCA, SKPCASolver};

#[test]
fn incremental_finalize_sorted_components_and_equals_full_pca() {
    let data = Array2::from_shape_fn((60, 6), |(row, column)| {
        let angle = (row as f64 + 1.0) * 0.21;
        match column {
            0 => angle.sin() * 9.0,
            1 => angle.cos() * 4.5,
            2 => (angle * 2.0).sin() * 2.0,
            _ => ((row * 7 + column * 11) % 13) as f64 * 0.15 + column as f64,
        }
    });
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let mut accumulator = SKCovarianceAccumulator::<f64>::new(6);
    for chunk in data.axis_chunks_iter(Axis(0), 13) {
        sk_update_dense(&mut accumulator, backend.as_ref(), chunk, 1);
    }

    let finalized = sk_finalize_accumulator(&accumulator, backend.as_ref(), 4).unwrap();
    assert_eq!(finalized.components.dim(), (4, 6));

    // Explained variance is non-increasing.
    for pair in finalized
        .explained_variance
        .as_slice()
        .unwrap()
        .windows(2)
    {
        assert!(pair[0] >= pair[1]);
    }

    // Components are orthonormal: W * W^T = I_k.
    let gram = finalized.components.dot(&finalized.components.t());
    for ((row, column), &value) in gram.indexed_iter() {
        let target = if row == column { 1.0 } else { 0.0 };
        assert!((value - target).abs() < 1e-11);
    }

    // Matches exact Full PCA on the same dataset.
    let full_pca = SKPCA::<f64>::build()
        .number_of_components(4)
        .svd_solver(SKPCASolver::Full)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    for (left, right) in accumulator.mean().iter().zip(full_pca.mean().iter()) {
        assert!((left - right).abs() < 1e-12);
    }
    for (left, right) in finalized
        .explained_variance
        .iter()
        .zip(full_pca.explained_variance().iter())
    {
        assert!((left - right).abs() < 1e-10, "left={left}, right={right}");
    }
    for (left, right) in finalized
        .explained_variance_ratio
        .iter()
        .zip(full_pca.explained_variance_ratio().iter())
    {
        assert!((left - right).abs() < 1e-11, "left={left}, right={right}");
    }
    for (left, right) in finalized
        .singular_values
        .iter()
        .zip(full_pca.singular_values().iter())
    {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
    for (left, right) in finalized
        .components
        .iter()
        .zip(full_pca.components().iter())
    {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
}

#[test]
fn incremental_finalize_handles_single_sample_and_constant_features() {
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);
    let mut single = SKCovarianceAccumulator::<f64>::new(3);
    sk_update_dense(
        &mut single,
        backend.as_ref(),
        array![[2.0_f64, -4.0, 6.0]].view(),
        1,
    );
    let finalized = sk_finalize_accumulator(&single, backend.as_ref(), 1).unwrap();
    assert_eq!(finalized.components.dim(), (1, 3));
    assert_eq!(finalized.explained_variance[0], 0.0);
    assert_eq!(finalized.explained_variance_ratio[0], 0.0);
}
