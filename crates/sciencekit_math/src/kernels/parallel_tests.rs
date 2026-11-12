//! Companion TDD tests for row-parallel kernel execution and concurrency
//! (spec `kernel-functions`).

use std::thread;

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sprs::TriMat;

use super::{SKKernel, SKKernelParameters, sk_kernel_matrix, sk_kernel_matrix_sequential};

#[test]
fn parallel_kernel_matches_sequential_on_large_dense_and_sparse_inputs() {
    let left_dense = Array2::from_shape_fn((96, 16), |(r, c)| {
        ((r * 7 + c * 3) % 19) as f64 * 0.05 - 0.4
    });
    let right_dense = Array2::from_shape_fn((80, 16), |(r, c)| {
        ((r * 5 + c * 11) % 17) as f64 * 0.05 - 0.3
    });

    let mut triplets = TriMat::new(right_dense.dim());
    right_dense
        .indexed_iter()
        .filter(|(_, value)| value.abs() > 0.15)
        .for_each(|((r, c), &v)| triplets.add_triplet(r, c, v));
    let right_csr = triplets.to_csr();

    let catalogue = [
        SKKernelParameters::new(SKKernel::Linear, 0.0, 1, 0.0),
        SKKernelParameters::new(SKKernel::Polynomial, 0.2, 3, 0.5),
        SKKernelParameters::new(SKKernel::RadialBasisFunction, 0.3, 1, 0.0),
        SKKernelParameters::new(SKKernel::Sigmoid, 0.15, 1, -0.25),
    ];

    for parameters in &catalogue {
        let parallel_dense = sk_kernel_matrix(
            parameters,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Dense(right_dense.view()),
        )
        .unwrap();
        let sequential_dense = sk_kernel_matrix_sequential(
            parameters,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Dense(right_dense.view()),
        )
        .unwrap();
        assert_eq!(parallel_dense.dim(), (96, 80));
        parallel_dense
            .iter()
            .zip(sequential_dense.iter())
            .for_each(|(&par, &seq)| assert!((par - seq).abs() < 1e-12));

        let parallel_mixed = sk_kernel_matrix(
            parameters,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Sparse(right_csr.view()),
        )
        .unwrap();
        let sequential_mixed = sk_kernel_matrix_sequential(
            parameters,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Sparse(right_csr.view()),
        )
        .unwrap();
        parallel_mixed
            .iter()
            .zip(sequential_mixed.iter())
            .for_each(|(&par, &seq)| assert!((par - seq).abs() < 1e-12));
    }
}

#[test]
fn concurrent_kernel_calls_are_deterministic_across_threads() {
    let features =
        Array2::from_shape_fn((72, 12), |(r, c)| ((r * 3 + c * 5) % 13) as f64 * 0.1 - 0.5);
    let parameters = SKKernelParameters::new(SKKernel::RadialBasisFunction, 0.35, 1, 0.0);
    let reference = sk_kernel_matrix_sequential(
        &parameters,
        SKDataView::Dense(features.view()),
        SKDataView::Dense(features.view()),
    )
    .unwrap();

    thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let view = features.view();
                scope.spawn(move || {
                    sk_kernel_matrix(
                        &parameters,
                        SKDataView::Dense(view),
                        SKDataView::Dense(view),
                    )
                    .unwrap()
                })
            })
            .collect();
        for handle in handles {
            let candidate = handle.join().unwrap();
            assert_eq!(candidate, reference);
        }
    });
}

#[test]
fn kernel_catalogue_supports_f32_across_dense_and_sparse_views() {
    let left_dense = Array2::from_shape_fn((12, 4), |(row, col)| {
        ((row * 5 + col * 3) % 11) as f32 * 0.1 - 0.3
    });
    let mut triplets = TriMat::new((10, 4));
    for row in 0..10 {
        triplets.add_triplet(row, row % 4, (row as f32 + 1.0) * 0.2);
    }
    let right_csr = triplets.to_csr();
    let right_dense = right_csr.to_dense();

    let catalogue = [
        SKKernelParameters::new(SKKernel::Linear, 0.0_f32, 1, 0.0),
        SKKernelParameters::new(SKKernel::Polynomial, 0.25_f32, 2, 0.5),
        SKKernelParameters::new(SKKernel::RadialBasisFunction, 0.4_f32, 1, 0.0),
        SKKernelParameters::new(SKKernel::Sigmoid, 0.2_f32, 1, -0.1),
    ];

    for parameters in &catalogue {
        let dense_result = sk_kernel_matrix(
            parameters,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Dense(right_dense.view()),
        )
        .unwrap();
        let mixed_result = sk_kernel_matrix(
            parameters,
            SKDataView::Dense(left_dense.view()),
            SKDataView::Sparse(right_csr.view()),
        )
        .unwrap();
        assert_eq!(dense_result.dim(), (12, 10));
        dense_result
            .iter()
            .zip(mixed_result.iter())
            .for_each(|(&dense_value, &mixed_value)| {
                assert!((dense_value - mixed_value).abs() < 1e-5_f32);
            });
    }
}
