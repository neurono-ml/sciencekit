//! TDD tests for the BLAS interface (spec `blas-interface`).
//!
//! The suite is host-centric: fixtures are built from `ndarray`/`sprs` and no
//! `faer` type appears in these tests. All fixtures are generic over the
//! sealed [`SKFloat`] bound so the same checks run for `f32` and `f64`.

use ndarray::{Array1, Array2};
use sciencekit_common::SKFloat;

use super::{
    SKFaerBackend, SKLeastSquaresSolution, SKMathBackend, SKMatrixMultiplyBackend, SKNormKind,
    SKQRDecomposition, SKSingularValueDecomposition, sk_default_math_backend,
};

/// Build a dense row-major matrix from slice data.
fn mat_from_rows<F: SKFloat>(rows: &[&[F]]) -> Array2<F> {
    let (nrows, ncols) = (rows.len(), rows[0].len());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| rows[i][j])
}

/// Assert two matrices agree element-wise within a tolerance.
fn assert_close<F: SKFloat + std::fmt::Debug>(actual: &Array2<F>, expected: &Array2<F>, tolerance: F) {
    assert_eq!(actual.shape(), expected.shape());
    for ((i, j), value) in actual.indexed_iter() {
        let expected_value = expected[(i, j)];
        let diff = (*value - expected_value).abs();
        assert!(
            diff < tolerance,
            "mismatch at ({i},{j}): {:?} vs {:?}",
            value,
            expected_value
        );
    }
}

/// GEMM matches a hand-computed reference product (generic over `F`).
fn check_gemm_matches_reference_product<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[&[F::one(), F::from(2.0).unwrap()], &[F::from(3.0).unwrap(), F::from(4.0).unwrap()]]);
    let b = mat_from_rows(&[&[F::from(5.0).unwrap(), F::from(6.0).unwrap()], &[F::from(7.0).unwrap(), F::from(8.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let product = backend.gemm(a.view(), b.view(), F::one(), 1);
    let expected = mat_from_rows(&[&[F::from(19.0).unwrap(), F::from(22.0).unwrap()], &[F::from(43.0).unwrap(), F::from(50.0).unwrap()]]);
    assert_close(&product, &expected, F::from(1e-4).unwrap());
}

#[test]
fn gemm_matches_reference_product_f64() {
    check_gemm_matches_reference_product::<f64>();
}

#[test]
fn gemm_matches_reference_product_f32() {
    check_gemm_matches_reference_product::<f32>();
}

/// GEMM honours the scaling factor `α` (generic over `F`).
fn check_gemm_applies_scalar_factor<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[&[F::one(), F::zero()], &[F::zero(), F::one()]]);
    let b = mat_from_rows(&[&[F::from(2.0).unwrap(), F::from(3.0).unwrap()], &[F::from(4.0).unwrap(), F::from(5.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let product = backend.gemm(a.view(), b.view(), F::from(2.0).unwrap(), 1);
    let expected = mat_from_rows(&[&[F::from(4.0).unwrap(), F::from(6.0).unwrap()], &[F::from(8.0).unwrap(), F::from(10.0).unwrap()]]);
    assert_close(&product, &expected, F::from(1e-4).unwrap());
}

#[test]
fn gemm_applies_scalar_factor_f64() {
    check_gemm_applies_scalar_factor::<f64>();
}

#[test]
fn gemm_applies_scalar_factor_f32() {
    check_gemm_applies_scalar_factor::<f32>();
}

/// SVD reconstructs its input: `A ≈ U Σ Vᵀ` (generic over `F`).
fn check_svd_reconstructs_input<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[&[F::from(3.0).unwrap(), F::one(), F::one()], &[-F::one(), F::from(3.0).unwrap(), F::one()]]);
    let backend = SKFaerBackend::new();
    let decomposition: SKSingularValueDecomposition<F> = backend.svd(a.view()).unwrap();
    let (m, n) = a.dim();
    let mut diag = Array2::<F>::zeros((m, n));
    for (index, &value) in decomposition.singular_values.iter().enumerate() {
        diag[(index, index)] = value;
    }
    let reconstructed = decomposition.u.dot(&diag).dot(&decomposition.v.t());
    assert_close(&reconstructed, &a, F::from(1e-3).unwrap());
}

#[test]
fn svd_reconstructs_input_f64() {
    check_svd_reconstructs_input::<f64>();
}

#[test]
fn svd_reconstructs_input_f32() {
    check_svd_reconstructs_input::<f32>();
}

/// QR reconstructs its input: `A = Q R` (generic over `F`).
fn check_qr_reconstructs_input<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(12.0).unwrap(), -F::from(51.0).unwrap(), F::from(4.0).unwrap()],
        &[F::from(6.0).unwrap(), F::from(167.0).unwrap(), -F::from(68.0).unwrap()],
        &[-F::from(4.0).unwrap(), F::from(24.0).unwrap(), -F::from(41.0).unwrap()],
    ]);
    let backend = SKFaerBackend::new();
    let decomposition: SKQRDecomposition<F> = backend.qr(a.view());
    let reconstructed = decomposition.q.dot(&decomposition.r);
    assert_close(&reconstructed, &a, F::from(1e-3).unwrap());
}

#[test]
fn qr_reconstructs_input_f64() {
    check_qr_reconstructs_input::<f64>();
}

#[test]
fn qr_reconstructs_input_f32() {
    check_qr_reconstructs_input::<f32>();
}

/// Cholesky reconstructs a positive-definite input: `A = L Lᵀ`.
fn check_cholesky_reconstructs_input<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[&[F::from(4.0).unwrap(), F::from(2.0).unwrap()], &[F::from(2.0).unwrap(), F::from(3.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let lower = backend.cholesky(a.view()).unwrap();
    let reconstructed = lower.dot(&lower.t());
    assert_close(&reconstructed, &a, F::from(1e-4).unwrap());
}

#[test]
fn cholesky_reconstructs_input_f64() {
    check_cholesky_reconstructs_input::<f64>();
}

#[test]
fn cholesky_reconstructs_input_f32() {
    check_cholesky_reconstructs_input::<f32>();
}

/// The default backend exists, is thread-safe, and computes GEMM.
fn check_default_backend_is_send_sync_and_usable<F: SKFloat + std::fmt::Debug>()
where
    SKFaerBackend: SKMathBackend<F>,
{
    let a = mat_from_rows(&[&[F::one(), F::zero()], &[F::zero(), F::one()]]);
    let b = mat_from_rows(&[&[F::from(2.0).unwrap(), F::zero()], &[F::zero(), F::from(3.0).unwrap()]]);
    let product = SKFaerBackend::new().gemm(a.view(), b.view(), F::one(), 1);
    assert_close(&product, &b, F::from(1e-4).unwrap());
}

#[test]
fn default_backend_is_send_sync_and_usable_f64() {
    check_default_backend_is_send_sync_and_usable::<f64>();
    fn assert_send_sync<T: Send + Sync>() {}
    let backend = sk_default_math_backend::<f64>();
    assert_send_sync::<Box<dyn SKMathBackend<f64>>>();
    let a = mat_from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let b = mat_from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]);
    let product = backend.gemm(a.view(), b.view(), 1.0, 1);
    assert_close(&product, &b, 1e-4);
}

#[test]
fn default_backend_is_send_sync_and_usable_f32() {
    check_default_backend_is_send_sync_and_usable::<f32>();
    fn assert_send_sync<T: Send + Sync>() {}
    let backend = sk_default_math_backend::<f32>();
    assert_send_sync::<Box<dyn SKMathBackend<f32>>>();
    let a = mat_from_rows(&[&[1.0f32, 0.0], &[0.0, 1.0]]);
    let b = mat_from_rows(&[&[2.0f32, 0.0], &[0.0, 3.0]]);
    let product = backend.gemm(a.view(), b.view(), 1.0, 1);
    assert_close(&product, &b, 1e-4);
}

/// The default backend is the pure-Rust faer backend on a default build.
#[test]
fn default_build_resolves_to_faer() {
    let backend = sk_default_math_backend::<f64>();
    let a = mat_from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b = mat_from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
    let product = backend.gemm(a.view(), b.view(), 1.0, 1);
    assert_eq!(product[(0, 0)], 19.0);
}

// ---------------------------------------------------------------------------
// Expanded kernel: solve_triangular, solve, eigh, lu, slogdet, pinv, inv,
// lstsq, norm/vector_norm.
// ---------------------------------------------------------------------------

/// `solve_triangular` recovers a known solution from an upper-triangular `A`.
fn check_solve_triangular_upper<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(2.0).unwrap(), F::from(1.0).unwrap()],
        &[F::zero(), F::from(3.0).unwrap()],
    ]);
    let b = mat_from_rows(&[&[F::from(5.0).unwrap()], &[F::from(6.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let x = backend.solve_triangular(a.view(), b.view(), false, false).unwrap();
    // 2x + y = 5; 3y = 6 => y = 2, x = 1.5
    assert!((x[(0, 0)] - F::from(1.5).unwrap()).abs() < F::from(1e-4).unwrap());
    assert!((x[(1, 0)] - F::from(2.0).unwrap()).abs() < F::from(1e-4).unwrap());
}

#[test]
fn solve_triangular_upper_f64() {
    check_solve_triangular_upper::<f64>();
}

#[test]
fn solve_triangular_upper_f32() {
    check_solve_triangular_upper::<f32>();
}

/// `solve_triangular` recovers a known solution from a lower-triangular `A`.
fn check_solve_triangular_lower<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(2.0).unwrap(), F::zero()],
        &[F::one(), F::from(3.0).unwrap()],
    ]);
    let b = mat_from_rows(&[&[F::from(4.0).unwrap()], &[F::from(5.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let x = backend.solve_triangular(a.view(), b.view(), true, false).unwrap();
    // 2x = 4 => x = 2; x + 3y = 5 => y = 1
    assert!((x[(0, 0)] - F::from(2.0).unwrap()).abs() < F::from(1e-4).unwrap());
    assert!((x[(1, 0)] - F::from(1.0).unwrap()).abs() < F::from(1e-4).unwrap());
}

#[test]
fn solve_triangular_lower_f64() {
    check_solve_triangular_lower::<f64>();
}

#[test]
fn solve_triangular_lower_f32() {
    check_solve_triangular_lower::<f32>();
}

/// `solve` recovers a known solution of a general square system.
fn check_solve_general<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(4.0).unwrap(), F::from(3.0).unwrap()],
        &[F::from(1.0).unwrap(), F::from(5.0).unwrap()],
    ]);
    let b = mat_from_rows(&[&[F::from(7.0).unwrap()], &[F::from(6.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let x = backend.solve(a.view(), b.view()).unwrap();
    let ax = a.dot(&x);
    assert_close(&ax, &b, F::from(1e-4).unwrap());
}

#[test]
fn solve_general_f64() {
    check_solve_general::<f64>();
}

#[test]
fn solve_general_f32() {
    check_solve_general::<f32>();
}

/// `eigh` reconstructs the input: `A ≈ V Σ Vᵀ` with ascending eigenvalues.
fn check_eigh_reconstructs_input<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(4.0).unwrap(), F::one(), F::zero()],
        &[F::one(), F::from(3.0).unwrap(), F::from(2.0).unwrap()],
        &[F::zero(), F::from(2.0).unwrap(), F::from(6.0).unwrap()],
    ]);
    let backend = SKFaerBackend::new();
    let (eigenvalues, eigenvectors) = backend.eigh(a.view()).unwrap();
    assert!(eigenvalues.len() == 3);
    let mut diag = Array2::<F>::zeros((3, 3));
    for (index, &value) in eigenvalues.iter().enumerate() {
        diag[(index, index)] = value;
    }
    let reconstructed = eigenvectors.dot(&diag).dot(&eigenvectors.t());
    assert_close(&reconstructed, &a, F::from(1e-3).unwrap());
}

#[test]
fn eigh_reconstructs_input_f64() {
    check_eigh_reconstructs_input::<f64>();
}

#[test]
fn eigh_reconstructs_input_f32() {
    check_eigh_reconstructs_input::<f32>();
}

/// `lu` satisfies `P A = L U`; reconstructs `P` from the pivot.
fn check_lu_reconstructs_pa_lu<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(4.0).unwrap(), F::from(3.0).unwrap(), F::from(2.0).unwrap()],
        &[F::from(1.0).unwrap(), F::from(5.0).unwrap(), F::from(3.0).unwrap()],
        &[F::from(2.0).unwrap(), F::one(), F::from(6.0).unwrap()],
    ]);
    let backend = SKFaerBackend::new();
    let decomposition = backend.lu(a.view()).unwrap();
    let n = a.nrows();
    let mut p = Array2::<F>::zeros((n, n));
    for (i, &pivot_row) in decomposition.pivot.iter().enumerate() {
        p[(i, pivot_row)] = F::one();
    }
    let pa = p.dot(&a);
    let lu = decomposition.l.dot(&decomposition.u);
    assert_close(&pa, &lu, F::from(1e-4).unwrap());
}

#[test]
fn lu_reconstructs_pa_lu_f64() {
    check_lu_reconstructs_pa_lu::<f64>();
}

#[test]
fn lu_reconstructs_pa_lu_f32() {
    check_lu_reconstructs_pa_lu::<f32>();
}

/// `slogdet` on a near-singular matrix returns a finite log-determinant (no
/// underflow to `0.0`).
fn check_slogdet_avoids_underflow<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let diagonal_value = F::from(1e-200).unwrap();
    let a = mat_from_rows(&[
        &[diagonal_value, F::zero()],
        &[F::zero(), diagonal_value],
    ]);
    let backend = SKFaerBackend::new();
    let (sign, log_abs_det) = backend.slogdet(a.view()).unwrap();
    assert!(sign == F::one());
    assert!(log_abs_det < F::zero());
    assert!(log_abs_det.is_finite());
}

#[test]
fn slogdet_avoids_underflow_f64() {
    check_slogdet_avoids_underflow::<f64>();
}

/// `pinv` satisfies the Moore–Penrose property `A pinv(A) A ≈ A`.
fn check_pinv_moore_penrose<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::one(), F::from(2.0).unwrap()],
        &[F::from(3.0).unwrap(), F::from(4.0).unwrap()],
        &[F::from(5.0).unwrap(), F::from(6.0).unwrap()],
    ]);
    let backend = SKFaerBackend::new();
    let pinv = backend.pinv(a.view()).unwrap();
    let reconstructed = a.dot(&pinv).dot(&a);
    assert_close(&reconstructed, &a, F::from(1e-3).unwrap());
}

#[test]
fn pinv_moore_penrose_f64() {
    check_pinv_moore_penrose::<f64>();
}

#[test]
fn pinv_moore_penrose_f32() {
    check_pinv_moore_penrose::<f32>();
}

/// `inv` inverts a square matrix: `A inv(A) = I`.
fn check_inv_inverts<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(4.0).unwrap(), F::from(3.0).unwrap()],
        &[F::from(1.0).unwrap(), F::from(5.0).unwrap()],
    ]);
    let backend = SKFaerBackend::new();
    let inverse = backend.inv(a.view()).unwrap();
    let identity = a.dot(&inverse);
    let expected = Array2::<F>::eye(2);
    assert_close(&identity, &expected, F::from(1e-4).unwrap());
}

#[test]
fn inv_inverts_f64() {
    check_inv_inverts::<f64>();
}

#[test]
fn inv_inverts_f32() {
    check_inv_inverts::<f32>();
}

/// `lstsq` recovers the exact solution of a consistent system.
fn check_lstsq_recovers_solution<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::one(), F::zero()],
        &[F::zero(), F::one()],
        &[F::one(), F::one()],
    ]);
    let b = mat_from_rows(&[&[F::one()], &[F::from(2.0).unwrap()], &[F::from(3.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let result: SKLeastSquaresSolution<F> = backend.lstsq(a.view(), b.view()).unwrap();
    assert_eq!(result.rank, 2);
    assert!((result.solution[(0, 0)] - F::one()).abs() < F::from(1e-4).unwrap());
    assert!((result.solution[(1, 0)] - F::from(2.0).unwrap()).abs() < F::from(1e-4).unwrap());
}

#[test]
fn lstsq_recovers_solution_f64() {
    check_lstsq_recovers_solution::<f64>();
}

#[test]
fn lstsq_recovers_solution_f32() {
    check_lstsq_recovers_solution::<f32>();
}

/// `lstsq` is stable under rank deficiency (returns the minimum-norm
/// solution).
fn check_lstsq_rank_deficient<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    // A has rank 1 (both columns equal); infinitely many solutions, the
    // minimum-norm one has both coordinates equal.
    let a = mat_from_rows(&[
        &[F::one(), F::one()],
        &[F::from(2.0).unwrap(), F::from(2.0).unwrap()],
        &[F::from(3.0).unwrap(), F::from(3.0).unwrap()],
    ]);
    let b = mat_from_rows(&[&[F::one()], &[F::from(2.0).unwrap()], &[F::from(3.0).unwrap()]]);
    let backend = SKFaerBackend::new();
    let result: SKLeastSquaresSolution<F> = backend.lstsq(a.view(), b.view()).unwrap();
    assert_eq!(result.rank, 1);
    assert!((result.solution[(0, 0)] - result.solution[(1, 0)]).abs() < F::from(1e-4).unwrap());
}

#[test]
fn lstsq_rank_deficient_f64() {
    check_lstsq_rank_deficient::<f64>();
}

/// Matrix `norm` matches hand-computed values for the common orders.
fn check_norm_matrix_orders<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let a = mat_from_rows(&[
        &[F::from(3.0).unwrap(), F::from(4.0).unwrap()],
        &[F::from(0.0).unwrap(), F::from(0.0).unwrap()],
    ]);
    let backend = SKFaerBackend::new();
    let frobenius = backend.norm(a.view(), SKNormKind::Frobenius).unwrap();
    assert!((frobenius - F::from(5.0).unwrap()).abs() < F::from(1e-4).unwrap());
    let infinity = backend.norm(a.view(), SKNormKind::Infinity).unwrap();
    assert!((infinity - F::from(7.0).unwrap()).abs() < F::from(1e-4).unwrap());
    let l1 = backend.norm(a.view(), SKNormKind::L1).unwrap();
    assert!((l1 - F::from(4.0).unwrap()).abs() < F::from(1e-4).unwrap());
}

#[test]
fn norm_matrix_orders_f64() {
    check_norm_matrix_orders::<f64>();
}

#[test]
fn norm_matrix_orders_f32() {
    check_norm_matrix_orders::<f32>();
}

/// Vector `norm` matches hand-computed values, including the `General`
/// p-norm fallback.
fn check_vector_norm_orders<F: SKFloat + std::fmt::Debug>()
where SKFaerBackend: SKMathBackend<F> {
    let x = Array1::from(vec![F::from(3.0).unwrap(), F::from(4.0).unwrap()]);
    let backend = SKFaerBackend::new();
    let l2 = backend.vector_norm(x.view(), SKNormKind::L2).unwrap();
    assert!((l2 - F::from(5.0).unwrap()).abs() < F::from(1e-4).unwrap());
    let l1 = backend.vector_norm(x.view(), SKNormKind::L1).unwrap();
    assert!((l1 - F::from(7.0).unwrap()).abs() < F::from(1e-4).unwrap());
    let general = backend
        .vector_norm(x.view(), SKNormKind::General { order: F::from(2.0).unwrap() })
        .unwrap();
    assert!((general - F::from(5.0).unwrap()).abs() < F::from(1e-4).unwrap());
}

#[test]
fn vector_norm_orders_f64() {
    check_vector_norm_orders::<f64>();
}

#[test]
fn vector_norm_orders_f32() {
    check_vector_norm_orders::<f32>();
}

// ---------------------------------------------------------------------------
// Internal parallelism honours the execution plan (Decision 5).
// ---------------------------------------------------------------------------

/// A large GEMM computed with a sequential plan matches the one computed with
/// a multi-threaded plan (results are unchanged).
#[test]
fn parallel_gemm_matches_sequential() {
    let n = 64usize;
    let a = Array2::<f64>::from_shape_fn((n, n), |(i, j)| ((i * n + j) % 7) as f64);
    let b = Array2::<f64>::from_shape_fn((n, n), |(i, j)| ((i + j) % 5) as f64);
    let backend = SKFaerBackend::new();
    let sequential = backend.gemm(a.view(), b.view(), 1.5, 1);
    let parallel = backend.gemm(a.view(), b.view(), 1.5, 4);
    assert_close(&parallel, &sequential, 1e-6);
}

// ---------------------------------------------------------------------------
// Compile-time assertion: the trait surface exposes no `faer` types.
// ---------------------------------------------------------------------------

/// A generic helper that exercises the full host-centric surface using only
/// `ndarray` types. This compiles if and only if the trait is backend-agnostic
/// (no `faer` types in the public surface).
fn assert_host_centric_surface<F: SKFloat, B: SKMathBackend<F> + ?Sized>(backend: &B, a: Array2<F>) {
    let _ = backend.gemm(a.view(), a.view(), F::one(), 1);
    let _: Result<SKSingularValueDecomposition<F>, _> = backend.svd(a.view());
    let _: SKQRDecomposition<F> = backend.qr(a.view());
    let _ = backend.cholesky(a.view());
    let _ = backend.solve_triangular(a.view(), a.view(), true, false);
    let _ = backend.solve(a.view(), a.view());
    let _ = backend.eigh(a.view());
    let _ = backend.lu(a.view());
    let _ = backend.slogdet(a.view());
    let _ = backend.pinv(a.view());
    let _ = backend.inv(a.view());
    let _ = backend.lstsq(a.view(), a.view());
    let _ = backend.norm(a.view(), SKNormKind::Frobenius);
    let _ = backend.vector_norm(a.slice(ndarray::s![.., 0]), SKNormKind::L2);
}

#[test]
fn trait_surface_is_host_centric() {
    let a = mat_from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    assert_host_centric_surface(&SKFaerBackend::new(), a.clone());
    let backend = sk_default_math_backend::<f64>();
    assert_host_centric_surface(&*backend, a);
}

// ---------------------------------------------------------------------------
// Acceptance (PRD §8.7): lots and little data, concurrency, exportability.
// ---------------------------------------------------------------------------

/// The full kernel runs identically on little (2×2) and large (64×64) inputs.
#[test]
fn kernel_runs_on_small_and_large_data() {
    let backend = SKFaerBackend::new();

    // Small: a well-conditioned 2×2.
    let small = mat_from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]);
    let small_svd = backend.svd(small.view()).unwrap();
    assert_eq!(small_svd.singular_values.len(), 2);
    let small_solve = backend.solve(small.view(), mat_from_rows(&[&[1.0], &[2.0]]).view()).unwrap();
    assert!((small_solve[(0, 0)] - 0.0909f64).abs() < 1e-3);
    let small_inv = backend.inv(small.view()).unwrap();
    let small_identity = small.dot(&small_inv);
    assert_close(&small_identity, &Array2::<f64>::eye(2), 1e-6);

    // Large: a 64×64 SPD matrix, exercises every decomposition.
    let n = 64usize;
    let large = Array2::<f64>::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            n as f64 + (i as f64)
        } else {
            0.01 * ((i * n + j) % 13) as f64
        }
    });
    let large_svd = backend.svd(large.view()).unwrap();
    assert_eq!(large_svd.singular_values.len(), n);
    let large_qr = backend.qr(large.view());
    assert_close(&large_qr.q.dot(&large_qr.r), &large, 1e-6);
    let large_lu = backend.lu(large.view()).unwrap();
    let mut p = Array2::<f64>::zeros((n, n));
    for (i, &row) in large_lu.pivot.iter().enumerate() {
        p[(i, row)] = 1.0;
    }
    assert_close(&p.dot(&large), &large_lu.l.dot(&large_lu.u), 1e-6);
    let (sign, log_abs_det) = backend.slogdet(large.view()).unwrap();
    assert!(sign == 1.0 || sign == -1.0);
    assert!(log_abs_det.is_finite());
    let large_inv = backend.inv(large.view()).unwrap();
    assert_close(&large.dot(&large_inv), &Array2::<f64>::eye(n), 1e-6);
}

/// Every backend is `Send + Sync` and safe to share across a rayon pool.
#[test]
fn backends_are_send_sync_and_concurrency_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKFaerBackend>();
    assert_send_sync::<SKMatrixMultiplyBackend>();
    assert_send_sync::<Box<dyn SKMathBackend<f64>>>();
    assert_send_sync::<Box<dyn SKMathBackend<f32>>>();

    let backend = std::sync::Arc::new(SKFaerBackend::new());
    let handles: Vec<_> = (0..8)
        .map(|thread| {
            let backend = std::sync::Arc::clone(&backend);
            std::thread::spawn(move || {
                let a = Array2::<f64>::from_shape_fn((8, 8), |(i, j)| {
                    ((i + thread * 7) % 11 + j % 3) as f64
                });
                let b = Array2::<f64>::from_shape_fn((8, 8), |(i, j)| {
                    ((i % 5 + j + thread) % 7) as f64
                });
                let product = backend.gemm(a.view(), b.view(), 1.0, 2);
                let _svd = backend.svd(a.view()).unwrap();
                let _lstsq = backend.lstsq(a.view(), b.view()).unwrap();
                product[(0, 0)]
            })
        })
        .collect();
    // Distinct threads see distinct results; no panics occur under contention.
    let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(results.windows(2).any(|w| w[0] != w[1]));
}

/// `kind()` is recorded and results are comparable across backends.
#[test]
fn backend_kind_is_recorded_and_results_are_exportable() {
    fn kind_f64<B: SKMathBackend<f64> + ?Sized>(backend: &B) -> sciencekit_common::SKBackendKind {
        backend.kind()
    }
    let faer = SKFaerBackend::new();
    let matrix_multiply = SKMatrixMultiplyBackend::new();
    assert_eq!(kind_f64(&faer), sciencekit_common::SKBackendKind::Faer);
    assert_eq!(
        kind_f64(&matrix_multiply),
        sciencekit_common::SKBackendKind::MatrixMultiply
    );
    let default = sk_default_math_backend::<f64>();
    assert_eq!(kind_f64(&*default), sciencekit_common::SKBackendKind::Faer);

    // The same input produces (within tolerance) the same result across
    // backends, so a model/result can move between them.
    let a = mat_from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);
    let faer_qr = faer.qr(a.view());
    let mm_qr = matrix_multiply.qr(a.view());
    assert_close(&faer_qr.q, &mm_qr.q, 1e-6);
    assert_close(&faer_qr.r, &mm_qr.r, 1e-6);

    // Decompositions are plain `Array2` containers: cloneable, serializable,
    // backend-independent.
    let faer_svd = faer.svd(a.view()).unwrap();
    let exported_u = faer_svd.u.clone();
    let _reconstructed = exported_u.clone().dot(&Array2::<f64>::eye(3));
}

// ---------------------------------------------------------------------------
// Out-of-core roadmap marker (Decision 8).
// ---------------------------------------------------------------------------

/// The host-centric surface does not preclude adding truncated decompositions
/// (`svds`/`eigsh`) later: it only fixes the concrete container types, which
/// a truncated SVD/eigh also returns.
#[test]
fn truncated_decomposition_roadmap_is_not_precluded() {
    // The trait result containers hold owned `Array2<F>` plus the scalar
    // values; a future truncated `svds`/`eigsh` (out-of-core) returns the
    // same types for a bounded rank, so the surface stays additive.
    fn assert_concrete_containers<F: SKFloat>() {
        let _: SKSingularValueDecomposition<F> =
            SKSingularValueDecomposition {
                u: Array2::<F>::zeros((2, 2)),
                singular_values: vec![F::one(), F::one()],
                v: Array2::<F>::zeros((2, 2)),
            };
    }
    assert_concrete_containers::<f64>();
    assert_concrete_containers::<f32>();
}