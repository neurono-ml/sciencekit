//! TDD tests for the BLAS interface (spec `blas-interface`).

use faer::{Mat, MatRef, Unbind};

use super::{SKFaerBackend, SKMathBackend, SKMatrixMultiplyBackend, sk_default_math_backend};

/// Build a faer matrix from row-major slice data.
fn mat_from_rows(rows: &[&[f64]]) -> Mat<f64> {
    let (nrows, ncols) = (rows.len(), rows[0].len());
    Mat::from_fn(nrows, ncols, |i, j| rows[i.unbound()][j.unbound()])
}

/// Assert two matrices agree element-wise within a tolerance.
fn assert_close(actual: &MatRef<f64>, expected: &MatRef<f64>, tolerance: f64) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let diff = (actual[(i, j)] - expected[(i, j)]).abs();
            assert!(
                diff < tolerance,
                "mismatch at ({i},{j}): {} vs {}",
                actual[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

/// GEMM matches a hand-computed reference product.
#[test]
fn gemm_matches_reference_product() {
    let a = mat_from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b = mat_from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
    let backend = SKFaerBackend::new();
    let product = backend.gemm(a.as_ref(), b.as_ref(), 1.0);
    // [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    let expected = mat_from_rows(&[&[19.0, 22.0], &[43.0, 50.0]]);
    assert_close(&product.as_ref(), &expected.as_ref(), 1e-12);
}

/// GEMM honours the scaling factor `α`.
#[test]
fn gemm_applies_scalar_factor() {
    let a = mat_from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let b = mat_from_rows(&[&[2.0, 3.0], &[4.0, 5.0]]);
    let backend = SKFaerBackend::new();
    let product = backend.gemm(a.as_ref(), b.as_ref(), 2.0);
    let expected = mat_from_rows(&[&[4.0, 6.0], &[8.0, 10.0]]);
    assert_close(&product.as_ref(), &expected.as_ref(), 1e-12);
}

/// SVD reconstructs its input: `A ≈ U Σ Vᵀ`.
#[test]
fn svd_reconstructs_input() {
    let a = mat_from_rows(&[&[3.0, 1.0, 1.0], &[-1.0, 3.0, 1.0]]);
    let backend = SKFaerBackend::new();
    let decomposition = backend.svd(a.as_ref()).unwrap();
    let (m, n) = (decomposition.u.nrows(), decomposition.v.ncols());
    let mut diag = Mat::<f64>::zeros(m, n);
    for (index, &value) in decomposition.singular_values.iter().enumerate() {
        diag[(index, index)] = value;
    }
    let reconstructed = decomposition.u * diag * decomposition.v.transpose();
    assert_close(&reconstructed.as_ref(), &a.as_ref(), 1e-9);
}

/// QR reconstructs its input: `A = Q R`.
#[test]
fn qr_reconstructs_input() {
    let a = mat_from_rows(&[
        &[12.0, -51.0, 4.0],
        &[6.0, 167.0, -68.0],
        &[-4.0, 24.0, -41.0],
    ]);
    let backend = SKFaerBackend::new();
    let decomposition = backend.qr(a.as_ref());
    let reconstructed = decomposition.q * decomposition.r;
    assert_close(&reconstructed.as_ref(), &a.as_ref(), 1e-9);
}

/// Cholesky reconstructs a positive-definite input: `A = L Lᵀ`.
#[test]
fn cholesky_reconstructs_input() {
    // [[4, 2],[2, 3]] is positive-definite; L = [[2,0],[1,sqrt(2)]].
    let a = mat_from_rows(&[&[4.0, 2.0], &[2.0, 3.0]]);
    let backend = SKFaerBackend::new();
    let lower = backend.cholesky(a.as_ref()).unwrap();
    let reconstructed = lower.clone() * lower.transpose();
    assert_close(&reconstructed.as_ref(), &a.as_ref(), 1e-12);
}

/// The matrixmultiply fallback GEMM agrees with the faer GEMM.
#[test]
fn matrixmultiply_gemm_matches_faer() {
    let a = mat_from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let b = mat_from_rows(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
    let faer = SKFaerBackend::new().gemm(a.as_ref(), b.as_ref(), 1.5);
    let mm = SKMatrixMultiplyBackend::new().gemm(a.as_ref(), b.as_ref(), 1.5);
    assert_close(&mm.as_ref(), &faer.as_ref(), 1e-12);
}

/// The default backend exists, is thread-safe, and computes GEMM.
#[test]
fn default_backend_is_send_sync_and_usable() {
    fn assert_send_sync<T: Send + Sync>() {}
    let backend = sk_default_math_backend();
    assert_send_sync::<Box<dyn SKMathBackend>>();
    let a = mat_from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let b = mat_from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]);
    let product = backend.gemm(a.as_ref(), b.as_ref(), 1.0);
    assert_close(&product.as_ref(), &b.as_ref(), 1e-12);
}

/// The default backend is the pure-Rust faer backend on a default build.
#[test]
fn default_build_resolves_to_faer() {
    let backend = sk_default_math_backend();
    let a = mat_from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b = mat_from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
    let product = backend.gemm(a.as_ref(), b.as_ref(), 1.0);
    assert_eq!(product[(0, 0)], 19.0);
}
