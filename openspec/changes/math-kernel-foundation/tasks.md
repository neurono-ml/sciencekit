# Tasks — math-kernel-foundation

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — use folder modules.

## 1. Crate structure

- [ ] 1.1 Create `crates/sciencekit_math` in the workspace with manifest (depends on
      `sciencekit_common`), declared license, empty folder-module tree (`kernels/`,
      `layout/`, `pairwise/`, `sparse_ops/`, `backend/`), compiling green
- [ ] 1.2 Add dependencies: `faer 0.24` (pure-Rust BLAS default), `matrixmultiply 0.3`,
      `wide`, plus `ndarray`/`sprs` (transitive via common); declare the `blas-backend`
      feature wiring `ndarray-linalg` + `blas-src`

## 2. Higher-order kernels

- [ ] 2.1 TDD of elementwise transform via `azip!` (e.g. `2x + 1`)
- [ ] 2.2 TDD of elementwise binary combine via `zip_mut_with`
- [ ] 2.3 TDD of axis-wise reduction (column sum on row-major) with layout awareness
- [ ] 2.4 TDD of in-place transform (scalar multiply) that avoids allocation

## 3. Layout helpers

- [ ] 3.1 TDD of contiguity detection (C/F-contiguous, strided)
- [ ] 3.2 TDD of a helper that forces contiguity (`to_owned` when strided, no-op when contiguous)

## 4. Pairwise distances

- [ ] 4.1 TDD of squared Euclidean via `||a||² − 2a·b + ||b||²` (symmetric, zero diagonal,
      non-negative)
- [ ] 4.2 TDD of Manhattan (L1 sum of abs differences)
- [ ] 4.3 TDD of cosine distance (`1 − cosθ`; identical=0, orthogonal=1)
- [ ] 4.4 SIMD hot path with `wide` for the Euclidean dot kernel

## 5. Sparse products

- [ ] 5.1 TDD of CSR×dense product via `CsMatView` (matches dense reference; sparse not densified)
- [ ] 5.2 TDD of sparse×sparse product via `sprs`

## 6. BLAS interface

- [ ] 6.1 TDD of the `SKMathBackend` trait (GEMM, SVD, QR, Cholesky signatures)
- [ ] 6.2 Implement `FaerBackend` (pure-Rust default) with GEMM + SVD + QR + Cholesky;
      verify it compiles on Rust 1.85 (Decision 1)
- [ ] 6.3 Implement the `blas-backend` feature path (`ndarray-linalg` + `blas-src`) as an
      alternative backend, gated off by default

## 7. Acceptance and review

- [ ] 7.1 Run all local gates (fmt, strict clippy, tests, doctests) and confirm green
- [ ] 7.2 Verify the change's adapted acceptance checklist: kernels correct with large and
      small data, correct under concurrency, pure functions returning owned/borrowed
      output, companion coverage, complete nomenclature with correct prefixes, no file
      beyond 200 lines
- [ ] 7.3 Record on the PR that full PRD §8.7/§10.3 acceptance (export + metrics) activates
      with the first estimator (Phase 1)