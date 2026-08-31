## Purpose

The BLAS/LAPACK interface: a `SKMathBackend` abstraction with `FaerBackend` (pure-Rust
default per `wave-plan-foundation` Decision 1) and an opt-in `blas-backend` path wiring
`ndarray-linalg` + `blas-src` for OpenBLAS/BLIS/MKL/Accelerate (PRD §6.2).

## Requirements

### Requirement: Pure-Rust default backend is `faer`
The default math backend SHALL be `faer` (pure-Rust BLAS/LAPACK), which compiles on MSRV
1.85 (verified by the BLAS spike); `oxiblas` is rejected because it fails to compile on
Rust 1.85.

#### Scenario: GEMM is available by default with no C toolchain
- **WHEN** a dense matrix product is requested on a default build
- **THEN** it is computed by the `faer` backend without requiring a C/Fortran BLAS

#### Scenario: SVD/QR/Cholesky are available by default
- **WHEN** a decomposition (SVD, QR or Cholesky) is requested on a default build
- **THEN** it is provided by `faer` without an FFI dependency

### Requirement: Opt-in BLAS backend
The `blas-backend` feature SHALL swap heavy dense algebra to `ndarray-linalg` +
`blas-src` (OpenBLAS default, BLIS/MKL/Accelerate alternatives), gated by a feature flag
and never enabled by default.

#### Scenario: BLAS feature selects a system backend
- **WHEN** `blas-backend` is enabled
- **THEN** the math backend resolves to `ndarray-linalg` over the configured `blas-src`
  backend, and the pure-Rust `faer` path remains available as a fallback

#### Scenario: Default build has no BLAS feature
- **WHEN** the crate is built without features
- **THEN** no BLAS/LAPACK C library is linked and `faer` is used