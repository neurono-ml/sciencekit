## MODIFIED Requirements

### Requirement: Pure-Rust default backend is `faer`
The default math backend SHALL be `faer` (pure-Rust BLAS/LAPACK), which compiles on MSRV
1.85 (verified by the BLAS spike); `oxiblas` is rejected because it fails to compile on
Rust 1.85. `faer` is an internal implementation detail of the backends and SHALL NOT
appear in the public `SKMathBackend` surface.

#### Scenario: GEMM is available by default with no C toolchain
- **WHEN** a dense matrix product is requested on a default build
- **THEN** it is computed by the `faer` backend without requiring a C/Fortran BLAS

#### Scenario: SVD/QR/Cholesky are available by default
- **WHEN** a decomposition (SVD, QR or Cholesky) is requested on a default build
- **THEN** it is provided by `faer` without an FFI dependency

#### Scenario: Backend surface is backend-agnostic
- **WHEN** a consumer uses the `SKMathBackend` trait
- **THEN** it interacts only with `SKFloat` and ndarray types, never with a concrete
  backend's matrix type (e.g. `faer::Mat`)

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

## ADDED Requirements

### Requirement: Backend is generic over SKFloat
The `SKMathBackend` trait SHALL be generic over the sealed `SKFloat` bound (`F`), so the
same trait serves both `f32` and `f64` with no hard-coded float type in any signature.

#### Scenario: f32 and f64 are both supported
- **WHEN** a consumer requests any backend operation with `F = f32` or `F = f64`
- **THEN** the operation completes with the corresponding scalar type throughout

### Requirement: Host-centric zero-copy surface
The `SKMathBackend` trait SHALL accept zero-copy ndarray views (`ArrayView2<F>`) as
inputs and return owned ndarray results (`Array2<F>`). Decomposition results SHALL be
concrete ndarray containers generic over `F`, with the LU pivot exposed as a
host-normalized `Vec<usize>`.

#### Scenario: Inputs are zero-copy views
- **WHEN** an operation receives an `ArrayView2<F>`
- **THEN** no copy of the input elements occurs at the public boundary

#### Scenario: Results are owned ndarray containers
- **WHEN** an operation returns a matrix or decomposition
- **THEN** the result is an `Array2<F>` (or a decomposition struct holding `Array2<F>`),
  independent of the backend that produced it

### Requirement: Expanded linear-algebra kernel
The `SKMathBackend` SHALL provide the operations needed by the roadmap algorithms:
`lstsq`, `solve_triangular`, `solve`, `eigh`, `lu`, `slogdet`, `pinv`, `inv`, and `norm`
in addition to `gemm`, `svd`, `qr` and `cholesky`. `slogdet` SHALL return the sign and
log-absolute-determinant `(sign, log_abs_det)`. `norm` SHALL cover matrix and vector
norms, supporting the full scipy `ord` set with a general/arbitrary fallback alongside
the specialized paths.

#### Scenario: Least-squares solver is available
- **WHEN** `lstsq` is called with a possibly rank-deficient design matrix
- **THEN** it returns the minimum-norm least-squares solution `w` and residual
  information, stable under rank deficiency

#### Scenario: Log-determinant avoids underflow
- **WHEN** `slogdet` is called on a near-singular covariance matrix
- **THEN** it returns the sign and log-absolute-determinant without materializing the
  determinant (no underflow to `0.0`)

#### Scenario: Norm covers matrix and vector orders
- **WHEN** `norm` is called with any supported matrix or vector `ord`
- **THEN** it returns the corresponding norm, including the general/arbitrary fallback
  for unspecialized orders

### Requirement: Internal parallelism honours the execution plan
The backend SHALL honour the execution plan's `parallelism` level for internally
parallelizable operations (e.g. GEMM) rather than hard-coding a sequential execution.

#### Scenario: Parallel GEMM uses the configured parallelism
- **WHEN** a large GEMM is requested and the execution plan specifies multiple threads
- **THEN** the backend dispatches the product across the configured parallelism

### Requirement: Truncated decomposition roadmap for out-of-core
The capability SHALL plan `svds`/`eigsh` (truncated SVD and symmetric eigendecomposition)
as the primitives for out-of-core and streaming workloads, to be delivered with the
streaming/out-of-core milestone; streaming orchestration itself lives in the algorithm
layer.

#### Scenario: Truncated decompositions are on the roadmap
- **WHEN** the out-of-core milestone is planned
- **THEN** `svds` and `eigsh` are the specified primitives for incremental/streaming
  PCA and related workloads