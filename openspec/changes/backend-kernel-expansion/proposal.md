## Why

The `SKMathBackend` abstraction (spec `blas-interface`) hard-codes `f64` and leaks the
`faer` matrix types (`Mat`, `MatRef`) into the public trait surface and its result
containers, coupling every consumer to a single concrete backend and a single numeric
type. The kernel is also too thin to serve the first algorithm on the roadmap
(`LinearRegression`), and `backend/mod.rs` violates the "mod.rs is a pure dispatcher"
convention. The backend must be generic over `SKFloat`, host-centric (zero-copy ndarray
in/out) and rich enough to support the dense linear algebra the upcoming algorithms
require.

## What Changes

- **BREAKING**: `SKMathBackend` becomes generic over `F: SKFloat` (`SKMathBackend<F>`),
  removing the hard-coded `f64` from every signature (`alpha: F`, `Vec<F>`, etc.).
- **BREAKING**: Remove `faer::{Mat, MatRef}` from the public trait surface. The trait
  takes zero-copy ndarray views (`ArrayView2<F>`) as inputs and returns owned ndarray
  results (`Array2<F>`) — the host-centric "lean" design (Decision: no associated matrix
  types; GPU is a deferred, separate change).
- **BREAKING**: Result containers become concrete and generic over `F`:
  `SKSingularValueDecomposition<F>`, `SKQRDecomposition<F>`, plus new
  `SKLUDecomposition<F>`. LU pivot is a host-normalized `Vec<usize>` (no `PivotInfo`
  associated type).
- Expand the kernel with the operations the roadmap algorithms need:
  `lstsq`, `solve_triangular`, `solve`, `eigh`, `lu`, `slogdet`, `pinv`, `inv`, and an
  expanded `norm` (matrix and vector, full scipy `ord` set with a general/arbitrary
  fallback alongside the specialized paths).
- Plan (not defer indefinitely) `svds`/`eigsh` (truncated SVD/eigh) as the
  out-of-core/streaming primitives; the streaming orchestration itself lives in the
  algorithm layer.
- Internal parallelism: the backend honours the execution plan's `parallelism`
  (e.g. faer `Par`) instead of a hard-coded sequential GEMM.
- Restructure `crates/sciencekit_math/src/backend/` into submodules with a pure
  dispatcher `mod.rs` (`kernel.rs`, `decompositions.rs`, `dispatch.rs`, `faer_backend.rs`,
  `ndarray_backend.rs`).
- **BREAKING**: `slogdet` returns `(sign, log_abs_det)`; `norm` covers matrix and vector
  norms.

## Capabilities

### New Capabilities
<!-- None: all behavior changes live in the existing blas-interface capability. -->

### Modified Capabilities
- `blas-interface`: the `SKMathBackend` trait is genericized over `SKFloat`, made
  host-centric over ndarray (no `faer` types in the public surface), and its operation
  set is expanded (`lstsq`, `solve_triangular`, `solve`, `eigh`, `lu`, `slogdet`, `pinv`,
  `inv`, expanded `norm`) with internal parallelism and an out-of-core/truncated
  (`svds`/`eigsh`) roadmap. Result containers and the LU pivot become concrete ndarray
  types. This is a behavior change at the spec level (public contract of the backend).

## Impact

- `crates/sciencekit_math/src/backend/` — rewritten (trait, containers, dispatch,
  modularization).
- `crates/sciencekit_math/src/lib.rs` — re-exports updated for the generic/host-centric
  surface.
- `crates/sciencekit_math/Cargo.toml` — no new dependencies expected; `faer` remains an
  internal implementation detail of the backends.
- Tests: `backend_tests.rs` (and companion tests) updated for the generic surface and new
  operations; acceptance per PRD §8.7 (large and small data, concurrency, export).
- **Dependency / ordering**: this change owns the restructure of `backend/mod.rs` (the
  14th module under the pure-dispatcher rule). The companion change `module-hygiene`
  fixes the remaining 13 non-conforming `mod.rs` files and adds the "tests at the end"
  rule to `AGENTS.md`, scoping `backend/` out. Apply this change before `module-hygiene`
  so the backend restructure lands once, without rework.