## Why

`sciencekit_math` is the computational heart of the library: every algorithm builds on
its kernels (pairwise distances, dense/sparse products, reductions) and its layout/BLAS
choices. The PRD (§12 Phase 0 item 3) and the wave plan (`wave-plan-foundation`, Decision 1)
define it as the substrate for all downstream algorithm crates. The pure-Rust BLAS/LAPACK
default is now **`faer`** (Decision 1 resolved by spike: `oxiblas` fails MSRV 1.85). This
change creates the math crate with the performance-first conventions mandated by PRD §4
(`azip!`/`par_azip!`, zero-copy views, memory layout) and a SIMD/BLAS interface that is
pure-Rust by default and opt-in BLAS.

## What Changes

- Creation of the **`crates/sciencekit_math`** crate (second sub-crate, depends on
  `sciencekit_common`), containing:
  - **Higher-order operation kernels**: reusable `azip!`/`par_azip!`-based patterns
    (elementwise, axis reductions, in-place transforms) honoring PRD §4.2.
  - **Memory-layout helpers**: C/F-contiguous detection, `.to_owned()` to force
    contiguity before hot loops, SoA/AoS guidance.
  - **Pairwise distances**: Euclidean (squared + sqrt), Manhattan, Cosine — computed with
    zero-copy `ArrayView` inputs and a pure-Rust `euclidean_rdist`-style kernel.
  - **Sparse integration**: `sprs` CSR/CSC products (sparse×sparse, sparse×dense) with
    zero-copy views.
  - **SIMD/BLAS interface**: a `SKBlasBackend`-style trait with `FaerBackend` (pure-Rust
    default, Decision 1) and `MatrixMultiplyBackend` (fallback GEMM); opt-in
    `blas-backend` feature wiring `ndarray-linalg` + `blas-src` for OpenBLAS/BLIS/MKL/Accelerate.
  - **Backend selection**: the math crate exposes `SKMathBackend` so execution planning
    (Phase 0.4) can route heavy dense algebra to the selected backend.
- New workspace dependency: `faer 0.24` (pure-Rust BLAS/LAPACK), `matrixmultiply 0.3`
  (GEMM fallback), `sprs` (already in common), `ndarray` (already), `wide` (SIMD hot paths).

Out of scope: any algorithm (Phase 1+), GPU backends (separate changes), allocator wiring
(Phase 0.4), Python bindings.

## Capabilities

### New Capabilities

- `pairwise-distances`: zero-copy distance kernels (Euclidean/Manhattan/Cosine) with
  SIMD-friendly `euclidean_rdist` formulation.
- `higher-order-kernels`: reusable `azip!`/`par_azip!` operations (elementwise, axis
  reductions) with layout-aware execution.
- `dense-sparse-products`: `sprs` CSR/CSC product kernels (sparse×dense, sparse×sparse)
  via zero-copy views.
- `blas-interface`: the `SKMathBackend` abstraction with `FaerBackend` (pure-Rust default)
  and the opt-in `blas-backend` path (OpenBLAS/BLIS/MKL/Accelerate via `ndarray-linalg`).

### Modified Capabilities

None — first addition to the math crate.

## Impact

- **Code:** new crate `crates/sciencekit_math`; depends on `sciencekit_common`.
- **Dependencies:** `faer 0.24` (pure-Rust BLAS default per Decision 1), `matrixmultiply 0.3`
  (GEMM fallback), `wide` (SIMD), plus `ndarray`/`sprs` (transitive from common).
- **Downstream:** all algorithm crates consume `sciencekit_math` kernels; Phase 0.4
  (execution decision) routes dense algebra through `SKMathBackend`.
- **Acceptance (PRD §8.7/§10.3):** distance kernels run correctly with large and small
  data; correctness under concurrency; kernels are pure functions returning owned output
  or borrowing views; companion `*_tests.rs` modules with mock data in `ndarray`/`sprs`;
  complete nomenclature with correct prefixes; files ≤ 200 lines.

**Decision recorded from `wave-plan-foundation`:** pure-Rust BLAS default is `faer` — the
spike (`temporary/2026-08-31/blas-spike/`) confirmed `oxiblas` fails to compile on MSRV
1.85. `ndarray-linalg`+`blas-src` is the opt-in `blas-backend` path, not the default.