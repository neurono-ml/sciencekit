# sciencekit_math — computational kernels and BLAS interface

The `sciencekit_math` crate is the numeric substrate every downstream algorithm crate
builds on (Phase 0.3). It provides the reusable operation kernels, memory-layout helpers,
pairwise-distance matrices, sparse products and the BLAS/LAPACK backend abstraction that the
algorithm crates consume.

## Purpose

- **Performance-first** kernels honouring PRD §4: higher-order iteration with
  `azip!`/`par_azip!` (never manual index loops), SIMD hot paths, and layout awareness.
- **Zero-copy** public APIs built on `ndarray::ArrayView`, `ndarray::CowArray` and `sprs`
  views.
- **A uniform linear-algebra abstraction** ([`SKMathBackend`]) with a pure-Rust default
  (`faer`) and an opt-in system-BLAS path.

## Modules

| Module | Responsibility |
|---|---|
| `kernels` | Elementwise transform, binary combine, axis reductions and in-place scaling. |
| `layout` | C/F-contiguity detection and contiguity-forcing. |
| `pairwise` | Squared-Euclidean (`‖a‖² − 2a·b + ‖b‖²`), Euclidean, Manhattan and Cosine distance matrices. |
| `sparse_ops` | `sprs` CSR×dense and sparse×sparse products over zero-copy views. |
| `backend` | `SKMathBackend` trait + `SKFaerBackend` (default), `SKMatrixMultiplyBackend`, and the opt-in `blas-backend` feature. |

```mermaid
flowchart LR
    A[`sciencekit_math`] --> B[kernels]
    A --> C[layout]
    A --> D[pairwise]
    A --> E[sparse_ops]
    A --> F[backend]
    B --> G[`azip!` / `par_azip!` / `zip_mut_with`]
    C --> H[`sk_memory_layout` / `sk_force_contiguous`]
    D --> I[`wide` SIMD dot kernel]
    E --> J[`sprs` views]
    F --> K[`SKFaerBackend` (pure Rust)]
    F --> L[`blas-backend` (opt-in)]
```

## Key public API

- **Kernels** — `sk_elementwise_transform`, `sk_binary_combine`, `sk_axis_sum`,
  `sk_scale_in_place`.
- **Layout** — `SKMemoryLayout`, `sk_is_c_contiguous`, `sk_is_f_contiguous`,
  `sk_force_contiguous`.
- **Distances** — `sk_squared_euclidean_distance_matrix`, `sk_euclidean_distance_matrix`,
  `sk_manhattan_distance_matrix`, `sk_cosine_distance_matrix`.
- **Sparse** — `sk_csr_dense_product`, `sk_sparse_product`.
- **Backend** — `SKMathBackend` (trait), `SKFaerBackend`, `SKMatrixMultiplyBackend`,
  `SKNdArrayLinalgBackend` (`blas-backend` only), `sk_default_math_backend`.

All structs/traits use the `SK` prefix and free functions the `sk_` prefix (PRD §3.4).

## Resolved parallelism and grain

Every dense kernel accepts the resolved plan's `parallelism` and honours it:
`1` runs the sequential `azip!`/`zip_mut_with` form with **zero dispatch overhead**;
`> 1` dispatches `par_azip!` (or a per-row/per-chunk parallel reduction) across the
rayon compute pool. Parallel results match the sequential reference within
floating-point tolerance.

| Kernel | Parallelism behaviour |
|---|---|
| `sk_elementwise_transform` | `1 → azip!`, `> 1 → par_azip!` |
| `sk_binary_combine` | `1 → zip_mut_with`, `> 1 → par_azip!` |
| `sk_axis_sum` | axis-0: per-chunk partial accumulation (no shared accumulator); axis-1: per-row |
| `sk_scale_in_place` | `1 → azip!`, `> 1 → par_azip!` |

Policy lives in `sk_resolve_execution_plan`, **not** in the kernels. Resolution
derives `parallelism = min(cores, ceil(unit / grain))` from the work-unit size
(the whole dataset in memory, one batch when streaming) capped by the core
count, and returns exactly `1` for single-core machines or sub-grain units.
Each kernel documents a **grain** — the minimum elements per work unit needed to
amortize parallel dispatch — measured by the `kernel_calibration` criterion
bench (cheap/medium/expensive closures × sizes 10¹–10⁷) and recorded beside the
kernel with machine + date provenance:

| Kernel grain constant | Value | Calibrated on |
|---|---|---|
| `SK_ELEMENTWISE_TRANSFORM_GRAIN` | `1_000_000` | 2026-09-11, i7-12700H (20 cores) |
| `SK_BINARY_COMBINE_GRAIN` | `10_000_000` | 2026-09-11, i7-12700H (20 cores) |
| `SK_AXIS_SUM_GRAIN` | `100_000` | 2026-09-11, i7-12700H (20 cores) |
| `SK_SCALE_IN_PLACE_GRAIN` | `1_000_000` | 2026-09-11, i7-12700H (20 cores) |

The same grain applies to in-memory and streaming regimes. Re-calibrating on
another machine changes only these constants and their provenance; the behaviour
contracts above are unaffected.

## Usage examples

Elementwise transform and an in-place scale (both take the resolved `parallelism`):

```rust
use ndarray::{array, Array1};
use sciencekit_math::sk_elementwise_transform;

let input = array![1.0_f64, 2.0, 3.0];
let transformed = sk_elementwise_transform(&input.view(), |x| 2.0 * x + 1.0, 1);
// transformed == [3.0, 5.0, 7.0]
```

Squared-Euclidean distance matrix between two feature sets:

```rust
use ndarray::array;
use sciencekit_math::sk_squared_euclidean_distance_matrix;

let points = array![[0.0_f64, 0.0], [3.0, 4.0]];
let distance = sk_squared_euclidean_distance_matrix(&points.view(), &points.view());
// symmetric, zero diagonal, non-negative
```

A dense matrix product through the pure-Rust default backend:

```rust
use faer::{Mat, MatRef};
use sciencekit_math::{SKFaerBackend, SKMathBackend};

let a = Mat::<f64>::from_fn(2, 2, |i, j| (i * 2 + j) as f64);
let b = Mat::<f64>::identity(2, 2);
let backend = SKFaerBackend::new();
let product = backend.gemm(a.as_ref(), b.as_ref(), 1.0);
```

These examples mirror the crate's companion `*_tests.rs` modules, which are the executable
source of truth for each function's behaviour.

## The `blas-backend` feature

By default the crate links no C/Fortran BLAS: heavy algebra runs on the pure-Rust `faer`
backend. Enabling the `blas-backend` feature swaps dense algebra to `ndarray-linalg` over the
configured `blas-src` backend (OpenBLAS/BLIS/MKL/Accelerate); `faer` remains available as a
fallback. `sk_default_math_backend` selects the active backend.

## Small fixed-size matrices (`nalgebra`)

For 2×2/3×3 hot paths (e.g. PCA covariance, affine transforms) where GEMM overhead dominates,
`nalgebra 0.35` was considered as a small-matrix fallback. Per the `wave-plan-foundation`
decision (ADR #37) it is **deferred, not adopted**: `sciencekit_math` exposes no small-matrix
hot paths yet, and `faer`'s small-matrix performance is assumed adequate. The decision is
retargeted to **W5.4 (PCA)**, where the 2×2/3×3 covariance path actually lands.

## Safety

This is the performance crate: `unsafe` is confined to the SIMD (`wide`) and matrix
multiply kernels and never appears in the higher-order or layout layers.