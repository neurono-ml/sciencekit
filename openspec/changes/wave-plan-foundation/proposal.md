## Why

The PRD (`docs/PRD.md` §12) defines seven implementation phases (0–7) but does not
specify the sub-phase granularity ("waves"), the inter-wave dependency ordering, or the
foundational technical decisions (BLAS backend, GPU arrival order, quantile sketch,
sparse SVD, nested rayon management) that gate every downstream algorithm change. Without
this layer, the first algorithm changes (W0.1 workspace, W0.3 math kernel) would open
without a shared reference for which crates to pin, which spike benchmarks to run first,
or how to order the GPU backends — inviting inconsistent, ad-hoc choices across PRs.

This change establishes the **wave plan** and the **spike decisions** as the project's
planning source of truth, so every subsequent algorithm/accelerator/docs change references
a single, reviewed plan rather than re-deriving structure per PR.

## What Changes

- **Wave structure**: decompose PRD §12's seven phases into ~35 algorithm changes + ~12
  accelerator waves + ~8 doc waves (≈55 changes total), organized so that dependencies
  are respected naturally and no throwaway/placeholder implementations are needed.
- **Spike decisions locked** (5 research-driven choices, with the evidence summarized in
  `design.md`):
  1. **Pure-Rust BLAS default**: `oxiblas 0.2.2` vs `faer 0.24.4` — deferred to a
     head-to-head spike benchmark (8 micro-benchmarks, GEMM/SVD/QR/Cholesky/eigh/solve)
     under `temporary/2026-08-26/blas-spike/` **before W2.1** (LinearRegression needs
     SVD). `matrixmultiply` is too narrow (GEMM-only); `ndarray-linalg`+`blas-src`
     remains the opt-in `blas-backend` path.
  2. **GPU backend order**: **OpenCL 3.0 first (ICD-agnostic / vendor-transparent) → Metal
     after OpenCL; CUDA and ROCm only on request.** Focus is **CPU + OpenCL**. The OpenCL
     backend uses `opencl3` + the OpenCL ICD loader, transparently using whatever OpenCL
     driver the user has installed (NVIDIA official `opencl-nvidia`, AMD, Intel, Rusticl,
     etc.) so NVIDIA/AMD/Intel/Ascend users work without forcing a vendor stack. Rusticl is
     **not** used on NVIDIA (it would force Nouveau, which conflicts with the official
     NVIDIA driver). Heavy linear algebra stays on CPU (pure-Rust BLAS); OpenCL covers
     GPU-tractable kernels (pairwise distance, GEMM, tree predict, elementwise). The "OpenCL-on-NVIDIA-is-deprecated"
     premise is disproven — OpenCL remains a viable cross-vendor path. CUDA/ROCm/Metal are
     not in the active roadmap; they are implemented only when explicitly requested.
  3. **Online quantile sketch for `SKRobustScaler`**: `tdigest 1.0.0` (Apache-2.0, mature,
    serde). Enables `partial_fit` for RobustScaler — a capability scikit-learn lacks
    (`np.nanpercentile` requires the full column).
  4. **Pure-Rust sparse SVD for `SKTruncatedSVD`**: depends on the BLAS spike outcome —
     `faer-sparse` + `rsvd-faer` if `faer` wins, `oxiblas-sparse::RandomizedSvd` + a
     `sprs`↔`oxiblas-sparse` CSR adapter if `oxiblas` wins. `single-svdlib` (sprs-native
     IRLBA, the gold standard) is attractive but MSRV 1.88 > our 1.85 blocks it unless
     we bump MSRV. `arpack-sys` as opt-in `arpack-backend` for sklearn-exact parity.
  5. **Nested rayon thread management**: rayon owns all parallelism (global pool,
     configurable via `ThreadPoolBuilder`); BLAS runs single-threaded inside rayon scopes
     (`MATMUL_NUM_THREADS=1` for `matrixmultiply`; disable `parallel` feature on
     `oxiblas`/`faer` in nested contexts). Nested `par_iter` is safe via work-stealing —
     **no semaphore needed** (corrects an earlier "Semaphore to cap threads" idea). This
     is cleaner than scikit-learn's OpenMP + `threadpoolctl` hack.
- **Dependency-inversion removed**: `SKKNNImputer` moves from Wave 1 to Wave 3 (after
  `SKKDTree`/`SKBallTree`), built directly on the tree — no brute-force placeholder, no
  `SKNearestNeighborsSearcher` trait hack. Wave 1 impute becomes `SimpleImputer` only.
- **Academic anchors catalogued** (in `design.md`): canonical paper per algorithm with
  stable URLs (arXiv/DOI/AAAI proceedings), corrected from the initial research (3 wrong
  arXiv IDs, 2 wrong DOIs, 6 wrong GitHub org names).
- **Rust ecosystem lessons catalogued** (in `design.md`): what to steal from `linfa`
  (per-algorithm sub-crate split), `burn` (`Backend` trait → `SKExecutionMode`),
  `candle`/`cudarc` (pure-Rust CUDA path), `sprs` (CSR/CSC iterator design), `ndarray`
  (`ArrayView`/`CowArray` + `azip!`/`par_azip!`). Anti-pattern: `rustlearn` (broad but
  shallow, no perf story → ages into irrelevance).

## Capabilities

### New Capabilities

This is a planning/meta change — no product behavior spec is introduced. `skip_specs: true`
is set in `.openspec.yaml`. The wave plan and spike decisions are captured in `design.md`
(architectural decisions) and `tasks.md` (wave-by-wave breakdown), which serve as the
reference contract for all subsequent algorithm/accelerator/docs changes.

### Modified Capabilities

None — this is the first change in the repository; no prior specs exist.

## Impact

- **Planning artifacts**: `design.md` (5 spike decisions in detail + academic anchors +
  Rust ecosystem lessons) and `tasks.md` (W0–W7 wave breakdown with dependencies,
  anchors, and Rust improvements per change) become the canonical reference for all
  downstream changes.
- **Downstream changes unblocked**: W0.1 (`bootstrap-workspace`), W0.2
  (`common-core-foundation`), W0.3 (`math-kernel-foundation` — carrying the BLAS spike
  sub-task), W0.4 (`execution-decision-and-observability`) can now be proposed against a
  shared plan.
- **Dependencies pinned** (reconciled in `design.md`): `ndarray ≥0.17`, `thiserror ^2`,
  `sprs` (pins `ndarray <0.18`), `ndarray-linalg 0.18` (needs `ndarray ^0.17.1` +
  `thiserror ^2`), `wide 1.6`, `memmap2 0.9.11`, `tikv-jemallocator 0.5`, `mimalloc 0.1`,
  `approx 0.5.1`, `tracing-opentelemetry 0.33`, `cudarc 0.19`, `ocl`/`opencl3`,
  `cubecl-hip-sys`, `tdigest 1.0.0`, and the BLAS-default candidate (`oxiblas 0.2.2` or
  `faer 0.24.4`, pending spike).
- **No code, no runtime change**: this change touches only `openspec/changes/wave-plan-foundation/`
  artifacts. Implementation of W0.1 onward happens in separate changes on their own
  worktree branches, per the AGENTS.md workflow.
