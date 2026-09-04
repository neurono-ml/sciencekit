## Context

The repository is freshly initialized: no Cargo workspace, no algorithm crates, no OpenSpec
changes prior to this one. The PRD (`docs/PRD.md` §12) lists seven phases (0–7) but leaves
three things undecided that every downstream change depends on:

1. **Sub-phase granularity and dependency ordering** — which changes can be developed in
   parallel, which must be sequential, and where the forward-dependencies are
   (e.g., `SKKNNImputer` needs `SKKDTree`).
2. **The pure-Rust BLAS/LAPACK default** — PRD §6.2 says "default: pure Rust" but does not
   name the crate. `matrixmultiply` (ndarray's fallback) is GEMM-only; `ndarray-linalg`
   is FFI to Fortran (fails the pure-Rust requirement). The candidates (`oxiblas`,
   `faer`) need empirical comparison before lock.
3. **The GPU backend arrival order** — PRD §6.4 lists OpenCL/CUDA/ROCm together as
   "Start" without ordering; the empirical question is which has the mature dynamic-loading
   path and companion libraries (cuBLAS/cuSOLVER/cuSPARSE).

A prior research stream (three parallel subagents: scikit-learn internals, academic
references + Rust ecosystem, Rust numerical foundations) produced the evidence base
summarised in this document. Two follow-up spike subagents (BLAS path, GPU backends) plus
targeted crate searches resolved the six decisions below. The decisions are
**research-locked** where the evidence is conclusive (spikes 2, 3, 5) and
**spike-resolved** where a spike had to run first (spikes 1, 4 — the latter a downstream
consequence of the former). Spike 1 ran at `temporary/2026-08-31/blas-spike/` and resolved
to **`faer`** (`oxiblas` fails to compile on MSRV 1.85); spike 4 follows it.

See `proposal.md` — Why for the motivation.

## Goals / Non-Goals

**Goals:**
- Lock the five foundational technical decisions (BLAS, GPU order, quantile sketch, sparse
  SVD, nested rayon) so downstream algorithm changes reference a single, reviewed choice.
- Catalogue the canonical academic reference per algorithm with stable URLs (arXiv/DOI/AAAI
  proceedings), corrected from the initial research (3 wrong arXiv IDs, 2 wrong DOIs, 6
  wrong GitHub org names).
- Catalogue what to learn from each existing Rust ML crate (`linfa`, `burn`, `candle`,
  `cudarc`, `sprs`, `ndarray`, `smartcore`, `rustlearn` anti-pattern).
- Reconcile the workspace dependency pins (`ndarray`, `sprs`, `ndarray-linalg`,
  `thiserror`) so W0.1's `Cargo.toml` resolves from day one.

**Non-Goals:**
- Implement any algorithm, crate, or runtime code. This change touches only
  `openspec/changes/wave-plan-foundation/` artifacts.
- Replace the PRD. The PRD remains the product source of truth; this design is a
  planning/architectural layer beneath it.
- Run the full 8-benchmark criterion suite for the BLAS default (spike 1) or the
  sparse-SVD crate (spike 4, which depends on 1) as an output of *this* change — that suite
  moved to the downstream `math-kernel-foundation` change. This change only ran the decisive
  spike at `temporary/2026-08-31/blas-spike/` that disqualified `oxiblas` on MSRV 1.85 and
  locked `faer`.

## Decisions

### Decision 1 — Pure-Rust BLAS/LAPACK default: `faer` (resolved by spike — `oxiblas` fails MSRV)

**Rationale.** The default build must be pure-Rust (PRD §6.2). Two candidates were
checked for MSRV-1.85 viability under `temporary/2026-08-31/blas-spike/` (an `oxiblas`
compile check against Rust 1.85 plus a `faer` build-and-GEMM smoke test — not a full
head-to-head benchmark, which moved downstream):

| Crate | License | MSRV (declared) | MSRV (actual) | Compiles on Rust 1.85? |
|---|---|---|---|---|
| `oxiblas 0.2.2` | Apache-2.0 | 1.85 | **>1.85 in practice** | ❌ **fails** (191 errors: AVX-512 `_mm512_loadu_si512`/`_mm512_storeu_si512` incompatible with Rust 1.85's stdarch) |
| `faer 0.24.4` | MIT | 1.84 | 1.84 | ✅ builds + GEMM correctness verified |

**Spike verdict (2026-08-31):** `oxiblas` is **disqualified** — despite declaring MSRV
1.85, it does not compile on Rust 1.85 (its hand-written AVX-512 intrinsics rely on a
newer stdarch). Choosing it would violate the project's MSRV gate. `faer` is the **only
viable pure-Rust candidate** meeting MSRV 1.85, and it provides the full LAPACK surface
(QR, Cholesky, LU, SVD, eig, solve) plus a `sparse-linalg` feature and the `rsvd-faer`
companion.

`matrixmultiply 0.3.11` remains the fallback GEMM kernel (used by `ndarray`'s `dot` when
`blas` is off) and `ndarray-linalg 0.18` + `blas-src` is the opt-in `blas-backend` path
(FFI to OpenBLAS/BLIS/MKL/Accelerate), not the default.

**Alternatives considered.** `nalgebra` (small-matrix dense LA, already an optional
`oxiblas` dep) — candidate for small fixed-size ops (2×2/3×3 covariance in PCA) where GEMM
overhead dominates, not a general default. `realfft`/`rustfft` — adjacent (FFT), not BLAS.

**Decision.** **`faer 0.24.4` is the pure-Rust BLAS/LAPACK default** (locked by the
2026-08-31 spike — `oxiblas` is disqualified because it does not compile on MSRV 1.85).
The escape clause below is moot for `oxiblas` (it cannot build), so no downgrade path is
needed; the accuracy gate is enforced in `sciencekit_math` against the
`ndarray-linalg`+OpenBLAS reference. The original benchmark plan (8 micro-benchmarks:
GEMM 1024² / 5000×128; full SVD 500²; truncated SVD 500×5000 k=50; economy QR 2000×500;
Cholesky 1000²; symmetric eigh 500²; linear solve 1000²×1000 RHS) and its lock criterion
(≥60% of OpenBLAS throughput on every bench, ≥80% on GEMM/Cholesky/QR, ≤1e-9 f64 /
≤1e-5 f32 relative error) moved to the downstream `math-kernel-foundation` change as a
validation gate.

### Decision 2 — GPU backend order: OpenCL 3.0 first, ICD-agnostic (vendor-transparent) → Metal; CUDA/ROCm only on request (operator-locked)

**Rationale.** The operator prefers **OpenCL 3.0 first** and wants a **CPU + OpenCL** focus,
with an OpenCL implementation that **transparently tolerates NVIDIA, AMD, Intel, Ascend and
other users** — i.e., it must work with whichever OpenCL driver the user has installed,
without forcing a particular vendor stack. CUDA and ROCm are postponed; **Metal** follows
*after* OpenCL; anything beyond OpenCL is implemented **only when requested**.

**Why not Rusticl-on-NVIDIA (verified).** The operator's concern is confirmed: **Rusticl on
NVIDIA runs exclusively through the Nouveau Gallium driver and cannot be used with the
official NVIDIA driver.**
- Rusticl is a Mesa OpenCL implementation over Gallium drivers; for NVIDIA the Gallium
  driver is `nouveau`. Rusticl provides OpenCL 3.0 on NVIDIA **only via nouveau**.
- The official NVIDIA driver **blacklists nouveau** — to run Rusticl on NVIDIA you must
  remove `nvidia-utils`/`opencl-nvidia` (verified on Arch Linux forum). They cannot coexist.
- Where Rusticl does run on NVIDIA (via nouveau), kernel latency is much higher and compute
  performance is mixed vs the official NVIDIA OpenCL driver (Phoronix clpeak).
- **Conclusion:** Rusticl is a good driver for **AMD / Intel / CPU** (open-source,
  Khronos-certified OpenCL 3.0/3.1, often outperforms ROCm's OpenCL on AMD) but it is
  **not** the driver for NVIDIA users, who normally use NVIDIA's own `opencl-nvidia` ICD.

**Design: ICD-agnostic OpenCL backend (transparent).** `sciencekit`'s `OpenClBackend` uses
**`opencl3` + the OpenCL ICD loader** (`libOpenCL.so`). The ICD loader discovers and
dispatches to whichever vendor ICDs are registered on the system — it does **not** bundle
or require Rusticl. This makes the backend **transparent to the user's hardware**:

| User's platform | OpenCL ICD auto-discovered by the loader |
|---|---|
| NVIDIA + official driver | NVIDIA's own `opencl-nvidia` ICD |
| NVIDIA + nouveau | Rusticl (via nouveau) |
| AMD | Rusticl (`radeonsi`) or ROCm OpenCL ICD |
| Intel | Intel Compute Runtime (`neo`) or Rusticl (`iris`) |
| Ascend / other | the vendor's OpenCL ICD if present, else CPU fallback |

The backend enumerates available platforms/devices (`clGetPlatformIDs`/`clGetDeviceIDs`)
and transparently selects a usable device, preferring discrete GPUs and falling back to the
CPU backend when no OpenCL device is available. Rusticl is **one possible driver**, not a
dependency — the design never forces it.

This still neutralises the "companion-library gap" that motivated CUDA-first: heavy
BLAS/SVD/eigen stays on the CPU backend (pure-Rust BLAS, Decision 1); OpenCL handles the
GPU-tractable kernels — pairwise distance, GEMM, tree predict, elementwise — written in
OpenCL C / SPIR-V, per the PRD's "no skipped steps" evolution.

| Backend | Crate | Discovery | Companion libs | Vendor transparency | Status |
|---|---|---|---|---|---|
| **OpenCL 3.0 (FIRST)** | `opencl3 0.12.3` | ✅ ICD loader (auto) | none (write own kernels) | ✅✅ NVIDIA/AMD/Intel/others | primary |
| **Metal (AFTER OpenCL)** | TBD (Metal FFI) | system framework | Metal Performance Shaders | Apple-only | next, on request |
| CUDA (on request) | `cudarc 0.19.9` | `libloading` | full suite | NVIDIA-only | on request |
| ROCm (on request) | `cubecl-hip-sys` | `hipconfig` | none | AMD-only (Linux) | on request |

**Decision.** **OpenCL 3.0 (`opencl3`, not `ocl` — edition 2024, OpenCL 3.0) is the first
and primary GPU backend, implemented as an ICD-agnostic layer that transparently uses the
user's installed OpenCL driver (NVIDIA official, AMD, Intel, Rusticl, etc.). Metal follows
after OpenCL. CUDA and ROCm are not in the active roadmap — implemented only when
explicitly requested.** CPU remains the always-present default backend.
`SKExecutionMode::Automatic` routes via a cached 1ms micro-kernel benchmark per (backend,
dtype, shape-class) — mirrors `cubecl`'s autotune-cache + `burn::Backend`. The
`sciencekit_gpu` crate ships `SKComputeBackend` with `CpuBackend` always present and
`OpenClBackend` behind the `gpu-opencl` feature flag.

**Alignment with PRD §6.4.** PRD §6.4 lists OpenCL/CUDA/ROCm together as "Start" and
OpenCL first in that list. This decision follows the PRD's **listed order** (OpenCL first)
but **defers CUDA/ROCm out of the active roadmap** and adds Metal as a future backend —
a scope reduction, not a reorder. Document as an ADR on the matching GitHub issue (per
AGENTS.md workflow) noting the Rusticl rationale and the on-demand policy for
CUDA/ROCm/Metal.

**Watch item.** `cubecl` (2333★, Burn-proven, edition 2024, unifies CUDA/ROCm/Metal/Vulkan/
WebGPU/CPU) remains the strongest long-term candidate to **be** `SKComputeBackend`'s
portable implementation, and its **OpenCL runtime** could eventually absorb the custom
`OpenClBackend` kernels. It is alpha today (breaking changes between minors). Track it;
adopt only when it stabilises — the `OpenClBackend` written now stays behind the trait so
a `CubeclBackend` can replace it without touching algorithms.

### Decision 3 — Online quantile sketch for `SKRobustScaler`: `tdigest 1.0.0` (research-locked)

**Rationale.** scikit-learn's `RobustScaler` has no `partial_fit` because
`np.nanpercentile` requires the full column. A streaming quantile sketch per feature under
`par_azip!` enables `partial_fit` — a capability sklearn lacks.

| Crate | License | MSRV | Maturity |
|---|---|---|---|
| **`tdigest 1.0.0`** | Apache-2.0 | 1.62 | ✅ mature (1.0.0), serde |
| `kll-rs 0.1.4` | Apache-2.0 | unknown | young (0.1.4) |
| `sketch_oxide 0.1.6` | MIT/Apache | unknown | young, 2025 SOTA (DDSketch) |

RobustScaler needs Q1 (25th), median (50th), Q3 (75th) — not extreme tails, so t-digest's
uniform accuracy suffices. KLL is theoretically optimal for space (O(1/ε log 1/ε)) but the
Rust crate is too young for a foundational library. `tdigest 1.0.0` is mature, Apache-2.0,
serde-serializable, MSRV 1.62 (well below our 1.85).

**Decision.** **`tdigest 1.0.0`** as the default. Document KLL as a future swap if memory
becomes tight at very high feature counts.

### Decision 4 — Pure-Rust sparse SVD for `SKTruncatedSVD`: `faer-sparse` + `rsvd-faer` (resolved via Decision 1)

**Rationale.** `SKTruncatedSVD` (LSA on TF-IDF) must not densify the input. ARPACK (what
scikit-learn wraps via `scipy.sparse.linalg.svds`) is opaque C/Fortran — against the
PRD default-build pure-Rust requirement.

| Option | Algorithm | sprs-native? | MSRV | Verdict |
|---|---|---|---|---|
| `faer-sparse 0.17.1` + `rsvd-faer 0.1.0` | sparse SVD/eig + randomized | via faer sparse | 1.84 | **✅ chosen (Decision 1 → faer)** |
| `single-svdlib 2.0.0` | IRLBA (R's irlba gold standard) + randomized | ✅ built for sprs | **1.88 ❌ (blocks)** | Attractive but MSRV-blocked |
| `oxiblas-sparse::RandomizedSvd` | randomized + Lanczos + IRAM | own formats (adapter needed) | 1.85 | moot — `oxiblas` fails MSRV |
| Hand-rolled Lanczos | simple recurrence (~80 LOC) | ✅ direct on sprs | n/a | Fallback / educational |
| `arpack-sys 0.0.2` | ARPACK-NG (FFI, matches sklearn) | via scipy-style | n/a | Opt-in `arpack-backend` only |

**Decision.** **`faer-sparse` + `rsvd-faer`** (Decision 1 resolved to `faer`). Hand-rolled
Lanczos as the fallback (the recurrence is simple, ~80 LOC, well-specified).
`single-svdlib` is the gold-standard algorithm (IRLBA) but MSRV 1.88 > 1.85 blocks it;
revisit if MSRV is bumped (see Open Questions). `arpack-sys` as opt-in
`arpack-backend` feature for sklearn-exact parity (research/forensics use case).

### Decision 5 — Nested rayon thread management: rayon owns all parallelism (research-locked)

**Rationale.** An earlier idea ("Semaphore to cap total threads when nesting trees +
features per node") was **wrong**. rayon's design already caps at pool size via
work-stealing: nested `par_iter` inside `par_iter` uses the same fixed pool, work-stealing
balances the ~100 trees × ~30 features/tasks against the N pool threads — total active
threads = N, not 100×30.

The real oversubscription risk is when BLAS (which has its own threadpool) is called inside
rayon's `par_iter`. scikit-learn's hack is `threadpoolctl` capping BLAS to 1 thread inside
OpenMP `prange`. The Rust equivalent is cleaner:

```
rayon global pool (N threads, configurable via ThreadPoolBuilder::new().num_threads(N).build_global())
  ├── outer par_iter (e.g., 100 trees in SKRandomForest)
  │     └── inner par_iter (e.g., 30 features per node split in SKDecisionTree)
  │           └── work-stealing: total active threads = N
  └── BLAS call inside rayon scope
        └── BLAS single-threaded:
              → matrixmultiply: set MATMUL_NUM_THREADS=1 (env, process-wide)
              → oxiblas: disable `parallel` feature in nested context
              → faer: gate `rayon` feature off in nested context (or accept same-pool reuse)
```

**Decision.** **rayon owns ALL parallelism; BLAS runs single-threaded inside rayon
scopes.** No semaphore, no `threadpoolctl`-equivalent hack — cleaner than scikit-learn's
OpenMP approach. For core pinning: `core_affinity` crate +
`ThreadPoolBuilder::start_handler(move |_| set_affinity(...))`.

**Alternatives considered.** (a) A `Semaphore`-based cap — rejected, rayon already caps;
(b) `threadpoolctl`-style runtime BLAS-thread setter — rejected, env-var
(`MATMUL_NUM_THREADS=1`) + feature-gate (`parallel` off) achieve the same with no runtime
indirection; (c) separate nested rayon pools via `ThreadPool::scope` — rejected, the global
pool with work-stealing is simpler and equally efficient.

### Decision 6 — KNNImputer placement: Wave 3, no brute-force placeholder (research-locked)

**Rationale.** Originally `SKKNNImputer` was placed in Wave 1 with a brute-force
implementation and a `SKNearestNeighborsSearcher` trait to defer the KDTree. The operator
clarified: "We have no need to use brute force first for KNNImputer and others. We may
implement directly the best algorithms."

**Decision.** **`SKKNNImputer` moves from Wave 1 to Wave 3 (W3.4), built directly on
`SKKDTree`/`SKBallTree` (W3.2).** The `SKNearestNeighborsSearcher` trait hack is removed.
Wave 1 impute becomes `SimpleImputer` only. This principle ("implement directly the best
algorithm, no throwaway placeholders") applies to all waves — the 4-stage evolution
(naive → tests → perf → streaming) happens WITHIN a change's tasks, not as throwaway
placeholder changes.

## Academic anchors (canonical reference per algorithm)

All URLs verified stable (arXiv/DOI/AAAI proceedings) unless marked "(verify)". Three
arXiv IDs from the initial research were wrong (resolved to unrelated papers); use the
DOIs/proceedings below instead.

**Free online reference texts (the shelf):**
- ESL — https://hastie.su.domains/ElemStatLearn/ (Hastie/Tibshirani/Friedman; §3 regression, §3.4 standardization, §9 shrinkage, §12 SVM, §13 NN, §14 trees, §15 RF, §16 ensembles, §10 boosting, §14.7 k-means, §14.5 PCA)
- MML — https://mml-book.github.io/ (Deisenroth/Faisal/Ong; Ch. 2 linear algebra, Ch. 4 SVD/PCA, Ch. 9 regression, Ch. 10 PCA, Ch. 12 SVM)
- Golub & Van Loan *Matrix Computations* (4th ed., ISBN 978-1421407944) — library reference for SVD/QR/LU/Cholesky numerics.
- Rust Performance Book — https://nnethercote.github.io/perf-book/

**Per-algorithm canonical papers:**

| Wave | Algorithm | Paper | Stable URL |
|---|---|---|---|
| W1.1 | StandardScaler | ESL §3.4 | https://hastie.su.domains/ElemStatLearn/ |
| W1.2 | RobustScaler + t-digest | ESL §2.4.2; Dunning t-digest | https://hastie.su.domains/ElemStatLearn/ |
| W1.3 | OneHotEncoder/PolynomialFeatures | Kuhn & Johnson *Feature Engineering* | http://www.feat.engineering/ |
| W2.1 | LinearRegression OLS | ESL §3; Golub & Van Loan §2.4, §8.6 | https://hastie.su.domains/ElemStatLearn/ |
| W2.2 | Ridge | Hoerl & Kennard 1970 | https://doi.org/10.1080/00401706.1970.10488634 |
| W2.3 | Lasso/ElasticNet (coord. descent) | Tibshirani 1996; Zou & Hastie 2005; Friedman et al. 2010 (glmnet) | https://doi.org/10.1111/j.2517-6161.1996.tb02080.x ; https://doi.org/10.1111/j.1467-9868.2005.00503.x ; https://doi.org/10.18637/jss.v033.i01 |
| W2.4 | LogisticRegression (SAGA/LIBLINEAR) | Defazio et al. 2014 SAGA; Fan et al. 2008 LIBLINEAR | https://arxiv.org/abs/1407.0202 ; https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf |
| W2.5 | SGDClassifier/Regressor | Bottou 2010; Pegasos (Shalev-Shwartz et al. 2007) | https://doi.org/10.1007/978-3-642-15849-9_28 ; https://doi.org/10.1145/1273496.1273598 |
| W3.2 | KDTree/BallTree | Bentley 1975; Omohundro 1989 | https://doi.org/10.1145/361002.361007 ; ICSI TR-89-063 |
| W3.3 | k-NN | Cover & Hart 1967 | https://doi.org/10.1109/TIT.1967.1053964 |
| W3.5 | SVC/SVR (SMO/LIBSVM) | Platt 1998 SMO; Chang & Lin 2011 LIBSVM | https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/ ; https://doi.org/10.1145/1961189.1961199 |
| W4.1 | DecisionTree CART | Breiman et al. 1984 (book); Quinlan 1986 (entropy) | ISBN 978-0412048418 ; https://doi.org/10.1007/BF00116251 |
| W4.2 | RandomForest | Breiman 2001 | https://doi.org/10.1023/A:1010933404324 |
| W4.3 | Bagging | Breiman 1996 | https://doi.org/10.1007/BF00058655 |
| W4.4 | GradientBoosting | Friedman 2001 | https://doi.org/10.1214/aos/1013203451 |
| W4.5 | HistGradientBoosting | Ke et al. 2017 LightGBM (NeurIPS, no arXiv) | https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf |
| W5.1 | KMeans (Lloyd/Elkan) | Lloyd 1982; k-means++ 2007; Elkan 2003 | https://doi.org/10.1109/TIT.1982.1056489 ; https://doi.org/10.5555/1283383.1283494 ; https://cdn.aaai.org/ICML/2003/ICML03-022.pdf |
| W5.2 | MiniBatchKMeans | Sculley 2010 | https://doi.org/10.1145/1772690.1772862 |
| W5.3 | DBSCAN | Ester et al. 1996 | https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf |
| W5.4 | PCA + randomized SVD | Halko, Martinsson, Tropp 2011 | https://arxiv.org/abs/0909.4061 |
| W5.7 | IsolationForest | Liu, Ting, Zhou 2008 | https://doi.org/10.1109/ICDM.2008.17 |

**Corrections log (vs. the initial research brief):**
1. arXiv `1010.0715` (coordinate descent) → wrong paper; **no arXiv version exists**. Use JSS DOI `10.18637/jss.v033.i01`.
2. arXiv `0709.0508` (Pegasos) → wrong paper; **no arXiv version exists**. Use ACM DOI `10.1145/1273496.1273598`.
3. arXiv `1706.05114` (LightGBM) → wrong paper; **no arXiv version exists**. Use NeurIPS 2017 proceedings PDF.
4. Bagging DOI corrected: `10.1007/BF00058655` (not `10.1023/A:1007551413461`).
5. KD-tree DOI corrected: `10.1145/361002.361007` (not `10.1145/356786.356787`).
6. Crate owners corrected: smartcore → `smartcorelib`; burn → `tracel-ai`; cudarc → `chelsea0x3b`; rustlearn → `maciejkula`; sprs → `sparsemat`; faer → `sarah-ek` (GitHub mirror of `faer-rs`).

## Rust ecosystem lessons (what to steal, what to avoid)

| Crate | What to steal | Gap vs sklearn (our opportunity) |
|---|---|---|
| `linfa` (rust-ml/linfa) | per-algorithm sub-crate split + `DatasetBase` view + `Fit`/`Predict` traits | no HGBDT, no SGD, no pipelines, no OOC, no GPU, no SIMD |
| `smartcore` (smartcorelib/smartcore) | clean trait-based estimator API, dense/sparse generic dispatch | no HGBDT, no coordinate-descent Lasso, no SGD, weak sparse |
| `burn` (tracel-ai/burn) | **`Backend` trait** (one code path, many backends) → `SKExecutionMode` | pure DL, no classical ML |
| `candle` (huggingface/candle) | minimal pure-Rust CPU+CUDA path via `cudarc` (no build step) | inference-focused |
| `cudarc` (chelsea0x3b/cudarc) | safe, zero-cost CUDA binding layer, feature-gated per-library, dynamic loading default | bindings only |
| `sprs` (sparsemat/sprs) | `CsMat`/`CsVec` CSR/CSC iterator design + generic index type | MSRV 1.64 (we need 1.85) |
| `ndarray` (rust-ndarray/ndarray) | `ArrayView`/`CowArray` + `azip!`/`par_azip!` — our zero-copy foundation | no SIMD helpers (pair with `wide`/`std::simd`) |
| `rustlearn` (maciejkula/rustlearn) | **ANTI-PATTERN**: broad-but-shallow port with no perf story (no SIMD/rayon/zero-copy) → ages into irrelevance | (the warning; our 4-stage ladder is the counter) |

## Workspace dependency pins (reconciled for W0.1)

W0.1's `Cargo.toml` must resolve these interdependencies from day one:
- `ndarray ≥0.17` (satisfies `sprs <0.18` and `ndarray-linalg 0.18`'s `ndarray ^0.17.1`)
- `thiserror ^2` (matches `ndarray-linalg 0.18`)
- `ndarray` `rayon` feature on → unlocks `par_azip!`
- `wide 1.6`, `matrixmultiply 0.3.11`, `memmap2 0.9.11`, `tikv-jemallocator 0.5`,
  `mimalloc 0.1`, `approx 0.5.1`, `tracing-opentelemetry 0.33`, `cudarc 0.19`,
  `opencl3`, `cubecl-hip-sys`, `tdigest 1.0.0`
- BLAS-default: `faer 0.24.4` (locked, Decision 1); `oxiblas 0.2.2` disqualified (fails MSRV 1.85)
- Build flags: `-C target-cpu=native` (enables target-specific `std::arch` intrinsics and auto-vectorization)

## Risks / Trade-offs

- **[`oxiblas` does not compile on MSRV 1.85]** (0.2.x, hand-written AVX-512 intrinsics, 191 build errors) → Resolved: `oxiblas` is disqualified and `faer` is the default. Its accuracy escape clause (Schur/GeneralEvd) is moot since it cannot build; the accuracy gate vs `ndarray-linalg`+OpenBLAS runs in `sciencekit_math`.
- **[`faer` may not match toolchain]** (MSRV 1.84, edition unverified) → Mitigation: verify edition in the spike; if edition <2024, either accept the mismatch (edition is per-crate, not workspace) or contribute upstream.
- **[`oxiblas`/`faer` does not interop with `sprs`]** (PRD §3 mandates `sprs`) → Mitigation: build a thin `sciencekit_sparse_blas` adapter (CSR↔CSR, zero-copy feasible). Verify layout equivalence in the spike.
- **[Single-maintainer bus factor on `oxiblas`]** (KitaSan / `cool-japan`, one publisher) → Moot — `oxiblas` is disqualified on MSRV; `faer` is the default. (Kept to record why the adapter-stays-thin principle applies to any future BLAS swap.)
- **[OpenCL lacks cuBLAS/cuSOLVER equivalents]** (SVD/QR/eigen not available on the OpenCL backend) → Mitigation: **accepted design** — heavy linear algebra stays on the CPU backend (pure-Rust BLAS, Decision 1); the OpenCL backend covers GPU-tractable kernels (pairwise distance, GEMM, tree predict, elementwise) written in OpenCL C / SPIR-V. `Automatic` routing delegates anything not implemented on OpenCL to CPU. This is consistent with the focus on CPU + OpenCL.
- **[Writing and validating OpenCL kernels is non-trivial]** (GEMM, distance, reduction kernels need care to avoid numerical drift vs CPU) → Mitigation: TDD with CPU reference outputs as the oracle (per PRD §8.7 acceptance); start with the simplest kernels (elementwise, pairwise distance) and evolve; reuse `Rusticl`'s SPIR-V path (via `libclc`) rather than hand-rolled OpenCL C where possible.
- **[Rusticl not available / NVIDIA-users on the official driver]** (Rusticl runs on NVIDIA only via nouveau, which conflicts with the official NVIDIA driver) → Mitigation: the backend is **ICD-agnostic** — it uses `opencl3` + the OpenCL ICD loader, so it transparently uses NVIDIA's `opencl-nvidia` ICD, AMD/Intel/Rusticl, or any registered ICD; Rusticl is never required. Document the required environment in `sciencekit_gpu`; the `gpu-opencl` feature degrades gracefully to CPU when no OpenCL device is present; CI uses a CPU-only fallback.
- **[ROCm bindings are raw bindgen unsafe]** (`cubecl-hip-sys`) → Deferred (on request only): if/when requested, layer a thin `RocmBackend` safe wrapper or adopt `cubecl`'s ROCm runtime; never expose raw FFI in `sciencekit_gpu` public API.
- **[`cubecl` is alpha]** (breaking changes between minors) → Track it; a `CubeclBackend` can later absorb the custom OpenCL kernels. Not a v1 dependency.
- **[CUDA deferred]** — NVIDIA users lose GPU accel until requested → Mitigation: the `SKComputeBackend` trait stays abstract so a future `CudaBackend` (`cudarc`) plugs in without touching algorithms; explicit ADR records the on-demand policy.
- **[Host↔device transfer cost can dominate]** for small/medium arrays, erasing GPU gains → Mitigation: mandate the GPU spike benchmarks before locking the OpenCL backend; require transfer amortization (batched/streamed calls) in acceptance criteria.
- **[`single-svdlib` MSRV 1.88 > 1.85]** blocks the gold-standard IRLBA → Mitigation: revisit if MSRV is bumped (see Open Questions); otherwise hand-roll Lanczos or use the BLAS-default's sparse SVD.
- **[`rsvd` (ekg) is stale]** (2023, 104 LOC) → Mitigation: do NOT depend on `rsvd`; use `oxiblas::RandomizedSvd` or `rsvd-faer` or hand-roll Halko rSVD on the BLAS-default's `Qr`+`Svd`.

## Open Questions

1. **Should we bump MSRV from 1.85 to 1.88 to unblock `single-svdlib` (sprs-native IRLBA)?**
   Rust 1.88 is recent but not bleeding-edge. `single-svdlib` is the gold-standard sparse
   SVD algorithm, built directly on `sprs`. Deferrable: the BLAS-default's sparse SVD
   (`faer-sparse` or `oxiblas-sparse`) covers TruncatedSVD; `single-svdlib` is a
   quality-of-life swap, not a blocker. Revisit at W5.5 (TruncatedSVD implementation).
2. **`cubecl` adoption timing (OpenCL runtime).** `cubecl` (alpha, Burn-proven, edition
   2024) has an OpenCL runtime that could eventually absorb the custom `OpenClBackend`
   kernels into a portable `#[cube]` IR. Deferrable: the hand-written `OpenClBackend` over
   `opencl3` (ICD-agnostic — driven by whatever ICD the user has installed) covers the v1
   surface; `cubecl` is a post-1.0 consolidation. Revisit at W7 (Interop + production).
3. **`nalgebra` role for small fixed-size matrices.** `nalgebra 0.35` (already an optional
   `oxiblas` dep) could be the pure-Rust fallback for 2×2/3×3 hot paths (PCA covariance,
   affine transforms) where GEMM overhead dominates. **RESOLVED (2026-09-04): deferred, not
   adopted.** The W0.3 trigger passed with no small-matrix assessment in the archived
   `math-kernel-foundation`; `sciencekit_math` has no small-matrix hot paths, so `nalgebra`
   would be a speculative dependency. Retargeted to **W5.4 (PCA)** — adopt `nalgebra 0.35`
   only if `faer`'s small-matrix performance is inadequate for 2×2/3×3 covariance there.
   ADR follow-up issue #37; the W5.4 spec must gate on the small-matrix micro-benchmark.
