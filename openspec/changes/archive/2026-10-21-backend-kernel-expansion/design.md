## Context

See proposal.md — Why. Current `SKMathBackend` in `crates/sciencekit_math/src/backend/`
hard-codes `f64` and leaks `faer::{Mat, MatRef}` into the trait and its result
containers. Both backends (`SKFaerBackend`, `SKNdArrayLinalgBackend`) produce and consume
`faer::Mat<f64>`. The rest of the crate is already host-centric: ndarray is the canonical
host container (`SKDataView` wraps `ArrayView2`/`CsMatView`), the execution planner
(`SKExecutionMode`, `SKExecutionContext`, `SKAccessPattern`) and streaming batches
(`SKDataBatch`, `SKLazySource`, `SKMappableSource`) already exist in `sciencekit_common`,
and `SKFloat` (sealed over `f32`/`f64`) is the established scalar bound.

## Goals / Non-Goals

**Goals:**
- A `SKMathBackend<F: SKFloat>` that is host-centric: zero-copy ndarray views in, owned
  ndarray results out, no concrete backend type (faer/ndarray-linalg) in the public
  surface.
- A kernel rich enough to serve `LinearRegression` (the first roadmap algorithm) and the
  denser algorithms that follow (PCA, Ridge, GMM, LDA/QDA).
- Keep the backend eager and pure; laziness and async live in the orchestration layer.
- Modularize `backend/` into a pure-dispatcher `mod.rs`.

**Non-Goals:**
- No GPU/device-centric design now (deferred, separate change after CPU validation per
  AGENTS.md). No associated matrix types, no `MatRef` GATs.
- No lazy kernels: the heavy operations stay eager pure functions over views; lazy
  streaming is the `SKLazySource`/batch layer and the planner.
- No async in the backend trait: sync kernels composed under rayon (compute) / Tokio
  (I/O) by the execution planner. See Decision 4.
- Streaming/incremental accumulation (e.g. `partial_fit`, incremental PCA/SVD) is the
  algorithm layer's responsibility, not the backend's.

## Decisions

**Decision 1 — Host-centric, concrete surface (no associated matrix types).**
The trait takes `ArrayView2<F>` and returns `Array2<F>`; decompositions are concrete
`SKSingularValueDecomposition<F>`, `SKQRDecomposition<F>`, `SKLUDecomposition<F>` holding
`Array2<F>`. Considered and rejected: (a) backend-native associated types
(`MatOwned`/`MatRef<'a>`) — device-ready but adds genericity/GATs for a GPU explicitly
deferred; (b) a generic matrix trait `M` — pushes the faer×ndarray×GPU unification problem
to every call site. Rationale: GPU is a separate later change; CPU out-of-core streams
batches of ndarray on the host, so host-centric is both simpler and sufficient for
zero-copy, lazy and out-of-core. A GPU backend later can add device handles behind the
same logical surface.

**Decision 2 — LU pivot is `Vec<usize>`.**
The permutation returned by LU is a host `Vec<usize>` (LAPACK `ipiv` and faer's pivot are
exactly this). Rejected the proposed `PivotInfo` associated type: since every result is
host-resident, there is no backend-specific pivot representation to abstract over.

**Decision 3 — `slogdet` returns `(sign, log_abs_det)`.**
Required to compute log-likelihoods without underflowing a tiny determinant to `0.0`
(GMM, FactorAnalysis, LDA/QDA densities). Implemented via the existing factorizations
(LU or Cholesky), never materializing `det`.

**Decision 4 — Backend stays synchronous and eager; async/lazy are external.**
Per AGENTS.md "CPU never blocks async threads (rayon for compute, Tokio for I/O)", the
backend is called from inside a rayon/`spawn_blocking` pool by the execution planner.
Making the backend async would duplicate the concurrency mechanism and break pure,
deterministic, testable kernels. `SKExecutionMode::InProcessAsynchronous` and
`OutOfCore*` are orchestration concerns.

**Decision 5 — Internal parallelism tied to the execution plan.**
`gemm` currently hard-codes `Par::Seq` in faer. The backend SHALL honour the plan's
`parallelism` (faer `Par`, or BLAS thread-level) instead, so large products scale with
the resolved plan.

**Decision 6 — `norm` covers matrix and vector, full ord set + general fallback.**
Mirrors `numpy.linalg.norm`/`scipy.linalg.norm` orders used by scikit-learn. Specialized
paths for `Frobenius`, `L2`, `L1`, `Infinity` (+ negatives, `Nuclear`); a
`General { order: F }` fallback covers arbitrary p-norms. Separate matrix vs vector norm
dispatch.

**Decision 7 — `lstsq` is a first-class backend op.**
`LinearRegression` calls LAPACK `gelsd` in scikit-learn. Exposing `lstsq` centralizes
rank-deficiency/`rcond` truncation in one place; backends delegate to the best available
(ndarray-linalg `gelsd` native; faer composes from QR-pivoted/SVD). Rejected composing it
in every algorithm from `svd`/`pinv`, which reimplements tolerance logic per caller and
is slower (full vs truncated SVD).

**Decision 8 — `svds`/`eigsh` planned, not deferred indefinitely.**
Out-of-core is essential, so truncated SVD/eigh are on the immediate roadmap (see
proposal). They are specified but their delivery is tied to the streaming milestone; the
batch loop and incremental accumulation live in the algorithm layer, not the backend.

**Decision 9 — Module restructure with pure-dispatcher `mod.rs`.**
```
backend/
  mod.rs              # declares + re-exports only
  kernel.rs           # SKMathBackend<F> trait + SKNormKind
  decompositions.rs   # SKSVD / SKQR / SKLU (concrete, generic over F)
  faer_backend.rs     # SKFaerBackend + SKMatrixMultiplyBackend
  ndarray_backend.rs  # SKNdArrayLinalgBackend (feature-gated)
  dispatch.rs         # sk_default_math_backend()
```

## Risks / Trade-offs

- [GPU deferred but host-centric] → A future GPU backend copies VRAM→host at each result
  boundary; acceptable for v1, revisited when GPU is planned (Decision 1).
- [`faer` needs a copy for non-contiguous/strided input] → the public boundary stays
  zero-copy (views); a strided-to-contiguous copy is internal to the backend when the
  input is not C-contiguous (`sk_force_contiguous` is available in `layout/`).
- [Composing `lstsq` in faer may be slower than native `gelsd`] → the BLAS backend uses
  native `gelsd`; correctness first, per-architecture optimization later.
- [Breaking public API] → `blas-interface` consumers are internal so far; the break is
  absorbed before algorithms depend on it.
- [`module-hygiene` overlap] → this change owns `backend/`; the companion
  `module-hygiene` change scopes `backend/` out (see proposal — Impact).

## Migration Plan

This is a pre-`1.0` internal crate with no external consumers yet; the trait and its two
backends are rewritten together. Backend tests (`backend_tests.rs`) are migrated in the
same change. No rollback surface beyond git history.

## Open Questions

- Whether `norm` is exposed as one dispatch function or two (`sk_matrix_norm` /
  `sk_vector_norm`) — a naming detail, resolvable during implementation without changing
  the spec.
- Exact tolerance/`rcond` semantics to centralize in `lstsq` (scikit-learn's `rcond =
  -1` default vs a fixed epsilon) — confirm against PRD/scikit-learn parity when
  implementing `LinearRegression`.