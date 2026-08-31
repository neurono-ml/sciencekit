## 1. Complete the wave-plan-foundation change artifacts

- [x] 1.1 Research scikit-learn implementations per algorithm category (subagent A — sklearn internals report)
- [x] 1.2 Research canonical academic references + Rust ML ecosystem (subagent B — references + crate map)
- [x] 1.3 Research Rust numerical foundations (subagent C — ndarray/sprs/rayon/SIMD/allocators/memmap/BLAS/GPU)
- [x] 1.4 Spike: pure-Rust BLAS path (`oxiblas` vs `faer` vs `matrixmultiply`) — subagent + `cargo info` verification
- [x] 1.5 Spike: GPU backend order (`cudarc` vs `ocl`/`opencl3` vs `cubecl-hip-sys`) + NVIDIA OpenCL deprecation claim verification
- [x] 1.6 Spike: online quantile sketch for RobustScaler (`tdigest` vs `kll-rs` vs `sketch_oxide`) — `cargo info` verification
- [x] 1.7 Spike: pure-Rust sparse SVD for TruncatedSVD (`single-svdlib` vs `faer-sparse` vs `oxiblas-sparse` vs hand-rolled Lanczos)
- [x] 1.8 Spike: nested rayon thread management (reasoning — rayon owns all parallelism, BLAS single-threaded in scopes)
- [x] 1.9 Write `proposal.md` (why + what + 5 spike decisions summary)
- [x] 1.10 Write `design.md` (6 decisions in detail + academic anchors + Rust ecosystem lessons + dependency pins + risks)
- [x] 1.11 Write `tasks.md` (this file — meta-tasks + BLAS spike benchmark plan + wave breakdown reference)

## 2. Validate the change

- [ ] 2.1 Run `openspec validate wave-plan-foundation` and resolve any errors
- [ ] 2.2 Confirm `skip_specs: true` is honoured (no spec deltas required — planning/meta change)
- [ ] 2.3 Independent review by a separate agent validating that the wave plan + spike decisions are internally consistent and the PRD §8.7/§10.3 acceptance criteria are reflected per wave
- [ ] 2.4 Commit this change's `openspec/` definitions on branch `docs/wave-plan-foundation-openspec` in worktree `temporary/worktrees/docs/wave-plan-foundation-openspec`
- [ ] 2.5 Open the matching GitHub issue as an ADR (Y-statement — this is a foundational planning decision; draft with the `architecture-decision-records` skill)
- [ ] 2.6 Open the PR with closing keyword referencing the issue (`Closes #<n>`)
- [ ] 2.7 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under an "Added" / "Changed" entry for the wave plan foundation

## 3. Deferred BLAS spike benchmark (gates Decision 1, runs before W2.1)

This benchmark runs under `temporary/2026-08-26/blas-spike/` (per AGENTS.md scratch-artifacts convention — never committed). It gates the lock of the pure-Rust BLAS default (`oxiblas 0.2.2` vs `faer 0.24.4`).

- [ ] 3.1 Create the scratch benchmark crate under `temporary/2026-08-26/blas-spike/` (a standalone Cargo project, not part of the sciencekit workspace)
- [ ] 3.2 Add dev-dependencies: `oxiblas 0.2.2` (features `std, sparse, ndarray, parallel`), `faer 0.24.4` (default features), `ndarray-linalg 0.18` + `blas-src`/`openblas-src` (reference), `criterion 0.5`, `approx 0.5.1`, `ndarray 0.17`
- [ ] 3.3 Benchmark 1: GEMM 1024×1024 (f32, f64) — compare oxiblas vs faer vs OpenBLAS reference
- [ ] 3.4 Benchmark 2: GEMM tall-skinny 5000×128 × 128×5000 (f64) — packing/threading efficiency
- [ ] 3.5 Benchmark 3: full SVD 500×500 dense (f64) — Jacobi vs divide-and-conquer path
- [ ] 3.6 Benchmark 4: truncated SVD 500×5000, k=50 (f64) — `RandomizedSvd` vs LAPACK `svddc`
- [ ] 3.7 Benchmark 5: economy QR 2000×500 (f64) — blocked vs recursive QR
- [ ] 3.8 Benchmark 6: Cholesky 1000×1000 SPD (f64) — `compute_blocked_par`
- [ ] 3.9 Benchmark 7: symmetric eigh 500×500 (f64) — most-used LAPACK call after SVD
- [ ] 3.10 Benchmark 8: linear solve Ax=b, 1000 RHS (f64) — LU pivot + triangular solve
- [ ] 3.11 Property-test numerical correctness: every result must agree to ≤1e-9 (f64) / ≤1e-5 (f32) relative error vs the OpenBLAS reference
- [ ] 3.12 Record the lock decision in an ADR (Y-statement): which crate is the pure-Rust default, with the benchmark evidence. If `oxiblas` fails accuracy on Schur/GeneralEvd, downgrade per Decision 1's escape clause.

## 4. Reference — Wave breakdown (catalogue for downstream changes)

> The following is the **reference catalogue** of downstream changes this plan establishes.
> These are NOT tasks of this `wave-plan-foundation` change — each is a separate change on
> its own worktree branch, opened against this plan. Listed here so the plan is
> self-contained. Per AGENTS.md: type ∈ `feat|bugfix|chore|docs`; algorithm changes go in
> group (1), accelerator (Python/GPU/BLAS/SIMD/allocator) in group (2), docs in group (3).

### Wave 0 — Foundations (PRD Phase 0) · sequential

- `bootstrap-workspace` (chore) — Cargo workspace, toolchain (Rust 1.85, edition 2024), CI, license (Apache-2.0), README badges. Anchor: PRD §3.1, §3.6.
- `common-core-foundation` (feat) — `sciencekit_common`: `SKEstimator`/`SKPredictor`/`SKTransformer`/`SKDataSource`/`SKLazySource`/`SKMappableSource`/`SKComputeBackend`/`SKToOnnx` traits, `SKExecutionMode`, central errors, types. Anchor: PRD §3.3. Steal: `linfa::DatasetBase` + `burn::Backend`.
- `math-kernel-foundation` (feat) — `sciencekit_math`: ndarray ops, layouts, pairwise distances, `sprs` sparse, SIMD/BLAS interface (BLAS spike sub-task §3 above). Anchor: PRD §4, §6.1-6.3.
- `execution-decision-and-observability` (feat) — automatic execution decision mechanism + base builders + `tracing`/OTel + custom allocators (`allocator-jemalloc`/`allocator-mimalloc` feature flags). Anchor: PRD §5.3, §9.

### Wave 1 — Preprocessing & Impute (PRD Phase 1) · after W0.4

- `standard-scaler` (feat) — `SKStandardScaler` with `partial_fit` (Chan-Golub-LeVeque incremental variance), `par_azip!`, memmap2 streaming. Anchor: ESL §3.4.
- `minmax-and-robust-scalers` (feat) — `SKMinMaxScaler` + `SKRobustScaler` with `tdigest 1.0.0` online quantile sketch (sklearn lacks `partial_fit` here). Anchor: ESL §2.4.2.
- `one-hot-and-polynomial-encoders` (feat) — `SKOneHotEncoder` (interned string table, perfect-hash, rayon CSR build) + `SKPolynomialFeatures` (streaming generator). Anchor: Kuhn & Johnson.
- `simple-imputers` (feat) — `SKSimpleImputer` strategies (mean/median/most_frequent/constant) + `MissingIndicator`, sparse-aware via `sprs`. Anchor: ESL §2.

### Wave 2 — Linear models (PRD Phase 2) · after W0.4 + BLAS spike

- `linear-regression-ols` (feat) — `SKLinearRegression` (SVD pseudoinverse, pure-Rust lstsq). Anchor: ESL §3; Golub & Van Loan.
- `ridge-regression` (feat) — `SKRidge` (Cholesky dense, sparse_cg, SVD fallback, zero-copy sparse `LinearOperator`). Anchor: Hoerl & Kennard 1970.
- `lasso-elasticnet-coordinate-descent` (feat) — `SKLasso` + `SKElasticNet` (coordinate descent + gap-safe screening, rayon across multi-target). Anchor: Tibshirani 1996; Zou & Hastie 2005; Friedman et al. 2010 (glmnet).
- `logistic-regression` (feat) — `SKLogisticRegression` (pure-Rust L-BFGS-B, SAGA for L1, multinomial). Anchor: Defazio et al. 2014 SAGA; Fan et al. 2008 LIBLINEAR.
- `sgd-classifier-regressor` (feat) — `SKSGDClassifier` + `SKSGDRegressor` with `SKLazySource` streaming. Anchor: Bottou 2010; Pegasos.

### Wave 3 — Neighbors + SVM (PRD Phase 3) · after W2

- `pairwise-metrics-complete` (feat) — `sciencekit_metrics` pairwise: Euclidean, Manhattan, Cosine, Chebyshev, nan_euclidean, SIMD. Anchor: foundational.
- `kd-tree-and-ball-tree` (feat) — `SKKDTree` + `SKBallTree` (array-based binary tree, rayon batch query, SIMD leaf scan, memmap2 backing). Anchor: Bentley 1975; Omohundro 1989.
- `knn-classifier-regressor` (feat) — `SKKNeighborsClassifier` + `SKKNeighborsRegressor` (uses `SKKDTree`/`SKBallTree` + brute-force fallback). Anchor: Cover & Hart 1967.
- `knn-imputer` (feat) — `SKKNNImputer` built directly on `SKKDTree`/`SKBallTree` (build once, query per feature with `nan_euclidean`); NO brute-force placeholder. Anchor: Cover & Hart 1967.
- `svm-svc-svr` (feat) — `SKSVC` + `SKSVR` (pure-Rust SMO, kernel trick, OvO multiclass, memmap2 kernel cache). Anchor: Platt 1998 SMO; Chang & Lin 2011 LIBSVM.
- `svm-linear-svc` (feat) — `SKLinearSVC` (dual coordinate descent, zero-copy CSR operator). Anchor: Fan et al. 2008 LIBLINEAR.

### Wave 4 — Trees + Ensembles (PRD Phase 4) · after W3

- `decision-tree-cart` (feat) — `SKDecisionTreeClassifier` + `SKDecisionTreeRegressor` (array-based Tree, nested rayon across features per node, `bitvec` categorical, memmap2 backing, serde persistence). Anchor: Breiman et al. 1984; Quinlan 1986.
- `random-forest` (feat) — `SKRandomForest` (rayon across trees, lock-free accumulator, OOB). Anchor: Breiman 2001.
- `bagging` (feat) — `SKBagging` (rayon bootstrap, generic over `SKEstimator`). Anchor: Breiman 1996.
- `gradient-boosting` (feat) — `SKGradientBoosting` (Friedman 2001, regression-tree base, Newton-Raphson leaf weight). Anchor: Friedman 2001.
- `histogram-gradient-boosting` (feat) — `SKHistGradientBoosting` (LightGBM-style bins, `bitvec` categorical, SIMD histogram, streaming over memmap2-binned). Anchor: Ke et al. 2017 LightGBM.

### Wave 5 — Unsupervised (PRD Phase 5) · after W3/W4

- `kmeans-lloyd-elkan` (feat) — `SKKMeans` (Lloyd + Elkan triangle inequality, k-means++ init, `par_azip!` + SIMD argmin, memmap2). Anchor: Lloyd 1982; k-means++; Elkan 2003.
- `minibatch-kmeans-streaming` (feat) — `SKMiniBatchKMeans` (`SKLazySource` streaming, per-center learning rates). Anchor: Sculley 2010.
- `dbscan` (feat) — `SKDBSCAN` (region query via `SKKDTree`, `petgraph` connected components, CSR ragged graph). Anchor: Ester et al. 1996.
- `pca` (feat) — `SKPCA` (full SVD, randomized SVD, covariance_eigh streaming via memmap2). Anchor: Halko, Martinsson, Tropp 2011.
- `truncated-svd` (feat) — `SKTruncatedSVD` (Decision 4 path: `faer-sparse`+`rsvd-faer` OR `oxiblas-sparse::RandomizedSvd`+adapter OR hand-rolled Lanczos; `arpack-sys` opt-in). Anchor: Halko et al. 2011.
- `incremental-pca` (feat) — `SKIncrementalPCA` (`partial_fit`, Chan-Golub-LeVeque Cov accumulator). Anchor: as above.
- `outlier-detection` (feat) — `SKIsolationForest` + anomaly detection (uses W4.1 tree infra). Anchor: Liu, Ting, Zhou 2008.

### Wave 6 — Selection, pipelines, metrics (PRD Phase 6) · after W1-W5

- `model-selection-core` (feat) — `sk_train_test_split`, `SKKFold`, `SKStratifiedKFold`, `SKGroupKFold`. Anchor: foundational.
- `grid-search-cv` (feat) — `SKGridSearchCV` (rayon, nested-pool Semaphore, HalvingGridSearch). Anchor: sklearn user guide.
- `metrics-suite` (feat) — full metrics: accuracy, f1, precision/recall, MSE/MAE/RMSE, confusion_matrix, roc_auc, streaming `MetricAccumulator`. Anchor: foundational.
- `type-safe-pipeline` (feat) — `SKPipeline` (associated types, compile-time step compatibility), `FeatureUnion`, DAGs via `petgraph`. Anchor: software architecture.
- `streaming-pipeline` (feat) — batched `Iterator` between stages, memmap2 backing. Anchor: PRD §4.4 differentiator.

### Wave 7 — Interop + production (PRD Phase 7) · after W1-W6

- `safetensors-internal-format` (feat) — `sciencekit_interop`: extended Safetensors (JSON header, sharding, padding, compression `.gz`/`.brotli`/`.snappy`). Anchor: Safetensors spec.
- `onnx-export-import` (feat) — `SKToOnnx` trait for all estimators; ONNX import as generic `SKPredictor`. Anchor: ONNX spec.
- `json-debug-format` (feat) — human-readable serialization via central export trait. Anchor: PRD §8.6.
- `polars-datafusion-sources` (feat) — `Into<>` conversions for Polars DataFrame/LazyFrame, DataFusion DataFrame, Arrow/Parquet. Anchor: Polars/DataFusion docs.
- `umbrella-crate` (chore) — `sciencekit` re-exports + unified docs. Anchor: PRD §3.1.
- `python-bindings-complete` (feat) — `sciencekit_python` complete PyO3 bindings, zero-copy `PyReadonlyArray`→`ArrayView`. Anchor: PyO3 docs.

### Cross-cutting (per wave, after CPU validation)

- **Python bindings** (feat, group 2): per-wave `python-bindings-<wave>` after each algorithm wave is CPU-validated.
- **GPU backends** (feat, group 2): **`gpu-opencl-backend` first** (via `opencl3` + Rusticl
  Mesa driver; kernels: pairwise distance, GEMM, tree predict, elementwise; CPU keeps heavy
  BLAS/SVD per Decision 2). **Metal after OpenCL** (on request). **CUDA/ROCm only when
  explicitly requested** — not in the active roadmap. Focus is CPU + OpenCL.
- **Docs** (docs, group 3): per-wave `docs-<wave>-chapter` mdBook chapter + Mermaid diagrams, merged to `docs/documentations` branch (not `main`).

## 5. Open Questions to track (deferred, do not block this change)

- [ ] 5.1 At W5.5 (TruncatedSVD): decide whether to bump MSRV 1.85 → 1.88 to unblock `single-svdlib` (sprs-native IRLBA). Revisit then.
- [ ] 5.2 At W7 (Interop): decide `cubecl`'s OpenCL-runtime adoption as a potential absorber of the custom `OpenClBackend` kernels. Revisit then.
- [ ] 5.3 At W0.3 (math kernel): decide `nalgebra` role for small fixed-size matrices. Revisit then.
