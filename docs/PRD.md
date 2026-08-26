# PRD — `sciencekit`

**Product Requirements Document**

Version: 0.1.0 (initial draft)
Date: 2026-08-24
Status: Approved for implementation

---

## Summary

1. [Overview](#1-overview)
2. [User Interface](#2-user-interface)
3. [Architecture and Code Organization](#3-architecture-and-code-organization)
4. [Memory Management](#4-memory-management)
5. [Threads and Concurrency](#5-threads-and-concurrency)
6. [Libraries and Dependencies](#6-libraries-and-dependencies)
7. [Technology Optimization](#7-technology-optimization)
8. [Model Export and Import](#8-model-export-and-import)
9. [Errors and Observability](#9-errors-and-observability)
10. [Development Methodology](#10-development-methodology)
11. [Versioning and License](#11-versioning-and-license)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Overview

### 1.1. Goal

`sciencekit` is a Machine Learning library written natively in Rust exposing **all scikit-learn algorithms and utilities**, rewritten from scratch in an optimized form. Its competitive differentiators are:

- **Extreme performance:** native code, SIMD, data parallelism, custom allocators.
- **Memory safety:** Rust guarantees without a garbage collector.
- **Native out-of-core:** every algorithm supports datasets larger than RAM, through streaming or memory-mapping.
- **Concurrency as a first-class citizen:** automatic execution-mode decision, configurable by the user.

### 1.2. Positioning

A library equivalent to scikit-learn in algorithmic coverage, but with Rust's performance and safety. Target audience:

- ML engineers in Rust who need a complete toolkit.
- Python teams that need high-performance inference/training through bindings.
- Production systems where scikit-learn's pickle is a security risk.

### 1.3. Scope

**Is:**
- A Rust library (Cargo workspace with sub-crates).
- An idiomatic native Rust API (builder pattern).
- Python bindings (PyO3) with zero-copy.
- Complete coverage of the scikit-learn taxonomy.
- ONNX, Safetensors and JSON debug export/import.

**Is not (at this time):**
- A CLI for running workflows *(recorded as a future improvement)*.
- A deep learning framework.
- An inference service/server.

---

## 2. User Interface

### 2.1. Native Rust API (primary)

The primary interface is the native Rust API, following the **builder pattern** for all configuration:

```rust
// Illustrative example of the expected interface
let model = SKKMeansClassifierBuilder::new()
    .number_of_clusters(8)
    .maximum_iterations(300)
    .execution_mode(SKExecutionMode::Automatic)
    .build()?;

model.fit(&training_data_view)?;
let predictions = model.predict(&test_data_view)?;
```

Principles:

- **Mandatory builder** for every estimator, transformer and pipeline. Direct constructors stay private.
- **Zero-copy inputs:** public APIs receive `ArrayView`, `CowArray` or sparse views (`sprs`) — never copies of huge matrices.
- **Type-safe pipelines:** compile-time validation of step compatibility through associated types.
- **Full descriptive names:** no abbreviations in any public or private item — the only exception being the project prefix `sk`/`SK` (see section 3.4).

### 2.2. Python Bindings (PyO3)

Secondary yet mandatory interface for every algorithm:

- **Timing:** as soon as an algorithm is complete and validated in Rust, an associated change creates the corresponding Python interface, with proper tests. No algorithm is considered "done" without a Python binding.
- **Its own idiomatic API** following the builder pattern (it is not a drop-in sklearn replacement).
- **Maximum zero-copy:** NumPy ↔ ndarray conversion through views without copying whenever possible (`PyReadonlyArray` → `ArrayView`).
- Implemented in the `sciencekit_python` crate.

### 2.3. CLI

There will be no CLI at this stage. Recorded on the roadmap as a future improvement (declarative workflow execution).

---

## 3. Architecture and Code Organization

### 3.1. Cargo Workspace

The project is split into cohesive sub-crates to optimize build times and isolate dependencies:

```
sciencekit/
├── Cargo.toml              # workspace root
├── crates/
│   ├── sciencekit/                  # umbrella crate (re-exports)
│   ├── sciencekit_common/           # base traits, types, central errors
│   ├── sciencekit_math/             # linear algebra, BLAS, SIMD, sparse, pairwise
│   ├── sciencekit_preprocessing/    # scalers, encoders, polynomial features
│   ├── sciencekit_impute/           # simple and multivariate imputers
│   ├── sciencekit_linear_model/     # linear regressions, logistic, SGD
│   ├── sciencekit_neighbors/        # SKKNeighborsClassifier/Regressor, SKKDTree, SKBallTree
│   ├── sciencekit_svm/              # SKSVC, SKSVR, SKLinearSVC
│   ├── sciencekit_tree/             # decision trees
│   ├── sciencekit_ensemble/         # random forest, bagging, gradient boosting
│   ├── sciencekit_cluster/          # SKKMeans, SKDBSCAN, SKMiniBatchKMeans
│   ├── sciencekit_decomposition/    # SKPCA, SKTruncatedSVD
│   ├── sciencekit_outlier/          # anomaly detection
│   ├── sciencekit_model_selection/  # SKKFold, sk_train_test_split, SKGridSearchCV
│   ├── sciencekit_pipeline/         # type-safe SKPipeline, DAGs
│   ├── sciencekit_metrics/          # evaluation metrics
│   ├── sciencekit_interop/          # ONNX, Safetensors, I/O traits
│   ├── sciencekit_gpu/              # SKComputeBackend + OpenCL/CUDA/ROCm
│   └── sciencekit_python/           # PyO3 bindings
└── docs/
```

### 3.2. Structural rules

| Rule | Detail |
|---|---|
| **One responsibility per module** | Modules small enough to have exactly one responsibility. |
| **200-line limit** | A `.rs` file exceeding 200 lines becomes a **folder module**: `my_module/mod.rs` (or `my_module.rs` + directory) with faithful, standardized sub-module organization. |
| **Single responsibility per item** | Functions, structs and traits do a single thing. Prefer **trait composition** over inheritance of responsibilities. |
| **Small PRs** | Each change/PR alters the minimum necessary. Avoid PRs with many modifications. |

Standard folder module organization:

```
standard_scaler/
├── mod.rs                    # public re-exports and module docs
├── builder.rs                # SKStandardScalerBuilder
├── core_implementation.rs    # core transformer logic
├── fitting_logic.rs          # fit phase
├── transformation_logic.rs   # transform phase
└── standard_scaler_tests.rs  # companion tests
```

### 3.3. Fundamental traits (`sciencekit_common`)

Trait composition as the foundation of the whole library:

```rust
pub trait SKEstimator { /* hyperparameters and fit */ }
pub trait SKPredictor: SKEstimator { /* predict */ }
pub trait SKTransformer: SKEstimator { /* transform / fit_transform */ }

pub trait SKDataSource { /* eager: full in-memory access */ }
pub trait SKLazySource { /* streaming: Iterator<Item = Batch> */ }
pub trait SKMappableSource { /* memmap: O(1) random access */ }
pub trait SKToOnnx { /* ONNX export */ }
pub trait SKComputeBackend { /* compute device abstraction */ }
```

Algorithms compose exactly the traits they support. An `SKSGDClassifier` implements `SKPredictor` + `SKLazySource`; an `SKStandardScaler` implements `SKTransformer`.

### 3.4. Naming convention

- **No abbreviations, with a single exception: the project prefix `sk`/`SK`.** Names of modules, functions, variables, structs, traits and any Rust object are complete and self-explanatory in context. Examples: `maximum_number_of_iterations` (not `max_iter`), `nearest_neighbors_count` (not `k`). The `sk` prefix is the only allowed abbreviation; its use is optional in internal/private items and mandatory in public items (rule below).
- **Mandatory prefix on public items:** every public object (externally accessible) created in this project receives the project prefix:
    - Structs and traits: `SK` + PascalCase — e.g.: `SKEstimator`, `SKStandardScaler`, `SKExecutionMode`;
    - Public free functions (outside `impl`), public variables and modules: `sk_` + snake_case — e.g.: `sk_train_test_split`;
    - Methods **do not** receive the prefix — methods are functions inside `impl` blocks of structs or traits (the OO-method equivalent; called methods from here on): they use complete names only — e.g.: `fit`, `predict`, `execution_mode`.
- Crate prefix: `sciencekit_*` (always the full name, never abbreviated).
- APIs hidden with `#[doc(hidden)]` are not considered public — they can be refactored without breaking changes.

### 3.5. Tests

- Tests live in **companion `_tests.rs` modules** next to implementation files (e.g.: `standard_scaler_tests.rs` beside `core_implementation.rs`), not inline nor in a global `tests/` directory.
- Mock data built with `ndarray`/`sprs`.
- **Mandatory TDD:** test first, confirmed failure, then implementation.

### 3.6. Toolchain

| Item | Value |
|---|---|
| MSRV | **Rust 1.85** (stable) |
| Edition | **2024** |

---

## 4. Memory Management

### 4.1. Zero-copy

Mandatory across the entire public surface and internal hot paths:

- Trait inputs receive `ArrayView`, `CowArray` (and sparse equivalents from `sprs`) — never copies of giant matrices.
- Mutable transformations use `.map_inplace()` / `CowArray::into_owned` only when unavoidable.
- Always consider **memory layout** (row-major/column-major, contiguity) in binary operations — CPU efficiency depends on it.

### 4.2. Higher-order functions (mandatory)

Performance depends on avoiding manual index-based iteration. Prioritize mandatorily:

- `.map()`, `.map_inplace()`, `.zip_mut_with()` for simple traversals;
- `azip!()` and `par_azip!()` for lock-efficient iterations;
- `rayon` for data parallelism.

### 4.3. Custom allocators

| Allocator | Feature flag | Platforms | Characteristics |
|---|---|---|---|
| glibc malloc | (none — fallback) | all | System default, zero dependencies |
| jemalloc | `allocator-jemalloc` | all | Per-thread arenas; shines with many same-size allocations |
| mimalloc | `allocator-mimalloc` | all | Very low p99 latency; variable sizes |

- Both available on **every platform**; the user chooses via the builder.
- The **automatic decision mechanism** (section 5.3) recommends/selects the allocator by default according to platform and detected workload.
- Expected gains vs glibc on ML workloads: 15–30% (allocation throughput).

### 4.4. Out-of-core: two complementary traits

Every algorithm supports out-of-core — it is a differentiator of the library. Each algorithm **declares its access pattern** and implements the trait(s) that work for it:

| Trait | Strategy | Typical algorithms |
|---|---|---|
| `SKLazySource` | Sequential streaming in batches (`Iterator<Item = Batch>`) | SKSGD*, SKMiniBatchKMeans, incremental SKPCA |
| `SKMappableSource` | Memory-mapped files (`memmap2`), O(1) random access | SKKMeans, SKKNN, SKDBSCAN, SKDecisionTree |

- If an algorithm can use both, it implements both; otherwise it implements only the one that works.
- Mappable format: contiguous binary with known layout (compatible with the internal model format, section 8).

### 4.5. Sparse arrays

Support from day one via **`sprs` + `ndarray`** (CSR/CSC/COO). Essential for Lasso, linear SVMs and text classification. Central traits accept dense or sparse views depending on the algorithm.

---

## 5. Threads and Concurrency

### 5.1. CPU / I/O separation

Non-negotiable rule: **never block asynchronous threads with CPU-intensive work.**

- Mathematical processing runs on a dedicated CPU pool (**rayon**).
- Async I/O (network, streams) uses its own runtime (Tokio), delegating computation blocks to the CPU pool.

### 5.2. Data parallelism

- Inside algorithms (multiplications, distances, aggregations): `rayon` + `par_azip!`.
- Enabled/disabled through **feature flags** (per capability).
- Thread pool: rayon default; **configurable** by the user (thread count, pinning) via the builder.

### 5.3. Automatic execution decision mechanism

Every builder exposes `execution_mode(SKExecutionMode::...)`. Default: `Automatic`.

**Execution modes:**

| Mode | Description |
|---|---|
| `InProcessSynchronous` | Dataset fits in RAM; eager algorithm; rayon only |
| `InProcessAsynchronous` | Async I/O source (network/stream); Tokio orchestrates, rayon computes |
| `OutOfCoreStreaming` | Dataset > RAM; sequential disk batches (`SKLazySource`) |
| `OutOfCoreMemoryMapped` | Dataset > RAM; random access required (`SKMappableSource`) |

**Parameters used by the automatic decision:**

- `available_memory`: free RAM (detected via sysinfo or informed);
- `dataset_size`: samples × features × bytes per element;
- `algorithm_access_pattern`: declared by the algorithm (sequential/random/iterative);
- `cpu_cores`: via `std::thread::available_parallelism()`;
- `batch_size_hint`: optional, provided by the user.

**Behavior:** transparent automatic defaults; the user may override any parameter in the builder (`SKPipeline` configuration has lower precedence than the specific estimator's).

---

## 6. Libraries and Dependencies

### 6.1. Core

| Crate | Use |
|---|---|
| `ndarray` | Dense matrices, zero-copy views, `azip!`/`par_azip!` |
| `sprs` | CSR/CSC/COO sparse arrays |
| `rayon` | Data parallelism |
| `tokio` | Async runtime for I/O |
| `serde` | Serialization basis (analogous to the standard export trait) |
| `thiserror` | Custom errors per algorithm |
| `anyhow` | Convenience in consumer applications |
| `tracing` | Structured observability |
| `memmap2` | Memory-mapped files |
| `sysinfo` | Available RAM detection |

### 6.2. Linear algebra (BLAS/LAPACK) — hybrid

- **Default:** pure Rust implementations (`sciencekit_math`), zero C/Fortran dependencies.
- **Feature flag `blas-backend`:** swaps critical operations (matmul, SVD, QR) for BLAS/LAPACK.

| Backend | License | Policy |
|---|---|---|
| OpenBLAS | BSD-3-Clause | Default of the `blas-backend` flag |
| BLIS | BSD-3-Clause | Alternative |
| Intel MKL | Proprietary | **Never a direct dependency**; explicit opt-in via `blas-mkl`, user-installed |
| Apple Accelerate | System framework | macOS-specific flag, no binary redistribution |

The default library configuration (no flags) is pure Rust, free of proprietary dependencies.

### 6.3. SIMD — hybrid

- **Base:** `ndarray`/`azip!` code written for LLVM autovectorization (covers ~80% of cases, stable).
- **Critical hot paths** (identified by profiling): the `wide` crate (portable explicit SIMD, stable).
- **Future:** migrate `wide` → `std::simd` once stabilized.

### 6.4. GPU

Abstraction via the own **`SKComputeBackend` trait** (in `sciencekit_gpu`) + existing FFI binding crates:

| Backend | Binding | Phase |
|---|---|---|
| OpenCL | `ocl` | Start |
| CUDA | `cudarc` | Start |
| ROCm/HIP | HIP bindings | Start |
| SYCL/oneAPI, Metal, Vulkan | TBD | Future |

- CPU is the default backend, always present.
- Support for each backend arrives as a **separate change, after the algorithm is ready** on CPU.

### 6.5. Data I/O

- Code depends only on **arbitrary I/O traits** (data sources/sinks defined in `sciencekit_interop`).
- Integration via `Into<>` conversions:
  - Polars DataFrame/LazyFrame → source;
  - DataFusion DataFrame → source;
  - Arrow/Parquet supported through those engines.

### 6.6. Interoperability

| Crate | Use |
|---|---|
| `pyo3` | Python bindings (`sciencekit_python`) |
| `safetensors` | Internal/export model format |
| `onnx`/`ort` (to validate in technical spec) | ONNX export/import |

---

## 7. Technology Optimization

### 7.1. Optimization layers

1. **Algorithmic:** choose the correct complexity before micro-optimizing.
2. **Memory layout:** contiguity, SoA vs AoS where relevant, cache-friendliness.
3. **Higher-order functions:** `azip!`/`par_azip!`/`zip_mut_with` (mandatory — see 4.2).
4. **SIMD:** autovectorization → `wide` on profiled hot paths.
5. **BLAS:** opt-in flag for heavy dense algebra.
6. **GPU:** additional backend per separate change.
7. **Allocators:** automatic choice/configurable.

### 7.2. Mandatory iterative evolution per algorithm

1. **Naive** implementation (simple, sequential).
2. Unit tests (TDD).
3. Refactoring for **performance** (SIMD, rayon, layout).
4. Refactoring for **streaming/out-of-core** (SKLazySource/SKMappableSource).

No stage may be skipped.

### 7.3. Feature flags

Dual granularity:

- **Per capability:** `parallel`, `allocator-jemalloc`, `allocator-mimalloc`, `blas-backend`, `blas-mkl`, `gpu-opencl`, `gpu-cuda`, `gpu-rocm`, `telemetry-opentelemetry`, ...
- **Per algorithm group:** `classification`, `regression`, `clustering`, `decomposition`, `preprocessing`, ...

---

## 8. Model Export and Import

### 8.1. Standard export trait

Analogous to serde's role: a central model serialization/deserialization trait from which concrete implementations derive (internal Safetensors, JSON debug, ONNX, public Safetensors).

### 8.2. Internal format: extended Safetensors

- **Single model format:** safetensors — JSON header + contiguous tensors with random access by name.
- The JSON header carries arbitrary metadata:
  - `"sciencekit_format_version"`: format version (for future migrations);
  - algorithm type, builder hyperparameters, training state;
  - **recoverability metadata:** indicates how the model can be reloaded to continue training (checkpointing).
- Clear delimitation of where a model starts/ends and of its parts/operators.

### 8.3. Partial writes (no full rewrite)

| Case | Strategy |
|---|---|
| Update existing tensor (same size) | In-place write at known offset (always works) |
| Add tensor — large model | **Sharding:** immutable `.safetensors` shards + external index; new tensors go to a new shard |
| Add tensor — small model | **Header padding:** reserved space allows append + in-place header while padding lasts |

Sharding/padding choice belongs to the automatic decision mechanism (model size), configurable.

### 8.4. Compression

Compressed variants of the internal format: `.safetensors.gz`, `.safetensors.brotli`, `.snappy.safetensors`.

### 8.5. ONNX

- **Export:** every estimator/pipeline implements `SKToOnnx`.
- **Import (immediate):** ONNX and Safetensors models trained in other frameworks load as a **generic SKPredictor**, usable standalone or inside an SKPipeline.
- **Import (future — recorded):** converting imported ONNX/Safetensors models into native library types, allowing resumed training or LoRA creation.

### 8.6. JSON debug

Human-readable serialization for inspection/debug, following the same central export trait.

### 8.7. Mandatory validation

Every implementation validates, at minimum:

1. Full algorithm execution with **lots of data** and with **little data**;
2. Execution under **concurrency**;
3. **Model export** and **metric generation**.

---

## 9. Errors and Observability

### 9.1. Errors

- `thiserror` for error enums **per algorithm** (specific errors preserved).
- `From<CentralError>` implemented in algorithm enums when it makes sense — common errors (shape mismatch, invalid dtype, I/O) propagate uniformly across algorithms.
- `anyhow` recommended for consumer applications (ergonomics), not inside the library.

### 9.2. Observability

- **`tracing`** as the base; `tracing-subscriber` when necessary.
- Spans/logs oriented to the **OpenTelemetry format**.
- Out-of-the-box **OpenTelemetry** support via `tracing-opentelemetry`.
- Disableable telemetry (feature flag + builder configuration); near-zero cost when disabled.

---

## 10. Development Methodology

### 10.1. Workflow

- **OpenSpec** guides every change specification; **graphify** aids navigation/comprehension of the project knowledge graph.
- All development happens in **git worktrees**. **No direct commits to the main branch.**
- Small changes; explicit preference for PRs with few modifications.

### 10.2. Mandatory TDD

1. Write the test (mock data in `ndarray`/`sprs`, companion `_tests.rs` module).
2. Confirm failure.
3. Implement the minimum to pass.
4. Refactor (performance → streaming), keeping tests green.

### 10.3. Acceptance checklist for any implementation

- [ ] Runs correctly with lots of data and with little data;
- [ ] Correct under concurrency (automatic mode + applicable explicit modes);
- [ ] Exports model (minimum Safetensors) and produces metrics;
- [ ] Companion `_tests.rs` modules covering everything above;
- [ ] Complete names, no abbreviations (single exception: sk/SK prefix — §3.4);
- [ ] Public items with mandatory SK/sk_ prefix per §3.4 (methods inside `impl` receive no prefix);
- [ ] No file > 200 lines without becoming a standardized folder module;
- [ ] Python binding created in an associated change (when applicable);
- [ ] GPU backend(s) added in a subsequent separate change (when applicable).

---

## 11. Versioning and License

### 11.1. Versioning

- **SemVer** with `0.x` prefix during development/instability.
- Migration to stable SemVer (1.x) once the API reaches maturity.
- Breaking public-API changes only between minor/major per SemVer; `#[doc(hidden)]` items may change freely.

### 11.2. Model format

- `"sciencekit_format_version"` in the JSON header of every model file; loaders support reading previous versions.

### 11.3. License

- **Apache-2.0** (includes patent protection).

---

## 12. Implementation Roadmap

Order faithful to the scikit-learn mapping (from the handoff), respecting dependencies between crates:

### Phase 0 — Foundations
1. Workspace, toolchain (MSRV 1.85, edition 2024), CI, license.
2. `sciencekit_common`: traits `SKEstimator`, `SKPredictor`, `SKTransformer`, `SKDataSource`, `SKLazySource`, `SKMappableSource`; central errors; types.
3. `sciencekit_math`: higher-order ops, layouts, pairwise distances, sparse (`sprs`), SIMD/BLAS interface.
4. Automatic execution decision mechanism + base builders + tracing/OTel.

### Phase 1 — Preprocessing
5. `sciencekit_preprocessing`: `SKStandardScaler`, `SKMinMaxScaler`, `SKRobustScaler`, `SKOneHotEncoder`, `SKPolynomialFeatures`.
6. `sciencekit_impute`: simple strategies, `SKKNNImputer`.

### Phase 2 — Linear models
7. `sciencekit_linear_model`: `SKLinearRegression`, `SKRidge`, `SKLasso`, `SKElasticNet`, `SKLogisticRegression`; `SKSGDClassifier`/`SKSGDRegressor` with `SKLazySource`.

### Phase 3 — Neighborhood and SVM
8. `sciencekit_metrics` (pairwise already in math) + `sciencekit_neighbors`: SKKNeighborsClassifier/SKKNeighborsRegressor, SKKDTree, SKBallTree.
9. `sciencekit_svm`: SKSVC, SKSVR, SKLinearSVC.

### Phase 4 — Trees and ensembles
10. `sciencekit_tree`: SKDecisionTreeClassifier/SKDecisionTreeRegressor.
11. `sciencekit_ensemble`: parallel aggregation with rayon (SKRandomForest, SKBagging, SKGradientBoosting).

### Phase 5 — Unsupervised
12. `sciencekit_cluster`: SKKMeans, SKDBSCAN, SKMiniBatchKMeans (streaming).
13. `sciencekit_decomposition`: SKPCA, SKTruncatedSVD.
14. `sciencekit_outlier`: anomaly detection.

### Phase 6 — Selection, pipeline, metrics
15. `sciencekit_model_selection`: SKKFold, sk_train_test_split, SKGridSearchCV.
16. `sciencekit_pipeline`: type-safe SKPipeline (associated types), DAGs.
17. `sciencekit_metrics` complete: accuracy, f1, MSE, confusion matrix.

### Phase 7 — Interop and production
18. `sciencekit_interop`: safetensors (internal, sharding/padding, compression), ONNX export/import, JSON debug, I/O traits + Polars/DataFusion.
19. `sciencekit` umbrella crate: re-exports and unified documentation.
20. `sciencekit_python`: complete PyO3 bindings.

### Cross-cutting (continuous)

- **Per completed algorithm:** Python binding in an associated change; GPU backend (OpenCL → CUDA → ROCm) in following separate changes.
- **Recorded future improvements (out of scope now):** workflows CLI; ONNX/Safetensors import converted to native types with retraining/LoRA.

---

*Document generated from decisions consolidated with the product owner. Relevant scope changes require updating this PRD before the technical spec.*
