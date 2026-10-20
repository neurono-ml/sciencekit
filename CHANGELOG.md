# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `README.md` with project overview, status badges and a validated architecture diagram.
- Repository workflow rules: keep-a-changelog policy, English-first durable artifacts, ADR issue per branch, Mermaid/SVG documentation standards.
- `CHANGELOG.md` seeded following Keep a Changelog 1.1.0.
- GitHub Actions workflow running the opencode agent on `/oc`/`/opencode` comments (issues and PR review comments).
- Versioned opencode project config registering the graphify plugin.
- `CI` workflow building and testing the Cargo workspace once it exists (green while Phase 0 is pending).
- Scratch-artifacts convention: temporary files go to `temporary/YYYY-MM-DD` (generic) or `temporary/YYYY-MM-DD/<change-name>` (change-related).
- Cargo workspace bootstrap (Phase 0.1): pinned Rust 1.85 toolchain, edition 2024, Apache-2.0 license, and CI gates (fmt, strict clippy, workspace tests + doctests, MSRV, examples).
- `sciencekit_common` sub-crate (Phase 0.2): the shared contract vocabulary — sealed `SKFloat` scalar bound, central `SKError` taxonomy, data/target views, label canonicalization, fit/scorer/execution/streaming traits.
- `sciencekit_math` sub-crate (Phase 0.3): higher-order `azip!`/`par_azip!` kernels, memory-layout helpers, zero-copy pairwise distances (Euclidean/Manhattan/Cosine) with a `wide` SIMD hot path, `sprs` CSR×dense and sparse×sparse products, and the `SKMathBackend` abstraction with `SKFaerBackend` (pure-Rust default) plus the opt-in `blas-backend` feature.
- `sciencekit_common` capability specs published (archive of `common-core-foundation`): scalar typing, error model, estimator contracts, data/target boundary, execution planning, scoring contracts, and streaming batches.
- `wave-plan-foundation` (planning): wave decomposition of PRD phases 0–7 into downstream changes (W0–W7), six foundational technical decisions locked (pure-Rust BLAS default → `faer 0.24.4`; OpenCL 3.0 ICD-agnostic GPU arrival order; `tdigest` online quantile sketch for `SKRobustScaler`; `faer-sparse` + `rsvd-faer` sparse SVD; nested-rayon thread management; `SKKNNImputer` moved to Wave 3), an academic-anchor catalogue, and reconciled dependency pins.
- `workspace-bootstrap` (planning): capability spec published and the change archived, capturing the Phase 0 workspace foundation, CI gates, and acceptance criteria.
- `sciencekit_common` (Phase 0.4): the automatic execution decision resolver (`sk_resolve_execution_plan` — turns `SKExecutionMode::Automatic` plus an `SKExecutionContext` into a concrete `SKExecutionPlan`), the mandatory builder foundation (`SKBuilder` trait + `SKBuilderState` defaulting to `Automatic`, `sk_validate_hyperparameter`, `SKReferenceEstimator` exemplar), observability (`SKOperationAttributes`/`sk_run_operation` tracing spans with an opt-in `observability-export` OpenTelemetry layer), and the `allocator-jemalloc`/`allocator-mimalloc` global-allocator feature flags. `sciencekit_math` exposes `sk_default_math_backend` (pure-Rust `faer` default, `blas-backend` opt-in) and per-backend `kind()` (`SKBackendKind`).
- **BREAKING** `sciencekit_common` generic float predictions: `SKRegressorPredictor<F>` returns `Array1<F>`, `SKClassifierPredictor<F>` returns `Array1<i64>` labels plus `Array2<F>` probabilities, `SKTargetView` is generic over `F: SKFloat`, and scorers split into continuous (`SKSupervisedScorer`, in `F`) and label (`SKLabelScorer`, over `i64`) contracts — f32 pipelines no longer pay an f64 detour.
- `sciencekit_preprocessing` sub-crate (Wave 1, first algorithm): `SKStandardScaler` with a builder-only surface (`with_mean`, `with_std`, `backend`, `execution_mode`) and a distinct `SKStandardScalerModel`; dense `fit`/`transform`, incremental `partial_fit` (exact Chan–Golub–LeVeque merge) and out-of-core `fit_streaming` through the Phase 0 streaming executor — all three regimes converge on the same model.
- `sciencekit_math` kernel `sk_axis_sum_squares`: the `Σx²` companion of `sk_axis_sum` (axis 0/1, sequential and race-free parallel paths, shared `SK_AXIS_SUM_GRAIN`).
- `backend-selection` capability published: every estimator builder exposes `backend(SKBackendKind)` (default `Faer`), every operation records the configured kind in its span, and `fit` carries it into the fitted model.
- `sciencekit_preprocessing` (Wave 1, min/max and robust scalers): `SKMinMaxScaler` (builder-only `feature_range`, `clip`, `backend`, `execution_mode`) and `SKRobustScaler` (builder-only `with_centering`, `with_scaling`, `quantile_range`, `unit_variance`, `backend`, `execution_mode`), each with a distinct fitted model; dense `fit`/`transform`, exact incremental `partial_fit` for min/max and an online `tdigest`-sketch `partial_fit` for the robust scaler, and out-of-core `fit_streaming` for both.
- `sciencekit_math` kernel `sk_axis_min_max`: the per-axis minimum and maximum in one pass, the extremum companion of the sum reductions (axis 0/1, sequential and race-free parallel paths, shared `SK_AXIS_SUM_GRAIN`).
- `sciencekit_common` categorical feature boundary (spec `data-view-boundary`): `SKNominalMatrixView` (zero-copy row-major borrowed labels) and the non-exhaustive `SKCategoricalView` enum (`Nominal` text columns / `Integer` `ArrayView2<i64>` codes), entered through the same fallible `TryInto` seam as the numeric boundary, plus the documented `SK_CATEGORICAL_MISSING_CODE` (`-1`) sentinel.
- `sciencekit_preprocessing` (Wave 1, encoders): `SKOneHotEncoder` (builder-only `sparse_output`, `handle_unknown`, `backend`, `execution_mode`) accepting nominal text and integer-coded input, learning deterministic first-occurrence per-column category tables (`SKLabelTable` / integer code table), rejecting or ignoring unknown categories and skipping the missing sentinel, emitting a dense `Array2` or a parallel-built `sprs` CSR matrix, with out-of-core `fit_streaming`; and `SKPolynomialFeatures` (builder-only `degree`, `interaction_only`, `include_bias`, `sparse_output`, `backend`, `execution_mode`) expanding numeric features into polynomial and interaction terms in the documented scikit-learn column order, dense or sparse, with out-of-core `fit_streaming`.
- Workspace dependency `sprs 0.11` promoted to a production dependency of `sciencekit_preprocessing` and `sciencekit_impute`, and `rayon 1` added, backing the parallel CSR output path.
- `sciencekit_impute` sub-crate (Wave 1, missing-data estimators): `SKSimpleImputer` (builder-only `strategy`, `missing_values`, `fill_value`, `keep_empty_features`, `add_indicator`, `backend`, `execution_mode`) with the mean (compensated sum), median (exact order statistic), most-frequent (ties to the smallest value) and constant strategies, dense and sparse fitting over one shared marker predicate (a `NaN` marker tests `is_nan`, any other marker tests equality, and a `0.0` marker also claims the CSR implicit zeros), all-missing columns recorded as dropped by default (or kept at zero), exact `partial_fit` and out-of-core `fit_streaming` — exact for mean/mode/constant and sketch-based for the median within the documented tolerance; and `SKMissingIndicator` (builder-only `features`, `error_on_new`, `sparse_output`, `missing_values`, `backend`, `execution_mode`) learning the missing-feature scope at fit and emitting a `0.0`/`1.0` mask, dense or CSR, rejecting missingness unseen at fit with the new `SKError::UnexpectedMissingness { column }`. `add_indicator` composes a real indicator, shares the marker and the backend kind, tolerates unseen missingness and stacks the mask after the imputed features in both representations.
- Workspace dependency `tdigest 1.0`, confined to `sciencekit_preprocessing` and `sciencekit_impute`, backing the robust scaler's and the imputer's online quantile sketches.

### Changed

- `SKOperationAttributes` shape fields renamed to `rows_count`/`columns_count` (they are counts); the tracing span field names stay `rows`/`columns`.

- Translated `AGENTS.md` and `openspec/config.yaml` to English (international open-source project).
- `README.md` now links the official GitHub Pages site (`https://neurono-ml.github.io/sciencekit/`) at the top; it is also set as the repository homepage.
- GitHub Actions opencode agent now uses the `opencode/big-pickle` model.
