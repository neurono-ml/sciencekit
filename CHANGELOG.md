# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native model import (spec `native-model-import`): `SKNativeImport<F>` turns a
  neutral state into the matching fitted type — linear/logistic/SGD models,
  decision trees and forests, scalers and k-means — with a descriptor that keeps
  the estimator name, feature count, class table, backend and retraining rule.
  `sk_import_native_model_from_bytes`/`..._from_path` accept Safetensors or the
  JSON debug format after structural format detection, and
  `SKOnnxNativeImporter` maps the documented ONNX subset (affine regressors and
  classifiers, per-column scalers, decision trees and forest regressors) onto the
  same state.
- Typed rejection at the import trust boundary: wrong container magic, truncated
  files, unknown format versions, missing and unexpected tensors, shape and dtype
  mismatches, invalid class tables and unsupported families/operators — every
  failure leaves no partially initialized model, and concurrent conversions are
  deterministic.
- Continued fitting for imported `SKSGDRegressor`, `SKSGDClassifier` and
  `SKLogisticRegression` models; inference-only families reject `partial_fit`
  with typed guidance instead of silently resetting the imported coefficients.


## [0.8.0] — 2026-11-13

Algorithms and capabilities landed since 0.7.0: the SVM family (kernel SVC/SVR
and LinearSVC), ensemble learning (random forests, bagging, stage-wise and
histogram gradient boosting), clustering (DBSCAN and streaming mini-batch
k-means), decomposition (PCA, truncated SVD and incremental out-of-core PCA),
successive-halving model selection, DAG and streaming pipelines, the expanded
interoperability layer (Polars/DataFusion sources, memory-mapped files,
safetensors and JSON debug formats), the OpenCL 3.0 device backend, opt-in
OpenTelemetry export, isolation forests, workspace-wide model export and ONNX
export.

### Added

- Model export coverage (spec `model-export-coverage`): all 47 fitted model
  types implement `SKExportable<F>` over the shared `SKModelState` contract with
  stable dotted tensor names, a single `SK_MODEL_FORMAT_VERSION` and
  resolved-backend metadata. **35 of them round-trip exactly** through
  `SKImportable<F>`, reproducing the original `predict`/`transform` output
  bit-for-bit: standard/min-max/robust scalers, one-hot encoder, polynomial
  features, simple imputer, missing indicator, KNN imputer, linear regression,
  ridge, lasso, elastic net, logistic regression, SGD classifier and regressor,
  k-nearest neighbours (dense and CSR), SVC, SVR, LinearSVC, decision trees,
  random forests, gradient boosting, histogram gradient boosting,
  KMeans/MiniBatchKMeans/DBSCAN, PCA/TruncatedSVD/IncrementalPCA and
  IsolationForest.
- Native import is **conditional** (spec `model-export-coverage`, requirement
  "Native import where reconstructible"). Ten composite model types export a
  state but refuse import with a typed `UnsupportedOperation` carrying
  guidance, because their state does not carry the configuration needed to
  rebuild them: the bagging classifier and regressor (member metadata is not
  embedded), the grid-search classifier and regressor and the successive-halving
  classifier and regressor (winner tensors and selection metadata, but not the
  winner's configuration), the typed pipeline classifier and regressor (the
  configured stage chain holds live estimator values) and the feature union
  (the configured member chain holds live estimator values). The graph and
  streaming pipelines erase their fitted stage types and therefore refuse at
  export with the same typed error. Refusal is the spec-mandated behaviour, never
  a partial model; no model is claimed to round-trip when it cannot.
- ONNX export (spec `onnx-export`): new `sciencekit_onnx` crate with
  `SKToOnnx<F>`, blanket-implemented over `SKExportable<F>`, so any fitted model
  translates its neutral state into an ONNX `ModelProto`. The validated
  `SKOnnxGraphBuilder` writes IR 8 / opset 17 over opset 3 of `ai.onnx.ml`
  using `prost`-declared messages (no `protoc`). Exported families: the three
  scalers, the integer one-hot encoder, linear/ridge/lasso/elastic-net/SGD
  regressors, logistic regression (binary, multinomial and one-versus-rest),
  SVC (binary and one-vs-one multiclass with linear/polynomial/rbf/sigmoid
  kernel expansions), SVR and LinearSVC, decision trees and random forests
  through ONNX-ML tree ensembles, k-nearest neighbours, k-means/mini-batch
  k-means, PCA, truncated SVD and the typed pipelines. Metadata records the producer, estimator name, format version,
  scalar dtype, feature count and the class table. Unsupported families (SVM,
  boosting, isolation forest, DBSCAN) return `UnsupportedOperation` and are
  listed in `sk_onnx_coverage`. Predictions are validated against the reference
  `onnxruntime` by the committed harness
  `crates/sciencekit_onnx/validation/compare_with_onnxruntime.py`.
- ONNX import (spec `onnx-import`): the same `sciencekit_onnx` crate now loads
  an ONNX model and runs it as a generic predictor through `SKOnnxPredictor<F>`
  (built by `SKOnnxPredictorBuilder`), implementing both
  `SKRegressorPredictor<F>` and `SKClassifierPredictor<F>`. A dependency-free
  interpreter executes a documented operator subset over the shared
  `serde-onnx` IR (pinned at revision `0c54592`), including the `ai.onnx.ml`
  tree ensembles. Validation rejects invalid bytes, wrong opsets, unsupported
  operators, missing initializers, external data, shape/dtype mismatches and
  ambiguous outputs with the typed `SKOnnxImportError`. Export was refactored
  onto the same IR, so both directions share one model vocabulary. The
  extended harness `validation/compare_with_onnxruntime.py` imports every
  fixture and reports 36/36 models matching both the Rust prediction and
  `onnxruntime`.
- Shared helpers for the retrofit: `sk_assert_export_round_trip`,
  `sk_states_equal`, the tensor readers and the backend/metadata parameter
  parsers.
- The OpenCL 3.0 device backend (spec `opencl-backend`) behind the `opencl`
  feature of `sciencekit_gpu`, re-exported by the `sciencekit_gpu_opencl`
  package through its `gpu-opencl` feature. It implements `SKComputeBackend`
  over an ICD loader loaded at runtime: deterministic platform/device
  enumeration with capability reporting, `SKOpenClBackend` built only through
  `SKOpenClBackendBuilder`, real device buffers with capacity and device-memory
  validation, explicit accounted uploads/downloads, runtime-compiled and cached
  kernels (pairwise distance for squared-Euclidean/Euclidean/Manhattan/
  Chebyshev, tiled GEMM, elementwise transform and combine, tree-ensemble
  prediction), CPU parity tests on a real device, concurrent dispatch parity,
  and an honest `SKFallbackReason` with structured errors when the feature is
  off, no platform exists or the device cannot be opened. The default build
  pulls in no OpenCL dependency and calls no OpenCL symbol.
- `sciencekit_ensemble` gains histogram gradient boosting (spec
  `histogram-gradient-boosting`): `SKHistGradientBoostingClassifier<F>`,
  `SKHistGradientBoostingClassifierBuilder<F>`,
  `SKHistGradientBoostingClassifierModel<F>`, `SKHistGradientBoostingRegressor<F>`,
  `SKHistGradientBoostingRegressorBuilder<F>`,
  `SKHistGradientBoostingRegressorModel<F>` and
  `SKHistGradientBoostingRegressionLoss` (`SquaredError`, `AbsoluteError`). Fit
  derives at most 255 quantile bins per numeric feature with a dedicated missing
  bin (`SKNumericBinning`, `SKBinnedDataset`, `SKFeatureBinning`), accumulates
  gradient/Hessian histograms in parallel across features
  (`sk_accumulate_histogram`), finds numeric and categorical splits with the
  Newton gain (`sk_scan_numeric_feature`, `sk_scan_categorical_feature`,
  `sk_split_gain`), subtracts the smaller child histogram from its parent
  (`sk_subtract_histogram`) and grows trees leaf-wise under `maximum_leaf_nodes`
  and `minimum_samples_leaf` (`sk_grow_histogram_tree`, `SKHistogramTree`,
  `SKHistogramNode`). Categorical partitions use `bitvec` membership masks with a
  documented unknown-category direction, missing values learn a `default_left`
  direction per split, and prediction bins dense or CSR queries without
  densification. Early stopping uses a seeded validation fraction and
  `n_iter_no_change`, and every operation records the configured backend.
- Feature-gated tabular sources in `sciencekit_interop` (spec
  `tabular-sources`): the materialized `SKTabularDataset<F>` with a projected
  `SKTabularSchema`, a dense or sparse `SKFeatureBlock` and an owned
  `SKNominalMatrixOwned` categorical block; `SKColumnKind` (`Numeric`,
  `Categorical`, `Boolean`, `Ignored`) and the builder-validated
  `SKTabularSourceConfig`; `SKTabularBatch::to_tabular_dataset` applying one
  missing-value policy (numeric nulls to `NaN`, categorical nulls to
  `SK_CATEGORICAL_MISSING_CODE`) and rejecting unsupported dtypes by column.
  The optional `polars` feature converts Polars 0.46 `DataFrame` values and
  exposes a reopenable sliced `LazyFrame` batch source; the optional
  `datafusion` feature converts DataFusion 46 `DataFrame`/`RecordBatch` values,
  streams a physical plan as record batches, and reads Parquet files through a
  temporary session. Both engines keep their own Arrow vocabulary behind the
  neutral column contract, and the default build resolves neither dependency.
- `sciencekit_interop` gains the optional `memmap` feature (spec
  `memory-mapped-sources`): read-only `memmap2` mappings exposed through
  `SKMappableSource` (O(1) borrowed rows) and `SKLazySource` (bounded owned
  blocks). `SKRawBinaryMappableSource<F>` maps a headerless fixed-width matrix
  driven by an explicit or sidecar `SKRawBinaryDescriptor` (`f32`, `f64`,
  `i64`); `SKNumpyMappableSource<F>` maps C-order `.npy` `f4`/`f8` arrays
  zero-copy and offers an explicit `open_with_cast` materialising path for
  `i64` and mismatched float dtypes; `SKArrowIpcMappableSource<F>` reads only
  the Arrow IPC footer, indexes record batches by cumulative row offsets and
  decodes a single batch per row lookup behind a shared cache. Opening
  validates little-endian data, dtype compatibility, exact file length,
  payload alignment and header framing, and reports structured
  `MissingMappedFile`, `MappedFilePermissionDenied`, `UnsupportedMappedDtype`,
  `NonLittleEndianMappedData`, `MappedLengthMismatch`, `MalformedMappedHeader`,
  `MisalignedMappedData` and `UnreadableMappedBatch` errors naming the file.
  Companion tests cover random/sequential equivalence, `f32`/`f64`,
  concurrent `Send + Sync` readers, large files with bounded batches,
  Fortran order, truncated files and misaligned headers.
- `sciencekit_math` gains the Mercer and similarity kernel catalogue (spec
  `kernel-functions`): `SKKernel` (`Linear`, `Polynomial`,
  `RadialBasisFunction`, `Sigmoid`, `Precomputed`), `SKKernelGamma<F>`
  (`Scale`, `Auto`, `Value`), `SKKernelParameters<F>`,
  `SK_KERNEL_PARALLEL_GRAIN`, `sk_validate_kernel_hyperparameters`,
  `sk_resolve_kernel_gamma`, `sk_resolve_kernel_parameters`,
  `sk_feature_population_variance`, `sk_linear_kernel_matrix`,
  `sk_polynomial_kernel_matrix`, `sk_radial_basis_function_kernel_matrix`,
  `sk_sigmoid_kernel_matrix`, `sk_precomputed_kernel_matrix`,
  `sk_validate_precomputed_query_kernel`, `sk_kernel_matrix` and
  `sk_kernel_matrix_sequential`. Every feature kernel dispatches once across all
  four dense/CSR representation pairs without densifying sparse inputs, clamps
  floating-point squared-distance cancellation in the RBF path, and evaluates
  independent rows in parallel with rayon.
- New crate `sciencekit_svm` delivering kernel support-vector machines (spec
  `support-vector-machines`): `SKSVC<F>`, `SKSVCBuilder<F>`, `SKSVCModel<F>`,
  `SKBinarySVCModel<F>`, `SKClassWeight<F>` (`None`, `Balanced`, `Explicit`,
  `LabelMap` / `from_labels`), `SKSVR<F>`, `SKSVRBuilder<F>` and
  `SKSVRModel<F>`, backed by a deterministic pure-Rust second-order SMO solver
  (`SKSmoState`, `SKSmoWorkingPair`, `SKSmoTracker`, `SKSmoSolution`,
  `SKSmoOptions`, `SKKernelRowCache`, `SKBinarySvcSmoResult`,
  `SKEpsilonSvrSmoResult`, `sk_solve_smo_quadratic_program`,
  `sk_solve_binary_c_svc`, `sk_solve_epsilon_svr`), owned dense/CSR
  support-vector storage (`SKOwnedFeatureBlock`), one-vs-one multiclass fitting,
  deterministic vote resolution (`sk_resolve_one_vs_one_votes`) and
  `decision_function` / `pairwise_decision_function` margins, plus
  deterministic five-fold Platt scaling and Wu-Lin-Weng pairwise coupling
  (`SKPlattParameters`, `sk_fit_platt_sigmoid`, `sk_calibrate_binary_platt`,
  `sk_couple_pairwise_probabilities`,
  `sk_couple_pairwise_probabilities_with_budget`).
- `SKLinearSVC<F>` in `sciencekit_svm` (spec
  `linear-support-vector-classifier`): a builder-only L2-regularized linear
  support-vector classifier (`C`, `loss`, `class_weight`, `include_intercept`,
  `tolerance`, `maximum_number_of_iterations`, `backend`, `execution_mode`)
  backed by a LIBLINEAR-style dual coordinate-descent solver for the L1-hinge
  and L2-squared-hinge losses. It exposes `SKLinearSVCLoss` (`Hinge`,
  `SquaredHinge`), `SKLinearSVCHead<F>` and `SKLinearSVCModel<F>` with
  `decision_function`, deterministic `predict_labels` and an explicit
  `UnsupportedOperation` from `predict_probabilities` (no calibrated
  probability model). Fitting runs one-vs-rest heads in parallel with
  deterministic order, accepts dense or borrowed CSR designs without
  densifying, honours `None`/`Balanced`/explicit class weights, and reports
  `NotConverged { iterations }` when the iteration budget is exhausted.
- Stage-wise gradient boosting in `sciencekit_ensemble` (spec `gradient-boosting`):
  builder-only `SKGradientBoostingClassifier<F>` / `SKGradientBoostingRegressor<F>` with
  `number_of_estimators` (default `100`), `learning_rate` (default `0.1`), `maximum_depth`
  (default `3`), `minimum_samples_leaf` (default `1`), `maximum_features` (default `All`),
  `subsample` (default `1.0`), `l2_regularization` (default `0.0`), `random_seed`
  (default `0`), `validation_fraction` (default `0.1`), `n_iter_no_change` (default off)
  and `tolerance` (default `1e-4`), plus the mandatory `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`) parameters recorded in every `fit` and `predict`
  span and carried into the fitted model. The classifier covers `LogLoss` with a logistic
  sigmoid link for two classes and a softmax link for multiclass (one tree per class per
  stage); the regressor covers `SquaredError`, `AbsoluteError` and `Huber` with a
  configurable `huber_delta`. Every stage fits a regression tree to the negative gradient
  and writes damped Newton leaves `-Σg / (Σh + λ)`, starting from the loss-optimal constant
  prediction; `subsample < 1.0` draws rows without replacement per stage and
  `n_iter_no_change` early-stops on a held-out validation fraction. Integer and nominal
  targets are canonicalized through an `SKLabelTable`, dense and zero-copy CSR fits and
  predictions never densify a sparse input, and the fitted
  `SKGradientBoostingClassifierModel` / `SKGradientBoostingRegressorModel` expose
  `number_of_stages`, the per-stage trees, `initial_prediction(s)`, `learning_rate`,
  `classes`, `number_of_classes`, `feature_importances` and per-stage feature importances.
- Streaming pipelines (spec `streaming-pipeline`): `sciencekit_pipeline::streaming`
  adds `SKStreamingTransformer`, `SKStreamingPipeline` /
  `SKStreamingPipelineBuilder`, `SKStreamingPipelineConfigured`,
  `SKFittedStreamingPipeline` and `SKStreamingFitReport`. A pipeline composes
  streaming stages erased behind object-safe adapters, fits them in order by
  reading an `SKSourceProvider` through the shared prefetch executor (each stage
  seeing the previous stage's transformed batches), and exposes a per-batch
  `transform_batch` that preserves dense/CSR without densifying. A reopenable
  provider replays each configured `epochs` pass; a one-shot provider is
  materialized once within `memory_budget_bytes`, and an over-budget
  materialization returns `SKError::BatchBufferOversize`. The batch finalizer
  accumulates the transformed blocks within the same budget and fits an
  `SKUnsupervisedFit` estimator, preserving CSR when every block is sparse.
  Source failures surface the central taxonomy error, stage failures carry the
  zero-based stage index, multi-epoch requests on a one-shot provider return a
  structured `SKError::UnsupportedOperation`, empty stage lists and zero
  epochs/budget return `SKError::InvalidHyperparameter`, and the fitted
  pipeline is `Send + Sync` for concurrent transformation. Every `fit` and
  `transform` span records the configured `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`).

- `sciencekit_decomposition` gains uncentered low-rank Singular Value
  Decomposition (spec `truncated-svd`): `SKTruncatedSVD<F>`, a builder-only
  estimator with `number_of_components` (default `2`), `algorithm`
  (`SKTruncatedSVDAlgorithm::Randomized` default; `Arpack` reserved and
  rejected as an explicit non-goal), `number_of_iterations` (default `5`),
  `tolerance` (default `0.0`), `random_seed` (default `0`), `backend` (default
  `Faer`) and `execution_mode` (default `Automatic`). Fitting delegates to the
  seeded randomized range finder (`SKRandomizedSvdBuilder`) on the uncentered
  design without subtracting feature means or densifying sparse CSR views,
  completes rank-deficient singular directions via deterministic Gram-Schmidt
  so `components` remains row-orthonormal, computes population
  `explained_variance` over projected training rows and divides by the non-densifying
  total input column variance for `explained_variance_ratio` (clamping rounding
  overshoot to `<= 1.0`), and returns `SKTruncatedSVDModel<F>` exposing
  `components`, `singular_values`, `explained_variance`,
  `explained_variance_ratio`, `number_of_components`, `number_of_features`,
  `algorithm`, `backend` and `execution_mode`. `SKTruncatedSVDModel<F>`
  implements `SKFeatureTransformer<F>` plus inherent `transform` and
  `inverse_transform` across all dense and CSR combinations using fixed 256-row
  chunks for thread-count-invariant evaluation, `SKTruncatedSVD::fit_transform`
  runs under a single `FitTransform` span, and `SK_TRUNCATED_SVD_PARITY_TOLERANCE`
  (`1e-6`) documents the dense/CSR equivalence and rank-`k` reconstruction
  bound. Companion and acceptance tests cover `f32`/`f64`, rank-1/rank-2 and
  constant fixtures, non-densification on wide CSR tables, validation guards,
  1- vs 4-thread Rayon invariance, `Send + Sync` concurrent sharing, and
  reconstruction metrics (`sk_mean_squared_error`, `sk_r2_score`) via
  `sciencekit_metrics`.
- `sciencekit_telemetry` sub-crate (spec `telemetry-export`): opt-in
  OpenTelemetry trace, log and metric export pipelines and process-wide
  lifecycle over `opentelemetry 0.32`, `opentelemetry_sdk 0.32`,
  `opentelemetry-otlp 0.32`, `opentelemetry-appender-tracing 0.32` and
  `tracing-opentelemetry 0.33`. Builder-validated `SKTelemetryConfiguration`
  (`SKTelemetryConfigurationBuilder`), `SKTelemetryBatchConfiguration`
  (`SKTelemetryBatchConfigurationBuilder`), `SKTelemetryResource`,
  `SKTelemetryTransport { Grpc, Http }` and
  `SKTelemetrySampler { AlwaysOn, AlwaysOff, TraceIdRatioBased, ParentBased }`
  validate service identity, HTTP/HTTPS endpoints, ASCII header tokens, timeouts,
  sampling ratios and batch queue bounds up front with `SKError::InvalidHyperparameter`.
  `SKTelemetryPipeline` and `sk_build_otlp_pipeline` (gated behind off-by-default
  `otlp-grpc`, `otlp-http` and `telemetry` features) bridge `sciencekit.operation`
  spans with error-status enrichment, structured `tracing` events with trace
  correlation, and bounded-cardinality operation metrics
  (`sciencekit.operation.duration`, `sciencekit.operation.count`,
  `sciencekit.operation.rows`, `sciencekit.operation.columns`,
  `sciencekit.operation.errors`). `SKTelemetry` manages once-per-process `init` /
  `init_with_exporters`, mutex-serialized `force_flush` honoring `flush_timeout`,
  `shutdown` and drop-safe detachment while leaving uninitialized default builds
  completely silent with zero active exporters and zero network traffic.

- Successive-halving hyperparameter search in `sciencekit_model_selection` (spec
  `halving-grid-search`): builder-only `SKHalvingGridSearchClassifier<F, E>` /
  `SKHalvingGridSearchRegressor<F, E>` with `candidates`, `splitter`, `scoring`, `factor`
  (default `3`), `minimum_resource` (default `1`), `maximum_resource` (default all rows),
  `resource` (`SKHalvingResource::NumberOfSamples` default), `aggressive_elimination`
  (default `false`), `random_seed` (default `0`), `refit` (default `true`), `error_score`,
  `groups`, `backend` and `execution_mode`. Fitting evaluates the survivors on a seeded
  per-round row subsample, keeps the best `ceil(n / factor)`, grows the resource
  geometrically (`10, 30, 90, …`), and stops at one survivor or the maximum resource;
  ranking is descending by mean score with ties broken by original candidate index and `NaN`
  ranked last. Subsamples preserve class proportions when the splitter requires
  stratification and group proportions when `groups` are supplied, and never densify sparse
  CSR input. The fitted `SKHalvingGridSearch…Model` exposes `elimination_trace()`,
  `number_of_rounds()`, `candidate_count_trajectory()`, `resource_schedule()`,
  `best_candidate_index()`, `best_score()`, `best_parameters()`, `best_model()` and
  `is_refitted()`, delegates classification/regression prediction and `export_state()` to the
  winner refitted on all rows, and rejects prediction and export with
  `UnsupportedOperation` when `refit(false)`. Free functions
  `sk_halving_resource_schedule` / `sk_halving_candidate_schedule` and the
  `SKHalvingRound` trace type make the schedule and every elimination step inspectable.
- Runtime directed-acyclic-graph pipelines (spec `pipeline-graph`):
  `sciencekit_pipeline::graph` adds `SKPipelineGraph` /
  `SKPipelineGraphBuilder` over the `petgraph` workspace dependency,
  `SKNodeIdentifier`, `SKPipelineGraphNodeKind`, `SKNodeBlockStore`,
  `SKPipelineGraphModel`, the `SKPipelineGraphClassifier` /
  `SKPipelineGraphRegressor` wrappers, the type-erased `SKBoxedFinalModel` and
  the node-attributed `SKPipelineGraphNodeError`. The builder is the only
  constructor (`SKPipelineGraph::start` / `SKPipelineGraph::build`) and carries
  the mandatory `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`) parameters recorded in every `fit`, `transform` and `predict`
  span. `build` validates acyclicity (naming the cycle), exactly one final
  estimator that is a sink, and full input reachability, rejecting violations
  with `SKError::InvalidHyperparameter`. Fit executes nodes in a deterministic
  level order (ascending node identifier within a level); nodes at the same
  level never depend on each other. Nodes with zero incoming edges receive the
  caller's block, nodes with one incoming edge receive it unchanged, and nodes
  with several incoming edges concatenate them horizontally in exact
  `connect(source, target)` insertion order, promoting to CSR without
  densifying any sparse input. Root, single-edge and final-estimator widths are
  re-checked during prediction and mismatches surface as node-attributed
  `SKError::ShapeMismatch` or `SKError::UnsupportedRepresentation`. Every stage
  runs over `SKFeatureBlock`, so dense, CSR and mixed graphs fit and predict
  without implicit densification, and the fitted graph is `Send + Sync` for
  concurrent inference. Scalar compatibility is enforced at compile time: a
  graph shares one `SKFloat`, so a stage implemented for another scalar cannot
  be inserted.
- `SKMiniBatchKMeans<F>` in `sciencekit_cluster` (spec `mini-batch-k-means`):
  builder-only mini-batch k-means with `number_of_clusters` (default `8`),
  `initialization` (`KMeansPlusPlus` default, `Random`), `number_of_initializations`
  (default `3`), `batch_size` (default `1024`), `initialization_batch_size`
  (default `3 × batch_size`), `maximum_number_of_iterations` (default `100`),
  `maximum_number_of_epochs` (default `1`), `tolerance` (default `0`, disabling
  early stopping), `reassignment_ratio` (default `0.01`), `random_seed`
  (default `0`), `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`). In-memory `fit`/`fit_batch` and incremental `partial_fit`
  update dense centers with the exact per-center learning rate
  `1 / (count[center] + 1)` through a closed form that touches only stored CSR
  entries, initialize from a deterministic bounded sample with best-of-`n_init`
  inertia selection, and deterministically reassign the most distant batch point
  to centers below `reassignment_ratio × mean_count`. `fit_streaming` drives an
  `SKSourceProvider` through the shared prefetch executor, opening a fresh source
  per epoch for reopenable providers up to `maximum_number_of_epochs` and
  stopping early when a positive tolerance is met. The fitted
  `SKMiniBatchKMeansModel<F>` exposes dense `cluster_centers`, per-center
  `counts`, typed non-noise training `labels` for in-memory fits, `inertia`,
  `batches_processed`, `epochs_consumed` and `converged`, implements
  `SKUnsupervisedFit`, `SKClustererPredictor<F>` (`predict_clusters`/`predict`),
  Euclidean `transform`, `fit_predict` and format-agnostic
  `SKExportable`/`SKImportable` state emission. Dense and CSR parity is verified
  byte-for-byte without densification, along with thread-count-invariant
  determinism, lock-free concurrent prediction, f32/f64 independence, validation
  and acceptance suites (50 000-row one-shot and multi-epoch streams, full
  KMeans comparison and clustering metrics).

- Extended binary Safetensors internal model format (spec `safetensors-format`):
  `sciencekit_interop::safetensors` adds `SKSafetensorsWriter` /
  `SKSafetensorsWriterBuilder`, `SKSafetensorsReader` /
  `SKSafetensorsReaderBuilder`, `SKShardPlan` / `SKShardPlanBuilder`,
  `SKShardIndex`, `SKAppendOutcome`, `SKSafetensorsCompression` (`None` default,
  `Gzip`, `Brotli`, `Snappy`), `SK_SAFETENSORS_FORMAT_VERSION` (`1`),
  `SK_SAFETENSORS_PADDING_TENSOR_NAME` (`"__padding__"`) and convenience entry
  points (`sk_write_model_state_safetensors`,
  `sk_write_model_state_safetensors_file`, `sk_read_model_state_safetensors`,
  `sk_read_model_state_safetensors_file`, `sk_export_model_safetensors`,
  `sk_export_model_safetensors_file`, `sk_import_model_safetensors`,
  `sk_import_model_safetensors_file`, `sk_append_safetensors_tensor`,
  `sk_read_safetensors_shard_index`, `sk_safetensors_index_path` and
  `sk_detect_safetensors_compression`) over `sciencekit_common::SKModelState`.
  Single-file output is standard HuggingFace Safetensors (`8`-byte little-endian
  header length, UTF-8 JSON header carrying `sciencekit.*` metadata, class
  labels and typed hyperparameters, plus safe little-endian `F32`, `F64`, `I64`
  and `BOOL` tensor payloads without `unsafe`). Large states split into
  immutable shards plus an atomically renamed `.safetensors.index.json` manifest
  that the reader merges transparently; reserved header (`0x20`) and payload
  (`__padding__`) space enables in-place seek appends without rewriting existing
  tensor bytes, spilling automatically to a new shard when exhausted; and
  optional Cargo features `compression-gzip`, `compression-brotli` and
  `compression-snappy` wrap whole files (`.safetensors.gz`,
  `.safetensors.brotli`, `.snappy.safetensors`) with magic-byte auto-detection
  and explicit `UnsupportedOperation` guidance when disabled.

- `sciencekit_decomposition` gains out-of-core Principal Component Analysis
  (spec `incremental-pca`): `SKIncrementalPCA<F>`, `SKIncrementalPCABuilder<F>`
  and `SKIncrementalPCAModel<F>`, builder-only with `number_of_components`
  (default `None`, meaning `min(n_samples, n_features)` up to the feature
  count), `batch_size` (default `1024`), `whiten` (default `false`),
  `maximum_number_of_epochs` (default `1`), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`). Dense batches accumulate exact
  `(samples_seen, mean, scatter)` statistics via the Chan-Golub-LeVeque merge
  across fixed 256-row parallel chunks without retaining rows; sparse CSR
  batches accumulate Neumaier-compensated column sums and Gram products
  (`Gram - n * mean * mean^T`) without ever densifying sparse storage.
  Finalization divides the scatter matrix by `n - 1`, runs the configured
  backend's symmetric eigensolver (`eigh`), sorts eigenvalues non-increasingly
  and orients principal component rows deterministically to match `SKPCA`.
  `fit` chunks in-memory dense/CSR matrices into `batch_size` blocks (coalescing
  undersized trailing remainders), `SKIncrementalPCAModel::partial_fit` chains
  dense and CSR batch updates, `fit_streaming` drives `SKSourceProvider`
  through the shared prefetch executor across one pass for one-shot providers
  and `maximum_number_of_epochs` passes for reopenable providers, and
  `SKIncrementalPCAModel` implements `SKFeatureTransformer`, `transform`,
  `inverse_transform` and format-agnostic `SKExportable`/`SKImportable` state
  round-tripping.

- `sciencekit_outlier` sub-crate and `sciencekit_common` outlier prediction
  contract (specs `outlier-prediction` and `isolation-forest`): one-byte typed
  `SKOutlierLabel { Inlier, Outlier }` with explicit external `to_i64` /
  `from_i64` conversions (`1` / `-1`), `SKOutlierPredictor<F>` (`offset`,
  `score_samples`, `decision_function`, `predict`), and builder-only
  `SKIsolationForest<F>` (`SKIsolationForestBuilder<F>`) with
  `number_of_trees` (default `100`), `maximum_samples`
  (`SKIsolationMaximumSamples::Auto` default, meaning `min(256, n_samples)`,
  plus `Count` and `Fraction`), `contamination` (`SKContamination::Auto`
  default with fixed offset `-0.5`, or `Fraction` in `(0, 0.5]` selecting the
  exact linear-interpolated training quantile), `maximum_features` (default
  `1.0`), `bootstrap` (default `false`), `random_seed` (default `0`), `backend`
  (default `Faer`) and `execution_mode` (default `Automatic`). Fitting derives
  independent per-tree SplitMix64 sub-seeds (`SK_RANDOM_DOMAIN_TREE` /
  `SK_RANDOM_DOMAIN_STAGE`), grows random isolation trees up to
  `ceil(log2(subsample_size))` over the shared `sciencekit_tree::SKTree` node
  layout (`SKTree::leaf_and_depth_for`), gathers sparse CSR subsamples directly
  into compact CSC columns without densification, and returns a `Send + Sync`
  `SKIsolationForestModel<F>` exposing `anomaly_scores`
  (`2^(-E[h(x)] / c(n))`), `score_samples` (`-anomaly_score`),
  `decision_function` (`score_samples - offset`), typed `predict`, `offset`,
  `subsample_size`, `feature_count`, `number_of_trees`, `tree`, `backend` and
  `execution_mode`, alongside `SKIsolationForest::fit_predict`. Companion and
  acceptance suites (up to 20 000 rows) verify hand-computed harmonic path
  lengths, four-way dense/CSR equivalence, scalar invariance (`f32`/`f64`),
  thread-count-invariant determinism, lock-free concurrent serving and
  evaluation metrics (`sk_roc_auc_score`, `sk_average_precision_score`,
  `sk_accuracy_score`, `sk_f1_score`) via `sciencekit_metrics`.

- `sciencekit_ensemble` gains generic bootstrap-aggregating meta-estimators
  (spec `bagging-ensembles`): `SKBaggingClassifier<B>` and
  `SKBaggingRegressor<B>`, builder-only (`SKBaggingClassifierBuilder<B>`,
  `SKBaggingRegressorBuilder<B>`) over any configured base estimator `B`
  satisfying `SKSupervisedFit<F>` whose fitted model implements
  `SKClassifierPredictor<F>` or `SKRegressorPredictor<F>`, with
  `number_of_estimators` (default `10`), `maximum_samples` (`All` default,
  `Count`, `Fraction`), `maximum_features` (`All` default, `Count`, `Fraction`,
  `Sqrt`, `Log2`), `bootstrap` (default `true`), `bootstrap_features` (default
  `false`), `oob_score` (default `false`), `random_seed` (default `0`),
  `backend` (default `Faer`) and `execution_mode` (default `Automatic`). Fit
  reuses the shared seeded bootstrap sampler and derives independent
  per-estimator feature-subset seeds to train lock-free parallel rayon slots
  without densifying borrowed CSR matrices, surfaces base estimator rejections
  through `SKBaggingError<E>` (`Central` vs `Base`), routes inference and
  out-of-bag accuracy / R² evaluation exclusively through each member's stored
  feature subset without leaking excluded columns, averages canonical class
  probabilities with smallest-index tie-breaking or continuous responses across
  deterministic row chunks, and implements `SKExportable<F>` whenever the base
  model is exportable.

- `sciencekit_pipeline` gains parallel feature branching and column-wise
  concatenation (spec `feature-union`): `SKFeatureUnion` and
  `SKFeatureUnionBuilder` (`build`/`start` for compile-time tuples up to eight
  heterogeneous members via `SKFeatureUnionStages`, plus `boxed` with
  `Vec<Box<dyn SKBlockTransformer<F>>>` for dynamic construction),
  `SKFeatureUnionModel` exposing `number_of_members`, `member_widths`,
  `number_of_features`, `number_of_output_features`,
  `number_of_input_features`, `stages`, `fitted_stages`, `backend` and
  `execution_mode`, CSR-preserving horizontal concatenation (all-dense outputs
  stay dense; any sparse member promotes the union output to CSR without
  densifying sparse inputs), member-indexed error context via
  `SKFeatureUnionMemberError` (`fit_checked`/`transform_checked`), and direct
  `SKPipelineTransformer` integration for both `SKFeatureUnion` and nested
  `SKPipelineBuilder` branches.

- `sciencekit_decomposition` sub-crate (spec `principal-component-analysis`):
  `SKPCA<F>`, a builder-only Principal Component Analysis estimator with
  `number_of_components` (default `None`, meaning `min(n_samples, n_features)`),
  `svd_solver` (`Auto` default, `Full`, `Randomized`, `CovarianceEigh`),
  `whiten` (default `false`), `tolerance` (default `0.0`),
  `number_of_power_iterations` (default `5`), `allow_densification` (default
  `false`), `random_seed` (default `0`), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`). Fitting computes per-feature empirical
  means and total unbiased centered variance, resolves `Auto` deterministically
  (`Full` on high component fractions or small dense designs, `CovarianceEigh`
  on tall-skinny dense designs where `n_samples > 10 * n_features` and
  `n_features <= 1000`, and `Randomized` on larger truncated or sparse designs),
  orients every principal component row so its largest-magnitude coordinate is
  non-negative, and returns a distinct `SKPCAModel<F>` exposing `components`,
  `explained_variance`, `explained_variance_ratio`, `singular_values`, `mean`,
  `resolved_solver`, `number_of_components`, `number_of_features`, `whiten`,
  `backend` and `execution_mode`. The `Randomized` solver fits sparse CSR views
  through implicit centering (`SKRandomizedSvdBuilder`) without materializing a
  dense centered matrix; `Full` and `CovarianceEigh` reject sparse input with
  `UnsupportedRepresentation` naming `Randomized` and `SKTruncatedSVD` unless
  `allow_densification(true)` is set, in which case a warning reports the
  estimated dense byte cost. `SKPCAModel<F>` implements `SKFeatureTransformer<F>`
  and inherent `transform` and `inverse_transform` over dense and CSR views
  without densification, with floored whitening scales and row-chunked parallel
  projection; `SKPCA::fit_transform` runs under a single `FitTransform` span.
  Companion and acceptance tests cover `f32`/`f64`, all solvers, component
  boundaries, whitening, validation errors, thread-count-invariant determinism,
  concurrent sharing and reconstruction metrics (`sk_mean_squared_error`,
  `sk_r2_score`) via `sciencekit_metrics`.

- `sciencekit_cluster` gains the density-based spatial clusterer (spec
  `dbscan`): `SKDBSCAN<F>`, builder-only with `eps` (default 0.5),
  `minimum_samples` (default 5), `metric` (`Euclidean` default, `Manhattan`,
  `Chebyshev`, `Cosine`), `algorithm` (`Auto` default, `KdTree`, `BallTree`,
  `BruteForce`), `leaf_size` (default 40), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`). Fit obtains each row's inclusive
  `eps` neighbourhood through `sciencekit_neighbors` (`SKKDTree` or
  `SKBallTree` for dense tree-compatible metrics, `SKBruteForceSearcher` for
  sparse CSR or cosine distance, never densifying sparse storage) with the row
  itself always included, identifies core samples (`>= minimum_samples`
  neighbours), and expands density-connected components deterministically in
  ascending training-index order so the smallest core index in each component
  seeds its sequential cluster identifier and shared border points join the
  earliest-seeded cluster; unreachable non-core points receive the explicit
  `SKClusterLabel::Noise` variant. `SKDBSCANModel<F>` is `Send + Sync`, exposes
  `labels`, `core_sample_indices`, `number_of_clusters`, `number_of_features`,
  `eps`, `minimum_samples`, `metric`, `search_strategy`,
  `training_representation`, `backend` and `execution_mode`, and implements
  `SKClustererPredictor<F>` through a documented nearest-core heuristic that
  assigns each query row to the cluster of its nearest fitted core sample when
  within `eps` and `SKClusterLabel::Noise` otherwise across all four dense/CSR
  combinations. Construction and fit/predict validate empty inputs,
  non-positive or non-finite `eps`, zero `minimum_samples` or `leaf_size`,
  feature-count mismatches, non-finite values, CSC views and incompatible
  tree/metric or tree/representation pairs.

- `sciencekit_ensemble` introduces Breiman random forests (spec
  `random-forests`): `SKRandomForestClassifier<F>`, `SKRandomForestRegressor<F>`
  and `SKMaximumSamples<F>`, builder-only with `number_of_trees` (default 100),
  `criterion` (`Gini` / `Entropy` for classification, `SquaredError` /
  `AbsoluteError` for regression), `maximum_depth`, `minimum_samples_split`,
  `minimum_samples_leaf`, `maximum_features` (`Sqrt` default), `bootstrap`
  (default true), `maximum_samples` (`All` default, `Count`, `Fraction`),
  `random_seed` (default 0), `class_weight` (classifier), `oob_score` (default
  false), `backend` (default `Faer`) and `execution_mode` (default `Automatic`).
  Fit grows one CART tree per bootstrap sample in parallel lock-free slots via
  rayon, deriving distinct deterministic per-tree split and bootstrap seeds from
  `random_seed` and the tree index so the fitted forest is identical across
  repeated fits and rayon thread counts. Prediction averages canonical class
  probabilities with smallest-index tie-breaking (classifier) or leaf responses
  (regressor), exposes normalized mean feature importances, and computes
  out-of-bag accuracy or R² through `sciencekit_metrics` exclusively over rows
  excluded from each tree's bootstrap sample (`UnsupportedOperation` when
  disabled or when zero rows fall out of bag). Dense and borrowed CSR designs
  have full representation parity on fit and prediction without densifying
  sparse inputs.

- Human-readable JSON debug model format (spec `json-debug-format`):
  `sciencekit_interop::json` adds `SKJsonDebugWriter` /
  `SKJsonDebugWriterBuilder`, `SKJsonDebugReader` / `SKJsonDebugReaderBuilder`,
  `SKJsonFormatMode` (`Pretty` default, `Compact`),
  `SK_JSON_DEBUG_FORMAT_VERSION` (`1`) and the convenience entry points
  `sk_write_model_state_json`, `sk_write_model_state_json_pretty`,
  `sk_write_model_state_json_compact`, `sk_write_model_state_json_file`,
  `sk_read_model_state_json`, `sk_read_model_state_json_file`,
  `sk_export_model_json` and `sk_import_model_json` over
  `sciencekit_common::SKModelState`. Both builders follow the shared builder
  contract with `backend` (default `Faer`) and `execution_mode` (default
  `Automatic`), and every write or read emits a structured `tracing` span. The
  schema records `format_version`, `estimator`, scalar `dtype` (`"f32"` or
  `"f64"`), `feature_count`, optional `class_labels`, untagged typed
  `hyperparameters` and `BTreeMap`-ordered named `tensors` carrying `dtype`
  (`"F32"`, `"F64"`, `"I64"` or `"Bool"`), `shape` and flat `values` for clean
  line-oriented diffs and exact `f32`/`f64` round-trips. Unknown fields within
  major version `1` are ignored for forward compatibility, while malformed JSON
  (`SKError::Conversion`), unknown major versions
  (`SKError::UnsupportedRepresentation`), non-finite floats
  (`SKError::NonFiniteValue`), duplicate tensor or hyperparameter keys, invalid
  or mismatched dtypes (`SKError::InvalidHyperparameter`) and tensor
  shape/value-count disagreements (`SKError::ShapeMismatch`) are rejected up
  front before materializing model state.

### Fixed

- `sciencekit_interop` `memmap` NumPy sources reject `.npy` byte lengths whose element-count multiplication overflows the address space, using `checked_mul` and reporting `SKError::MalformedMappedHeader` instead of wrapping into an undersized mapping.
- `sciencekit_interop` `memmap` Arrow IPC row access validates crafted batch metadata with checked arithmetic and a safe shared cache, rejecting out-of-range offsets with structured errors rather than reading past a batch.
- New `hostile_input_tests` and Arrow IPC layout tests exercise truncated raw files, absurd `.npy` dimensions, wrong-shape descriptors and tampered batch metadata as typed errors.

## [0.7.0] — 2026-11-11

Arrow interop, the GPU compute contracts, the randomized SVD kernel and the language bindings moving to their own repositories.

### Added

- Randomized SVD kernel (spec `randomized-svd`): `sciencekit_math::decomposition`
  gains `sk_randomized_svd` and the builder `SKRandomizedSvdBuilder`, returning
  `SKRandomizedSvdResult` (left singular vectors, non-increasing singular
  values and right singular vectors). The range finder draws a seeded Gaussian
  test basis from the shared deterministic generator, orthonormalizes the
  projected basis with a thin Householder QR, refines it with the configured
  number of power iterations and decomposes the small projected matrix. Dense
  and CSR inputs are supported and dispatch once per operation; a sparse input
  is never densified and the dense intermediates stay at
  `(k + oversampling) × max(rows, columns)` cells. An optional centering vector
  centers the matrix implicitly, without materializing a centered copy, for
  dense and sparse inputs alike. The same seed always yields the same
  factorization, independently of the thread count, and the singular-vector
  signs follow a fixed convention (the largest-magnitude entry of the left
  singular vector is positive).

- `sciencekit_interop` gains the Apache Arrow adapter behind the optional,
  default-off `arrow` feature (spec `arrow-columnar-interop`, change
  `arrow-interop`). It imports Arrow record batches, chunked columns
  (`Vec<ArrayRef>` — Arrow's Rust implementation removed `Table` and
  `ChunkedArray` upstream) and batch readers into the neutral tabular seam,
  preserving column names, order, logical types and row count. Arrow validity
  bitmaps feed the shared missing-value policy: floating-point nulls become the
  NaN sentinel or the constant fill, nullable integer columns retain their
  64-bit values across import and promote at materialization with an exactness
  guard for the target scalar (`f32` or `f64`), the categorical policy reserves
  the missing code, and the reject policy names the Arrow field. Primitive `f32`, `f64` and `i64` columns
  without nulls expose borrowed, pointer-identical views; text columns expose a
  borrowed UTF-8 view; dictionary columns become category codes with a
  deterministic, reversible label table. The Arrow-to-neutral schema mapping and
  its reverse preserve roles and disambiguate category codes through field
  metadata, while nested, temporal, binary, decimal, `UInt64` and `Float16`
  columns are rejected with a structured error naming the field and a conversion
  suggestion. Reopenable and one-shot tabular sources stream a
  `RecordBatchReader` in batches of the configured size and mark the final
  batch, and path-based sources open Parquet (row-group aware, with projection
  pushdown), CSV (inferred or explicit schema, header and delimiter options),
  Arrow IPC/Feather files and streams, and record-oriented JSON/NDJSON, all
  reopenable and yielding structured errors for truncated or malformed files.
  The Arrow C Data Interface is implemented in both directions for arrays and
  batch streams with ownership transferred exactly once, release-once tests and
  dictionary/text round trips. Exports cover predictions as floating arrays,
  dense and supervised blocks as record batches, and CSR blocks as row pointers,
  column indices, stored values and shape metadata, for both `f32` and `f64`.
  Arrow versions are pinned to 58.3, the last release whose MSRV is Rust 1.85.

- `sciencekit_gpu` — the device-compute contracts shared by the OpenCL, CUDA
  and ROCm backends (spec `gpu-compute-contracts`), re-exported at the crate
  root. `SKDeviceInfo` and `SKDeviceCapabilities` describe an enumerated
  device, its memory, scalar types and supported kernels; `SKDeviceDiscovery`
  is the availability-detection abstraction; `SKComputeBackend<F>` adds
  `transfer_policy`, `upload`, `download` and `dispatch` over
  `SKDeviceBuffer<F>`, whose builder validates device ownership, shape,
  overflow, capacity and payload. The closed `SKGpuKernel` catalogue names
  pairwise distance, GEMM, elementwise and tree prediction, and an unsupported
  kernel reports `UnsupportedOperation` naming it. `gpu-opencl`, `gpu-cuda` and
  `gpu-rocm` are off by default and the foundation crate pulls in no GPU
  runtime; tests drive a mock backend, so no device is required to build or
  test.
- `SKBackendKind` gains the `OpenCl`, `Cuda` and `Rocm` kinds alongside the CPU
  kinds, with documented display names, `FromStr` parsing, `Default`,
  host/device predicates and `SKLinearRegressionModel` export/import round-trip
  support. Every workspace match is updated; the host math dispatch keeps
  device kinds on the pure-Rust backend until a GPU kernel backend owns them.
- Deterministic device selection and honest span telemetry: `sk_select_device`
  filters by the configured kind, the required scalar type and the required
  kernel, then orders by `(kind priority, device index)`; when nothing matches
  it returns `SKFallbackReason` (`NoDeviceAvailable`, `ScalarTypeNotSupported`,
  `KernelNotSupported` or `ExplicitModeDoesNotSelectDevice`).
  `SKExecutionContext` carries an optional `SKDeviceContext` and threads
  `scalar_size_bytes` into selection; `SKExecutionPlan` exposes the selected
  `backend` and `fallback_reason`. Runtimes populate the explicit process-wide
  inventory via `sk_register_device_inventory`, and `SKOperationObservation`
  resolves device kinds through `sk_resolve_operation_backend` so spans record
  the backend that actually served the operation plus any fallback reason.
- New structured errors for device work: `DeviceUnavailable` and
  `UnsupportedScalarType` (mapped to `SKFfiStatus::Unsupported` across FFI),
  and `DeviceTransferFailure` (mapped to `SKFfiStatus::Internal`).

### Moved

- The Go bindings (spec `go-bindings`) now live in their own repository,
  [`neurono-ml/sciencekit-go`](https://github.com/neurono-ml/sciencekit-go), as
  the module `github.com/neurono-ml/sciencekit-go`. The binding itself is
  unchanged: it loads `sciencekit_ffi` through cgo and exposes typed builders
  and models for linear regression and the standard scaler (`Fit`, `Predict`,
  `Transform`, `Score`, `Save`, `LoadLinearRegression`). Dense
  `Matrix[float32|float64]` values convert with Go-side shape validation and
  copied outputs; native status codes map to `ScienceKitError` values carrying
  the code, operation and message; builders and models close explicitly
  (`Close`) or through `runtime.SetFinalizer`, safely and exactly once; models
  are safe for concurrent use. The new repository carries the examples, the
  build/link README, a `scripts/build-native-library.sh` helper, a
  `gofmt`/`go vet`/`go test`/`go test -race`/`GOEXPERIMENT=cgocheck2 go test`
  CI workflow and the test suite covering every family, persistence, errors,
  lifecycle, concurrency and large arrays.
- The Lua bindings (spec `lua-bindings`) now live in their own repository,
  [`neurono-ml/sciencekit-lua`](https://github.com/neurono-ml/sciencekit-lua).
  The `sciencekit_lua` crate becomes a standalone Cargo project — no longer a
  member of this workspace — that depends on `sciencekit_ffi` through a git
  dependency on this repository. The module is unchanged: a native Lua 5.4
  module loaded with `require("sciencekit")`, with Lua 5.4 embedded through
  `mlua` with `lua54` + `vendored` (neither a system `liblua` nor
  `pkg-config` is required), chainable builder userdata for linear regression
  and the standard scaler, `fit`/`predict`/`transform`/`score`/`save`/`load`,
  handles released exactly once by close, the `__close` metamethod or the
  garbage collector, copy semantics for every numeric table, structured Lua
  runtime errors headed by the stable status name and code, a LuaRocks 3.0
  rockspec and the Lua suite.
- The `go/` module, the `go.work` file, the `crates/sciencekit_lua` crate, the
  `go-bindings` CI workflow and `scripts/build-go-library.sh` are removed from
  this repository. The stable C ABI (`sciencekit_ffi`) stays here: it is the
  shared contract both bindings consume.

## [0.6.0] — 2026-11-10

The k-NN imputer, grid search and the embedded Lua bindings.

[0.2.0]: https://github.com/neurono-ml/sciencekit/compare/v0.1.0...v0.2.0
[0.3.0]: https://github.com/neurono-ml/sciencekit/compare/v0.2.0...v0.3.0
[0.4.0]: https://github.com/neurono-ml/sciencekit/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/neurono-ml/sciencekit/compare/v0.4.0...v0.5.0
[0.6.0]: https://github.com/neurono-ml/sciencekit/compare/v0.5.0...v0.6.0
[0.7.0]: https://github.com/neurono-ml/sciencekit/compare/v0.6.0...v0.7.0

### Added

- `sciencekit_impute` gains the nearest-neighbor imputer (spec
  `nearest-neighbor-imputation`): `SKKNNImputer<F>`, builder-only with
  `missing_values` (default `NaN`), `nearest_neighbors_count` (default 5),
  `weights` (`Uniform` default, `Distance`), `add_indicator` (default false),
  `keep_empty_features` (default false), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`). Fit retains the training design, its
  missingness mask, the rows observing every column, the observed column means
  and the columns with at least one observed value; all-missing columns are
  dropped by default and kept at zero when `keep_empty_features` is enabled.
  Transform computes the exact NaN-aware Euclidean distance over the
  co-observed features in fixed-size blocks, selects the nearest donors per
  column with partial selection, and fills each cell with the uniform or
  inverse-distance weighted average of the donors — zero-distance duplicates
  take the whole weight — falling back to the observed column mean when no
  donor distance is finite. `add_indicator` stacks the fitted missingness mask
  after the imputed features and tolerates missingness unseen at fit. A scalar
  marker is normalized to `NaN` for the distance kernel, so `-999` and similar
  sentinels behave like `NaN` while a `NaN` under another marker is rejected as
  a non-finite value. Sparse input is the documented dense-only exception: a
  CSR implicit zero is an observed zero, so fit and transform return
  `UnsupportedRepresentation` and point at `SKSimpleImputer`. Empty input, a
  zero or excessive neighbour count, a feature-count mismatch and non-finite
  values report structured errors; the fitted model is `Send + Sync`, the
  transform is deterministic under concurrency and parallel rows equal the
  sequential result.

- `sciencekit_model_selection` gains the hyperparameter grid search (spec
  `grid-search`): `SKGridSearchCVClassifier<F, E>` and
  `SKGridSearchCVRegressor<F, E>`, builder-only with a list of fully configured
  candidates, a splitter (`SKKFold`, `SKStratifiedKFold` or `SKGroupKFold`), a
  scoring policy, `refit` (default true), `error_score` (`SKErrorScore::Raise`
  by default or a numeric value), an optional group vector, `backend` (default
  `Faer`) and `execution_mode` (default `Automatic`). Fit evaluates every
  candidate on every fold in parallel, stores the scores by candidate and fold
  index so results are independent of thread count, aggregates mean and
  population standard deviation per candidate, and selects the highest mean
  with ties broken by candidate index. When refit is enabled the winner is
  retrained on all rows and exposed through `best_model`; otherwise prediction
  returns `UnsupportedOperation`. The fitted models expose
  `best_candidate_index`, `best_score`, `best_parameters`, `results`,
  `mean_scores`, `standard_deviations`, `fold_scores` and
  `ranked_candidate_indices`, and the classifier/regressor wrappers implement
  the matching prediction contracts, delegating to the best model while
  preserving dense and CSR views. Classification policies are accuracy,
  balanced accuracy, F1/precision/recall with an averaging policy, binary AUC
  and log loss; regression policies are negative MSE, negative MAE and R². The
  builder rejects empty candidates, a missing splitter and an incompatible
  scoring family, and a sparse-only candidate proves no fold is ever densified.

- `sciencekit_lua` (spec `lua-bindings`): a native Lua 5.4 module loaded with
  `require("sciencekit")`, built on the stable `sciencekit_ffi` entry points so
  the Lua surface exercises exactly the C ABI. Lua 5.4 is embedded through
  `mlua` with `lua54` + `vendored`, so neither a system `liblua` nor
  `pkg-config` is required to build or to run the tests (the crate's Rust tests
  drive an embedded VM; a plain `tests/run.lua` suite covers the
  system-interpreter path). It exposes chainable builder userdata for linear
  regression (`include_intercept`, `backend`) and the standard scaler
  (`with_mean`, `with_std`) with `fit`, `predict`, `transform`, `score`, `save`,
  module-level `load`, `close` and the Lua 5.4 `__close` metamethod; handles
  are released exactly once by close, `__close` or the garbage collector, with
  `live_handles()` confirming nothing leaked. Numeric tables convert to/from
  row-major `f64` buffers with rectangularity, emptiness, cell-type and target
  length checks before any ABI call — every value is copied by design, and no
  zero-copy path across Lua values is promised. Failures become Lua runtime
  errors headed by the stable status name and code
  (`sciencekit shape (2): ...`). Includes a LuaRocks 3.0 rockspec and a Lua
  suite covering every ABI family, errors and the lifecycle, plus large-fixture
  and concurrent-use Rust tests.
  `sciencekit_ffi` now also builds as an `rlib` for in-process bindings.

## [0.5.0] — 2026-11-10

Clustering metrics and the k-nearest-neighbour estimators.

[0.2.0]: https://github.com/neurono-ml/sciencekit/compare/v0.1.0...v0.2.0
[0.3.0]: https://github.com/neurono-ml/sciencekit/compare/v0.2.0...v0.3.0
[0.4.0]: https://github.com/neurono-ml/sciencekit/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/neurono-ml/sciencekit/compare/v0.4.0...v0.5.0

### Added

- `sciencekit_metrics` clustering evaluation (spec `clustering-metrics`):
  per-sample and mean silhouette over chunked Euclidean pairwise distances,
  with noise rows excluded and fewer than two clusters rejected; the adjusted
  Rand index; normalized mutual information in arithmetic, geometric and
  maximum forms; homogeneity, completeness and V-measure; and the
  Davies-Bouldin (lower is better) and Calinski-Harabasz (higher is better)
  centroid scores. `SKSilhouetteScorer` is the first concrete
  `SKUnsupervisedScorer`, delegating to the free function, which excludes the
  typed `Noise` variant. Dense and CSR features share the pairwise/centroid
  paths: the representation is dispatched once and a sparse design is never
  densified. Pairwise labeling metrics treat noise as an ordinary label by
  design, while silhouette and the centroid scores exclude it. Empty input,
  length mismatch, single-cluster and noise-only assignments, non-finite
  values and insufficient degrees of freedom report structured errors, and
  every score is deterministic under concurrent calls.

- `sciencekit_neighbors` gains the k-nearest-neighbour estimators (spec
  `nearest-neighbor-estimators`): `SKKNeighborsClassifier<F>` and
  `SKKNeighborsRegressor<F>`, both builder-only with `nearest_neighbors_count`
  (default 5), `weights` (`Uniform` default, `Distance`), `algorithm` (`Auto`
  default, `KdTree`, `BallTree`, `BruteForce`), `metric` (`Euclidean` default,
  `Manhattan`, `Chebyshev`, `Cosine`), `leaf_size` (default 40), `backend`
  (default `Faer`) and `execution_mode` (default `Automatic`). `Auto` builds a
  tree for dense Euclidean, Manhattan or Chebyshev data with at least 64 rows
  (a ball tree above 20 features) and uses brute force for sparse, mixed or
  cosine data; an explicit tree request the input cannot serve returns
  `UnsupportedRepresentation` with the alternative named. Classification takes
  the weighted majority vote of the `k` nearest rows and reports class
  probabilities as weighted neighbour fractions over the canonical class table;
  vote ties go to the smallest canonical class index, distance ties to the
  smallest training row index. Regression reports the weighted mean of the
  neighbour responses. `Distance` weights follow the reference convention: when
  a selected neighbour sits exactly on the query row, only the zero-distance
  duplicates contribute. Both estimators accept dense and sparse fit inputs and dense, sparse
  and mixed prediction inputs, dispatch once per operation, never densify sparse input
  and expose the resolved search strategy, the stored training representation,
  the carried backend and the execution intent on the fitted model. A new
  sparse-aware brute-force searcher implements the shared
  `SKNearestNeighborsSearcher` contract over chunks sized from a fixed memory
  budget, selects the `k` smallest `(distance, training_index)` pairs with
  `select_nth_unstable_by` and sorts them deterministically. Fitted models are
  `Send + Sync`; the companion tests cover validation errors, determinism,
  concurrency, dense/sparse parity and accuracy/R² production through
  `sciencekit_metrics`.

## [0.4.0] — 2026-11-10

Model selection and the C ABI: reproducible splits, folds and cross-validation scoring, plus
the stable C boundary every non-Rust binding wraps.

[0.2.0]: https://github.com/neurono-ml/sciencekit/compare/v0.1.0...v0.2.0
[0.3.0]: https://github.com/neurono-ml/sciencekit/compare/v0.2.0...v0.3.0
[0.4.0]: https://github.com/neurono-ml/sciencekit/compare/v0.3.0...v0.4.0

### Added

- `sciencekit_ffi` sub-crate (spec `ffi-abi`): the stable C ABI for the core data
  types and the first end-to-end path. Opaque builder/model handles are integer
  tokens resolved through a registry, so null, unknown, wrong-kind and freed
  handles return `SK_FFI_STATUS_INVALID_HANDLE` without dereferencing memory, and
  in-flight predictions survive a concurrent free. Every entry point maps the
  central error taxonomy onto the stable status codes (`OK`, `INVALID_HANDLE`,
  `SHAPE`, `INVALID_ARGUMENT`, `UNSUPPORTED`, `NOT_FITTED`, `IO`, `INTERNAL`) and
  exposes a thread-local last-error message that a successful call clears. Array
  descriptors (`dtype`, `ndim`, `shape`, `strides`, `data`, `len`) are validated
  (null pointers, dimensions, length-versus-shape, dtype, row-major strides)
  before any element is read; inputs are borrowed without copying and outputs are
  library-owned buffers released by `sk_ffi_array_free`. Descriptors whose data
  pointer is misaligned for its dtype, or whose `shape`/`strides` are misaligned,
  are rejected before any typed view is built. The ABI exposes builder setters,
  `fit`, `predict`, `transform` and `score` for `SKLinearRegression` (f32/f64, R²
  scoring) and `SKStandardScaler` (f32/f64); `save`/`load` currently cover fitted
  linear regression in f64 only (an f32 model or a scaler returns
  `SK_FFI_STATUS_UNSUPPORTED`) through the format-agnostic `SKModelState`
  contract with a documented little-endian file layout. Also `catch_unwind` on
  every entry point, an ABI version/feature query, a hand-maintained
  `include/sciencekit.h`, and a compiled `tests/c_smoke.c` that fits, predicts,
  scores, saves, loads and transforms. `SKLinearRegressionModel` now implements `SKExportable` and
  `SKImportable`, and the companion tests cover invalid descriptors, use after
  free, dtype mismatches, concurrent predictions, panic containment and
  persistence round-trips.

- `sciencekit_cluster` sub-crate (spec `k-means`): `SKKMeans<F>`, a builder-only
  Lloyd/Elkan k-means estimator with `number_of_clusters` (default 8),
  `initialization` (`KMeansPlusPlus` default, `Random`), `number_of_initializations`
  (default 1 for k-means++ and 10 for random), `maximum_number_of_iterations`
  (default 300), `tolerance` (default `1e-4`), `algorithm` (`Auto` default, `Lloyd`,
  `Elkan`), `random_seed` (default 0), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`). Centroid initialization is deterministic
  from the seed: k-means++ D² sampling or distinct uniform rows, one derived seed per
  initialization, with the lowest-inertia run winning and exact ties broken by the
  earliest index. `Auto` selects Elkan for dense input and Lloyd for sparse input;
  an explicit Elkan request over sparse data resolves to Lloyd and reports the
  resolved algorithm. Empty clusters are repaired deterministically by moving the
  farthest still-claimed point (largest distance, then smallest row index). The
  fitted `SKKMeansModel<F>` exposes `cluster_centers`, typed `labels`,
  `inertia`, `iterations_taken`, `converged`, `number_of_clusters`,
  `number_of_features`, `algorithm` and `backend`, implements `SKUnsupervisedFit`
  and `SKClustererPredictor` (always `SKClusterLabel::Cluster`, never `Noise`), and
  offers `transform` distance-to-centers, `fit_predict` and concurrent assignment.
  Dense and CSR training/assignment share one norm-expansion kernel without
  densifying sparse input; validation rejects empty input, zero or excessive
  clusters, non-finite values (naming their flat position), invalid tolerance,
  iteration, initialization counts and feature-count mismatches. Companion tests
  cover tiny, large (50 000-row) and wide sparse data, Lloyd/Elkan/Auto equivalence,
  dense/sparse parity, determinism across repeated fits and thread counts,
  concurrency, inertia and classification metric production through
  `sciencekit_metrics`.

- `sciencekit_tree` sub-crate (spec `decision-trees`): the CART estimators
  `SKDecisionTreeClassifier<F>` and `SKDecisionTreeRegressor<F>`, each builder-only with
  `criterion` (Gini default and Entropy for classification; SquaredError default and
  AbsoluteError for regression), `maximum_depth` (default unlimited),
  `minimum_samples_split` (default 2), `minimum_samples_leaf` (default 1),
  `maximum_features` (`All` default, plus `Count`, `Fraction`, `Sqrt` and `Log2`),
  `splitter` (`Best` default or seeded `Random`), `random_seed` (default 0), classifier
  `class_weight` (`None` default, `Balanced` or explicit), `backend` (default `Faer`) and
  `execution_mode` (default `Automatic`). Fitting grows a flat, cache-friendly binary tree
  by recursive partitioning, maximizing the impurity decrease at every node; classification
  leaves store the normalized class distribution and predict the majority class, while
  regression leaves store the mean (squared error) or median (absolute error). The fitted
  models expose node count, depth, feature count, normalized impurity-based feature
  importances, the canonical class table, criterion, backend and execution intent, and
  predict labels, probabilities or responses. Dense and borrowed CSR designs share the
  same boundary and produce identical trees and predictions without densifying sparse
  input; sparse split search re-indexes CSR to CSC once and merges stored entries with
  implicit zeros. Validation rejects empty input, non-finite values, feature-count and
  target-length mismatches, single-class classification targets, unsupported target
  representations and invalid depth/split/leaf/feature-subsampling/class-weight
  configurations with structured central errors. Fits are deterministic for a fixed seed,
  including the random splitter and seeded feature subsampling, and fitted models are
  `Send + Sync` and serve concurrent prediction.

- `sciencekit_model_selection` sub-crate (spec `model-selection-splits`): reproducible
  data partitioning through index splits. `sk_gather_rows` gathers dense rows into a dense
  block and CSR rows into CSR (never densifying) while preserving the target
  representation; `sk_shuffle_indices` is a seeded Fisher–Yates permutation from the shared
  deterministic generator. `SKKFold` (builder-only, `number_of_splits` default 5, `shuffle`
  default false, `random_seed` default 0) produces disjoint, exhaustive folds; the
  default `shuffle` stays off while an explicit seed makes repeated runs identical.
  `SKStratifiedKFold` deals each class round-robin so every fold sees every class in
  proportion, rejecting a class smaller than the fold count; `SKGroupKFold` keeps every
  group in one fold and defaults the split count to the distinct group count.
  `sk_train_test_split` accepts test/train sizes as fractions or counts, seeded shuffling,
  optional stratification and gathers both partitions (surplus rows stay unassigned when
  both sizes are configured). `sk_cross_val_score_classifier` and
  `sk_cross_val_score_regressor` fit one model per fold in parallel and return scores in
  fold order through the shared scorer contracts. Validation rejects empty input, invalid
  fractions or explicit counts larger than the input, too many splits, class/group cardinality
  below the split count, feature/target/group length mismatches and non-finite dense or sparse
  values with structured central errors.
  Companion tests cover tiny and 10 000-row data, dense/sparse parity, all splitters,
  stratification and group properties, determinism across threads, concurrency and metric
  production through `sciencekit_metrics`.

## [0.3.0] — 2026-11-10

The interop, pipeline, spatial-index and streaming-audit batch: format-agnostic model
export/import contracts, compile-time typed pipelines, exact k-d/ball trees, the
engine-neutral dataset interop foundation and the workspace streaming coverage audit.

[0.2.0]: https://github.com/neurono-ml/sciencekit/compare/v0.1.0...v0.2.0
[0.3.0]: https://github.com/neurono-ml/sciencekit/compare/v0.2.0...v0.3.0

### Added

- `sciencekit_neighbors` sub-crate (spec `nearest-neighbor-search`): the exact dense
  spatial indexes `SKKDTree` (median split over the widest-spread feature) and
  `SKBallTree` (farthest-pivot split over the node centroid), each builder-only through
  `SKKDTreeBuilder`/`SKBallTreeBuilder` with `leaf_size` (default 40), `metric` (default
  Euclidean), `backend` (default `Faer`) and `execution_mode` (default `Automatic`), plus
  shape and configuration accessors (`training_rows`, `training_features`, `node_count`,
  `depth`, `backend`, `metric`, `execution_mode`). The shared
  `SKNearestNeighborsSearcher<F>` contract exposes exact `k_nearest` and inclusive
  `radius_neighbors` returning the owned `SKNeighborQuery<F>` (row-major distances and
  indices) and `SKRadiusNeighbors<F>` result types; results are ordered by distance
  ascending with ties broken by the smallest training row index, and large batches are
  evaluated row-parallel with rayon while producing the same output as the sequential
  path. The closed `SKNeighborMetric` catalogue covers Euclidean, squared Euclidean,
  Manhattan and Chebyshev distances with metric-specific KD box and ball bounds, while
  `SKRequestedNeighborMetric` registers cosine and NaN-aware Euclidean as rejected at
  build time. The index is the documented dense-only exception: sparse training or query
  views return `UnsupportedRepresentation` with a brute-force suggestion and are never
  densified; empty training sets, zero `leaf_size`, zero or oversized `neighbors_count`,
  feature-count mismatches, non-finite values and unsupported metrics return structured
  errors. Both indexes are `Send + Sync` and serve concurrent query batches.

- `sciencekit_interop` sub-crate (spec `dataset-interop-core`): the engine-neutral tabular seam.
  `SKTabularSchema` / `SKColumnSchema` / `SKColumnRole` / `SKLogicalType` describe ordered, named,
  typed columns and reject empty or duplicate-name schemas; `SKColumnSelection` resolves ordered
  features and at most one target by name or position, rejecting unknown, duplicate and multi-target
  selections before any row is read; `SKMissingValuePolicy` defaults to `Reject` and documents
  `NumericNaN`, `ConstantFill` and `CategoricalMissingCode`. `SKColumnData` stores nullable `f32`,
  `f64`, `i64`, `bool`, nominal text and category codes; `SKTabularBatch` validates row alignment and
  materializes dense `Array2`, CSR, `SKDataBatch`, `SKSupervisedBatch`, `SKTargetView` and
  `SKCategoricalView` conversions, delegating label canonicalization to `SKLabelTable`. Contiguous
  row-major matrices and single numeric columns are borrowed (`SKRowMajorFeatureView`,
  `SKNumericColumnView`); columnar tables are materialized because the layout requires it.
  `SKTabularSource` is the reopenable, fallible source contract, bridged into
  `SKSourceProvider` / `SKSupervisedSourceProvider` by `SKTabularSourceProvider`; the
  `SKAsyncTabularSource` bridge polls async streams on a dedicated reader thread with a bounded
  channel, backpressure and cancellation. A conversion-span helper records source kind, shape and policy for adapters that use it.
- `sciencekit_common` interop error variants in the central taxonomy (`UnknownColumn`,
  `DuplicateColumn`, `UnsupportedColumnType`, `NullValueInColumn`, `UnevenColumnLengths`,
  `SchemaMismatch`), each naming the offending column where the error is column-oriented.


- Streaming coverage audit (spec `sparse-streaming-coverage`): a workspace-level matrix listing every `fit_streaming`/`partial_fit` implementation with the representations it accepts, a completeness test that fails when a streaming estimator is added without an audit row, the documented-exception test proving `SKOneHotEncoder` is the only dense-only streaming path (its input is categorical), and a dense-only guard test showing a streaming estimator rejecting CSR with `UnsupportedRepresentation` before processing any element.

- Streaming coverage audit (spec `sparse-streaming-coverage`): a workspace-level matrix listing every `fit_streaming`/`partial_fit` implementation with the representations it accepts, a completeness test that fails when a streaming estimator is added without an audit row (a source scan counts the actual `fit_streaming`/`partial_fit` definitions), the documented-exception test proving `SKOneHotEncoder` is the only dense-only streaming path (its input is categorical), and a dense-only guard test showing a streaming estimator rejecting CSR with `UnsupportedRepresentation` before processing any element; sparse `partial_fit` is now implemented for the three scalers and OLS, which previously rejected CSR.
- `sciencekit_common` model export/import contracts (spec `model-export-contracts`): the format-agnostic `SKModelState<F>` (estimator metadata, a deterministically ordered map of named tensors and typed hyperparameters), `SKModelMetadata` (estimator name, format version, scalar dtype, feature count, canonical class labels and a `SKParameterValue` map), `SKTensor<F>` with `SKTensorDType` (`F32`/`F64`/`I64`/`Bool`) and shape/dtype/name validation, and the `SKExportable<F>`/`SKImportable<F>` trait pair with the shared `sk_validate_import_metadata` check — no file format and no `serde` dependency.

- `sciencekit_pipeline` sub-crate (spec `typed-pipeline`): the compile-time typed linear
  pipeline. `SKPipeline::start` opens a builder whose nested-tuple type records every
  composed stage; `then` appends a transformer stage and `finalize_classifier` /
  `finalize_regressor` close the chain with a supervised estimator, rejecting an empty
  chain with `SKError::InvalidHyperparameter`. The scalar and the stage output are checked
  by the type system, with `compile_fail` doctests covering a mixed-scalar stage and a
  regressor used as a classifier finalizer.
- `sciencekit_pipeline` stage adapter: `SKPipelineTransformer` plus a blanket
  implementation over every `SKUnsupervisedFit` estimator whose model implements
  `SKFeatureTransformer` and whose output enters the owned block medium, so every existing
  transformer (the three scalers, the simple imputer and the missing indicator) composes
  without changing its trait.
- `sciencekit_common` pipeline block boundary: `From<Array2>`, `From<CsMat>`,
  `From<SKDataView>` and `TryFrom<&SKFeatureBlock>` conversions, plus the
  `SKIntoFeatureBlock` trait implemented for the impute crate's `SKImputedOutput` and
  `SKIndicatorOutput`, keeping the dense/CSR representation across the boundary.
- `sciencekit_pipeline` fitted models: `SKPipelineClassifier` / `SKPipelineRegressor`
  implement the shared classifier and regressor prediction contracts by transforming the
  input through the fitted stages and delegating to the final model, and expose the
  configured stages, the fitted stage models, the final model, the stage count and the
  final feature count. Dense and sparse blocks flow without implicit densification, and
  the fitted models are `Send + Sync` for concurrent prediction.

## [0.2.0] — 2026-11-10

The first algorithm-and-contract batch after the evaluation foundation: representation
parity, the Wave 7 scope clarification, shared deterministic randomness, typed cluster
assignments, sparse pairwise distances and the ranking metrics. See the entries below.

[Unreleased]: https://github.com/neurono-ml/sciencekit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/neurono-ml/sciencekit/compare/v0.1.0...v0.2.0

### Added

- `sciencekit_common` typed cluster assignments (spec `clustering-contracts`): `SKClusterLabel` (a cluster identifier or the explicit `SKClusterLabel::NOISE`, four bytes per element through a `NonZeroU32` niche), `SKClusterIdentifier`, `SKClustererPredictor<F>` returning one label per row through the shared `SKDataView` seam, the `to_i64`/`from_i64` interop conversions (`Noise ↔ -1`, values below `-1` or above the maximum cluster index rejected with a structured error, never a panic), and `sk_validate_cluster_assignments`, the shared assignment-shape check.
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
- `sciencekit_impute` capability specs published (archive of `simple-imputers`): the marker semantics dense and sparse share, the four per-column strategy statistics, the empty-column policy, the exact incremental and out-of-core guarantees, and the missingness mask; the same archive corrected the `backend-selection` rule to match the library, where the backend is never a type parameter and the scalar `SKFloat` is the estimator's only possible one — present only when its configuration carries scalar values.
- `sciencekit_linear_model` sub-crate (Wave 2, first supervised estimator): `SKLinearRegression` (builder-only `include_intercept`, `backend`, `execution_mode`) with a distinct fitted `SKLinearRegressionModel<F>` — dense `fit` solving the augmented design matrix through the backend's least-squares kernel (SVD/pseudoinverse, never the normal equations), surfacing `rank` and `singular_values` and returning the minimum-norm solution under rank deficiency; `SKRegressorPredictor<F>` returning responses as `X·coefficients + intercept`; an incremental `partial_fit` and an out-of-core `fit_streaming` accumulating the Gram state `XᵀX`/`Xᵀy` and solving once at the end, with the squared-condition trade-off documented.
- `sciencekit_math` kernels `sk_gram_matrix` (`XᵀX`) and `sk_gram_vector` (`Xᵀy`): the cross-product companions of the axis reductions (sequential and race-free parallel forms, shared reduction grain).
- `sciencekit_common` supervised streaming contract: `SKSupervisedBatch` (owned features + targets), the `SKSupervisedLazySource` fallible iteration, and `sk_run_supervised_streaming_driver`, which shares the existing prefetch engine with the unchanged `sk_run_streaming_driver`.
- `sciencekit_math` backend `SKBackendSelection` / `sk_math_backend_for_kind`: the per-operation `SKBackendKind`→backend dispatch, keeping the feature-gated `blas-backend` resolution inside `sciencekit_math`.
- `sciencekit_linear_model` (Wave 2, ridge): `SKRidge<F>` (builder-only `alpha` scalar-or-sequence, `include_intercept`, `solver`, `tolerance`, `maximum_number_of_iterations`, `backend`, `execution_mode`) with a distinct fitted `SKRidgeModel<F>` carrying one coefficient vector and one intercept per `alpha`; a solver portfolio (`Automatic`/`Cholesky`/`Svd`/`SparseCg`) with a deterministic `Automatic` resolution (dense → Cholesky, sparse → SparseCg) and a Cholesky→SVD fallback; an unpenalized intercept via centering (Gram/SVD) and the algebraically equivalent unpenalized augmentation (CG); dense, CSR fits (Cholesky on the sparse-accumulated Gram), and `SparseCg` sparse fits whose solve never forms `XᵀX`; and incremental `partial_fit` plus out-of-core `fit_streaming` through the centered Gram accumulator.
- `sciencekit_math` iterative linear-solver substrate (spec `iterative-linear-solvers`): `SKLinearOperator` (zero-copy SPD operator), `sk_conjugate_gradient` (structured `NotConverged` on budget exhaustion) and `SKCsrNormalEquationOperator` (zero-copy `AᵀA` over a borrowed `CsMatView`, never densifying) — the substrate the sparse ridge path consumes and later linear models reuse.
- `sciencekit_linear_model` (Wave 2, L1-regularized linear models): `SKLasso<F>` and `SKElasticNet<F>` (builder-only `alpha` scalar-or-sequence, `l1_ratio` for ElasticNet, `include_intercept`, `positive`, `selection`, `screening`, `tolerance`, `maximum_number_of_iterations`, `backend`, `execution_mode`) with distinct fitted `SKLassoModel<F>`/`SKElasticNetModel<F>` carrying one coefficient vector and one intercept per `alpha`; a shared glmnet-style **coordinate-descent** solver (soft-threshold coordinate updates, an unpenalized residual-mean intercept, cyclic/deterministic-random sweeps, convergence on the maximum coordinate change) with structured `NotConverged` on budget exhaustion; **gap-safe screening** (a strong initial active set, a dual-feasible gap radius and a KKT reactivation pass that guarantees the screened solution equals the unscreened one); dense and CSR fits that never densify a sparse design, sharing one coordinate loop through a column-access abstraction; incremental `partial_fit` and out-of-core `fit_streaming` through a **retained-design** accumulator (the L1 objective is not a function of the Gram matrix) that re-solves at read time; and rayon parallelism across the independent per-alpha solves.
- `coordinate-descent-solver`, `lasso` and `elastic-net` capability specs published: the shared solver contract (configuration and validation, the alpha-conversion trait, sweep selection, screening, the retained-design accumulator and the parallel per-alpha solve) and the two estimator/model contracts.
- `sciencekit_linear_model` (Wave 2, the first classifier): `SKLogisticRegression<F>` (builder-only `penalty` {l2, l1, elasticnet, none} with `l1_ratio`, both `C` and `alpha` with last-setter precedence and a warning when both are set, `solver` {lbfgs, liblinear, saga}, `multi_class` {auto, ovr, multinomial}, `class_weight` {none, balanced, explicit}, explicit `warm_start_from`, `include_intercept`, `tolerance`, `maximum_number_of_iterations`, `backend` default `Faer`, `execution_mode` default `Automatic`) with a distinct fitted `SKLogisticRegressionModel<F>` carrying the class table and one coefficient vector per head; integer and nominal targets canonicalized through `SKLabelTable`; the `SKClassifierPredictor<F>` impl (`predict_labels`, `predict_probabilities`) plus a `decision_function`; dense and zero-copy CSR fits; incremental `partial_fit` and out-of-core `fit_streaming` over a retained-design accumulator; and a shared `logistic` solver substrate — a pure-Rust L-BFGS-B (orthant-wise) solver for the smooth L2/none path, a SAGA variance-reduced incremental-gradient solver for L1/ElasticNet/none and the sparse path, and a LIBLINEAR-style coordinate-descent solver for the L1/L2 one-vs-rest path — over a dense/sparse design abstraction that never densifies a sparse design.
- `logistic-solver` and `logistic-regression` capability specs published: the shared solver contract (configuration and validation, penalty/solver/multi-class resolution, class-weight normalization, the three solvers, the design abstraction and the retained-design accumulator) and the estimator/model contract.
- `sciencekit_linear_model` (Wave 2, stochastic gradient): a shared `sgd` substrate — the sealed loss contract with nine losses split into `SKSGDClassifierLoss` and `SKSGDRegressorLoss`, four learning-rate schedules (`Constant`, `Optimal` with an estimated `t₀`, `InverseScaling`, `Adaptive`), the proximal L1/L2/ElasticNet penalty, per-sample class weights, the optional averaged iterate, a self-contained seeded SplitMix64 shuffler, the single-sample update and running epoch objective, dense/CSR/streaming epoch drivers, and convergence plus early-stopping bookkeeping with `SKConvergencePolicy { Strict, Permissive }` (a provider-limited stream never raises; the model reports `converged()` and `epochs_taken()`); `SKSGDClassifier` (one-vs-rest heads, integer and nominal targets through `SKLabelTable`, labels/decisions/probabilities for `log_loss` and `modified_huber`, class weighting, explicit warm start, dense and zero-copy CSR fits, one-pass `fit_streaming`, `partial_fit`); and `SKSGDRegressor` (`squared_error`, `huber` and both epsilon-insensitive losses with an `epsilon` band, the same schedule/penalty/averaging surface, dense and CSR fits, streaming and incremental fitting) — the library's first supervised estimators whose cost does not depend on the dataset size.
- `stochastic-gradient-solver`, `stochastic-gradient-classifier` and `stochastic-gradient-regressor` capability specs published: the loss catalogue, schedules, proximal penalty, weighting, averaging, shuffling, convergence and the epoch drivers, and the two estimator/model contracts.
- `sciencekit_common` error taxonomy: `SKError::UnsupportedOperation { operation, reason }` for an operation an otherwise valid configuration cannot provide (probabilities under a non-probabilistic loss, early stopping over a one-shot stream).
- `sciencekit_linear_model` shared class weighting: `SKClassWeight` and its per-sample expansion moved to a crate-shared module, re-exported through `logistic` so the existing path keeps resolving.
- `sciencekit_common` streaming source providers: `SKSourceProvider<F>` and `SKSupervisedSourceProvider<F>` with a generic associated source type, a declarative `IS_REOPENABLE` flag and a fallible `open`; the residual implementation for `&mut S` (so every existing `fit_streaming(&mut source)` call keeps compiling and takes one pass) and `SKReopenableSource<G>` wrapping a fallible factory (one fresh source per pass). The executor gains `SKStreamingPassPolicy { SinglePass, Iterative { maximum_number_of_passes } }` and provider entry points that open one source per pass, warn when a one-shot provider cannot honour the budget, and propagate factory errors.
- `sciencekit_preprocessing`, `sciencekit_impute` and `sciencekit_linear_model` (streaming providers): every `fit_streaming` now consumes a provider, except the categorical `SKOneHotEncoder`, which keeps its iterator seam (the documented categorical exception); the exact accumulators (scalers, polynomial features, simple imputer, OLS, ridge, lasso, elastic net, logistic regression) always take exactly one pass, while `SKSGDClassifier` and `SKSGDRegressor` run one epoch per pass from a reopenable provider, honouring `maximum_number_of_iterations`, carrying the schedule counter and average across passes, and reporting `epochs_taken()` and `converged()` — a streamed multi-epoch fit with one batch per pass now matches the in-memory fit (multi-batch shuffles differ by construction: the stream shuffles within a batch while the in-memory fit permutes globally).
- `sciencekit_common` sparse streaming blocks: `SKFeatureBlock<F>` — an owned, non-exhaustive `Dense(Array2<F>)` / `Sparse(CsMat<F>)` block with a zero-copy `view() -> SKDataView`, shape counts, an `is_sparse` flag and a `byte_size` measured from the stored buffers. `SKDataBatch` and `SKSupervisedBatch` now carry a block, so a stream can deliver CSR data without ever densifying it; the prefetch driver passes the representation through untouched and the double-buffer memory guard checks each batch's stored byte size as it is read, before the update sees it.
- `sciencekit_preprocessing`, `sciencekit_impute` and `sciencekit_linear_model` (sparse streaming): every streaming estimator accepts sparse batches — the scalers accumulate per-column moments/extrema in `O(nnz)` with implicit zeros participating, polynomial features expand sparse input directly and emit sparse output, the simple imputer applies its dense marker semantics over CSR, and every linear model (OLS, ridge, lasso, elastic net, logistic regression, SGD classifier/regressor) reuses its in-memory sparse accumulators so a sparse stream equals the dense stream and the dense `fit` within the documented tolerance.
- `sciencekit_metrics` sub-crate (evaluation foundation): `SKLabelSpace` (explicit or observed-union label resolution), `SKConfusionMatrix`, accuracy, per-class precision/recall/F1 with `SKAveraging { Macro, Micro, Weighted, Binary { positive_label } }`, mean squared error, root mean squared error, mean absolute error and `R²` with the documented finite policies; scorer adapters (`SKAccuracyScorer`, `SKPrecisionScorer`, `SKRecallScorer`, `SKF1Scorer`, `SKMeanSquaredErrorScorer`, `SKRootMeanSquaredErrorScorer`, `SKMeanAbsoluteErrorScorer`, `SKR2Scorer`) implementing the shared scoring contracts; and mergeable streaming accumulators (`SKClassificationAccumulator`, `SKRegressionAccumulator`) using Neumaier compensated sums and producing the same values as whole-input evaluation.
- `sciencekit_common` probability scoring contract: `SKProbabilityScorer<F, M>` with a pure form over stored class probabilities and a convenient form that calls `predict_probabilities`, plus `sk_validate_probability_shape` rejecting empty targets, a row mismatch and a matrix without class columns.
- `sciencekit_metrics` ranking metrics: binary `sk_roc_auc_score` and `sk_roc_curve` with average-rank tie handling; multiclass one-vs-rest `sk_multiclass_roc_auc_score` and `sk_one_versus_rest_roc_auc_scores` with `SKRankingAveraging { Macro, Micro, Weighted }`; `sk_precision_recall_curve` and `sk_average_precision_score` with step interpolation; `sk_log_loss` and `sk_binary_log_loss` with per-scalar epsilon clipping; `sk_balanced_accuracy_score` and `sk_matthews_correlation_coefficient` with a zero-denominator policy of zero; the scorer adapters `SKAucScorer`, `SKAveragePrecisionScorer`, `SKLogLossScorer`, `SKBalancedAccuracyScorer` and `SKMatthewsCorrelationScorer`; the documented `O(n)` mergeable `SKAucAccumulator`; and the new `SKError::ProbabilityOutOfRange { position }` for probabilities outside `[0, 1]`.
- `sciencekit_math` pairwise catalogue gained the Chebyshev (L∞) distance and the NaN-aware Euclidean distance (co-observed-feature semantics with `total_features / present_features` scaling, NaN for pairs without co-observed features), both with row-parallel rayon paths.
- `sciencekit_common` central errors gained `EmptyInput { operation }`, `NonFiniteValue { position }` and `UnknownLabel { label }`, plus the `evaluation-metrics`, `pairwise-distances`, `error-model` and `backend-selection` spec deltas that document the metric foundation.
- `random_seed(u64)` builder parameter (default `0`) and matching estimator/model accessors on every stochastic estimator: `SKLogisticRegression` (the SAGA solver derives one stream per head from `sk_derive_seed`), `SKLasso` and `SKElasticNet` (`SKSelection::Random` derives one stream per `alpha` from its configured position). The same seed, data and configuration reproduce the coefficients, and parallel heads/alphas never share a generator, so results are independent of the thread count and completion order.
- `sciencekit_common` seeded randomness (spec `deterministic-randomness`): `SKRandomGenerator`, a SplitMix64 stream shared by every stochastic estimator, with `next_u64`, `next_f64`, `uniform`, `normal` (Box–Muller with a cached spare), `next_index`, `shuffle`, `shuffled_indices` and `sample_indices_without_replacement`; `sk_derive_seed(global_seed, domain, index)` with the named domain tags `SK_RANDOM_DOMAIN_TREE`, `SK_RANDOM_DOMAIN_INITIALIZATION`, `SK_RANDOM_DOMAIN_STAGE`, `SK_RANDOM_DOMAIN_FOLD` and `SK_RANDOM_DOMAIN_PROJECTION`, so every component derives an order- and thread-independent sub-seed, plus the cross-cutting rule that every stochastic builder exposes `random_seed(u64)` defaulting to `0` and never reads an ambient RNG. Golden-vector, range, permutation, moment, sub-seed distinctness and thread-count invariance tests pin the contract. `sciencekit_linear_model`'s SGD substrate now consumes the shared generator and its private SplitMix64 copy is gone, with the seeded behavior unchanged.

- `sciencekit_math` sparse pairwise distances (spec `pairwise-distances`): `sk_sparse_squared_euclidean_distance_matrix`, `sk_sparse_euclidean_distance_matrix`, `sk_sparse_manhattan_distance_matrix`, `sk_sparse_chebyshev_distance_matrix` and `sk_sparse_cosine_distance_matrix` over borrowed `sprs` CSR views, covering sparse×sparse, sparse×dense and dense×sparse inputs without ever densifying; the closed `SKPairwiseMetric` enum and `sk_pairwise_distance_matrix` dispatch once per operation to the dense or the sparse kernel, return a dense matrix, validate feature counts with `ShapeMismatch` instead of panicking, and reject a sparse NaN-aware Euclidean request with `UnsupportedRepresentation` (a CSR implicit zero is an observed value, so no missing marker exists); every sparse metric is row-parallel with the dense grain, deterministic and concurrency-safe, and the dense fast paths keep their signatures and behaviour.

### Changed

- **BREAKING** `sciencekit_common::SKUnsupervisedScorer` now binds its model to `SKClustererPredictor<F>` and its pure form takes `ArrayView1<SKClusterLabel>` instead of `ArrayView1<usize>`: clustering evaluation no longer rounds a regressor response into an index and can represent noise explicitly.
- Representation parity is now a normative rule (spec `data-view-boundary`, PRD §4.5.1): an algorithm that mathematically tolerates more than one feature representation accepts dense, sparse and mixed inputs on every public operation, dispatches once per operation and never densifies a sparse input. A dense-only exception (§4.5.2) is admissible only when the mathematics requires dense geometry, the kernel rejects a sparse view with `UnsupportedRepresentation`, and the estimator keeps sparse/mixed parity through a documented alternative path.
- The PRD now states that the `memmap2`-backed indexes and kernel caches, model persistence and model export are Phase 7 (interop) deliverables; the Phase 3 (neighbours/SVM) and Phase 4 (trees) sections carry the corresponding non-goal cross-references, and algorithms consume them through the Phase 0 `SKMappableSource` contract instead of growing private `memmap2` layers.

- `SKLogisticRegression` (SAGA), `SKLasso` and `SKElasticNet` (`SKSelection::Random`) no longer carry private xorshift64* streams with hard-coded seeds: they draw from the shared `SKRandomGenerator`, seeded through `random_seed(u64)`. Default (`0`) results can differ from the previous hard-coded stream, while determinism for a fixed seed is preserved.
- **BREAKING** `sciencekit_common` streaming batch accessors: `SKDataBatch::data()` and `SKSupervisedBatch::features()` are replaced by representation-aware accessors (`view()`, `number_of_rows()`, `number_of_columns()`, `is_sparse()`, `byte_size()`), and the supervised constructors become fallible (`Result<Self, SKError>`) to reject a target count that disagrees with the block's rows. The dense constructors keep their meaning; the compiler catches every old call site.
- **BREAKING** `sciencekit_common` execution surface: `SKExecutionContext` gains `batch_byte_size_hint` and `SKExecutionPlan` gains `available_memory_bytes`, so external struct literals must set both fields. The provider entry points now enforce the double-buffer guard on every batch they read.
- `SKOperationAttributes` shape fields renamed to `rows_count`/`columns_count` (they are counts); the tracing span field names stay `rows`/`columns`.
- Translated `AGENTS.md` and `openspec/config.yaml` to English (international open-source project).
- `README.md` now links the official GitHub Pages site (`https://neurono-ml.github.io/sciencekit/`) at the top; it is also set as the repository homepage.
- GitHub Actions opencode agent now uses the `opencode/big-pickle` model.

### Fixed

- `sciencekit_math` least-squares kernel now uses the **thin** SVD instead of the full SVD: the full decomposition allocates an `m × m` `U`, which made a tall-design fit (e.g. 20 000 rows) take seconds; the pseudoinverse path only needs the first `min(m, n)` singular vectors, and the minimum-norm solution is unchanged.
- Workspace `[profile.dev.package."*"] opt-level = 2`: dependency code (pure-Rust linear algebra) stays optimized in dev/test builds while the project crates stay unoptimized.
- `linear-regression` capability published (archive of `linear-regression-ols`): the `SKLinearRegression` estimator/model contract; the `higher-order-kernels` capability gained the `sk_gram_matrix`/`sk_gram_vector` cross-product reductions and `streaming-batches` gained the supervised streaming contract.

