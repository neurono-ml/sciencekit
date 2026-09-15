# Proposal: How to add an Algorithm (Tutorials)

## Why

There is no teaching material that shows how to actually **write an algorithm by hand** against `sciencekit`'s contracts — builders, fit/transform/predict traits, execution plans, kernels with measured grain, the streaming driver, and backends all exist, but the knowledge of how they compose into one new estimator lives only in the minds of the maintainers and in the agent's context. The library is entering the phase where the first real algorithm crates (preprocessing, linear models, nearest neighbors) must be written, and the Development Explained book explains *why existing code was built*, not *how the next one gets written*. Contributors — including the project owner working without an agent — need a single, complete, worked guide.

## What Changes

- Add a new mdBook part `## Tutorials` (parallel in level to `## Development Explained`) under `docs/src/tutorials/`, registered in `docs/src/SUMMARY.md`.
- New anchor chapter `how-to-add-an-algorithm.md`: the mandatory, algorithm-agnostic recipe — file/module layout (≤200-line files, folder modules, companion `*_tests.rs`), naming rules (PRD §3.4), builder pattern with `SKBuilderState` and `sk_validate_hyperparameter`, fit/transform/predict trait composition with zero-copy inputs (`SKDataView`/`SKTargetView`), `SKExecutionContext` wiring (grain, access pattern), `sk_resolve_execution_plan`, the three execution regimes (in-memory kernels with resolved parallelism; `sk_run_streaming_driver` multi-batch lazy loading; `SKMappableSource` memory-mapped random access), `sk_run_operation` observability spans, TDD loop, local CI gates, and the PRD §8.7 acceptance checklist.
- Four worked case-study chapters, each presenting **complete reference source code** (written against the current contracts of `feat/parallel-kernel-execution`-level `sciencekit_common`/`sciencekit_math`), each algorithm shown with its **full dual-regime formulation** (the same combinable state serving the in-memory path and the streaming/mapped path, selected by the resolved `SKExecutionMode`, never hardcoded):
  - `adding-a-transformer.md` — `SKStandardScaler` and `SKRobustScaler`: in-memory parallel reduction via `sk_axis_sum`; streaming via Welford/Chan combiner; quantiles in stream for RobustScaler discussed honestly; catastrophic cancellation vs Σx² naive; sources.
  - `adding-a-linear-model.md` — `SKLinearRegression`: in-memory OLS via backend `lstsq` (SVD) and why not the normal equations directly; multi-batch streaming via incremental Gram accumulation with explicit precision trade-off; sources (Golub & Van Loan, Trefethen & Bau).
  - `adding-a-streaming-model.md` — `SKSGDClassifier` and `SKSGDRegressor`: gradient descent from first principles (losses, convexity, learning-rate schedules incl. why scikit-learn's `optimal` is non-trivial), in-memory as the degenerate single-batch case of the same update, `SKOperationKind::PartialFit` spans, pros/cons vs full-batch.
  - `adding-a-neighbors-model.md` — `SKKNeighborsClassifier` and `SKKNeighborsRegressor`: the honest special case (fit stores the base; predict does the work), brute-force via `sk_squared_euclidean_distance_matrix` + top-k, `SKMappableSource`/memory-mapped base, why sequential streaming does not fit kNN, kd-tree vs brute-force trade-off, curse of dimensionality; sources.
- Each chapter follows the Development Explained style: decisions and roads not taken, numerical/Rust concepts taught from zero (Rust language details referenced, not re-explained), Mermaid diagrams rendered before commit, per-algorithm study sources.
- Docs-only change: everything is `docs/src/` content plus `docs/src/SUMMARY.md`. No code, spec, or CI behavior changes.

## Capabilities

### New Capabilities
(none — documentation-only change)

### Modified Capabilities
(none — no spec-level behavior changes)

This change sets `skip_specs: true` in its `.openspec.yaml` (no existing behavior is specified or altered).

## Impact

- **Files added:** `docs/src/tutorials/index.md`, `how-to-add-an-algorithm.md`, `adding-a-transformer.md`, `adding-a-linear-model.md`, `adding-a-streaming-model.md`, `adding-a-neighbors-model.md`.
- **Files modified:** `docs/src/SUMMARY.md` (register the new part).
- **Book build:** `mdbook build docs` must pass locally before push; the GitHub Pages deploy workflow on `docs/documentations` publishes it automatically.
- **Audience/effect:** contributors and the project owner gain a complete manual recipe to implement the first algorithm crates (`sciencekit_preprocessing`, `sciencekit_linear_model`, `sciencekit_neighbors`) entirely by hand, without agent assistance.
