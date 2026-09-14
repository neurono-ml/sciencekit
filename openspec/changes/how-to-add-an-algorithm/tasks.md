# Tasks: How to add an Algorithm (Tutorials)

## 1. Worktrees and issue scaffolding

- [x] 1.1 Create the ADR issue (Nygard format, English) on GitHub for the "how-to-add-an-algorithm" tutorials part.
- [x] 1.2 Create the docs worktree `temporary/worktrees/docs/how-to-add-an-algorithm` on branch `docs/how-to-add-an-algorithm` (based on `docs/documentations`).
- [x] 1.3 Create the openspec worktree `temporary/worktrees/docs/how-to-add-an-algorithm-openspec` on branch `docs/how-to-add-an-algorithm-openspec` and commit the OpenSpec artifacts there.

## 2. Anchor chapter (first writing pass)

- [x] 2.1 Write `docs/src/tutorials/index.md`: audience (hand-written implementations, no agent), prerequisites, the builder→fit→plan→dual-regime cycle overview, chapter map.
- [x] 2.2 Write `docs/src/tutorials/how-to-add-an-algorithm.md`: mandatory recipe — folder-module layout (≤200-line files, `*_tests.rs` companions), PRD §3.4 naming, builder pattern (`SKBuilderState`, `sk_validate_hyperparameter`, `SKReferenceEstimator` shape), fit/transform/predict trait composition, zero-copy `SKDataView`/`SKTargetView` inputs, `SKExecutionContext` (grain, access pattern, memory), `sk_resolve_execution_plan`, the three execution regimes (in-memory parallel kernels / `sk_run_streaming_driver` lazy batches / `SKMappableSource` random access), `sk_run_operation` spans, TDD loop, local CI gates (`cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`), PRD §8.7 acceptance checklist.
- [x] 2.3 Register the `## Tutorials` part in `docs/src/SUMMARY.md` (between `## Engineering` and `## Community`).

## 3. Transformer case study (first writing pass)

- [x] 3.1 Write `docs/src/tutorials/adding-a-transformer.md` — `SKStandardScaler` + `SKRobustScaler`: decision table; moments from zero (mean/variance, catastrophic cancellation, Welford/Chan et al. with sources); combinable state; full reference code (builder, estimator, model, in-memory path via `sk_axis_sum` partials without races, streaming path via Welford combiner + `sk_run_streaming_driver`, `transform` with `sk_scale_in_place`, `SKFeatureTransformer::Output`); RobustScaler quantiles-in-stream honest discussion (exact two-pass vs approximations, sources); both regimes Mermaid diagram; acceptance walk-through; study sources.

## 4. Linear model case study (first writing pass)

- [x] 4.1 Write `docs/src/tutorials/adding-a-linear-model.md` — `SKLinearRegression`: decision table; OLS from zero (least squares, conditioning, SVD pseudoinverse vs normal equations — Golub & Van Loan, Trefethen & Bau); combinable state; full reference code (in-memory via backend `lstsq`, streaming via incremental Gram accumulation per batch with explicit precision trade-off; `SKSupervisedFit`, model `SKPredictor`); both regimes Mermaid diagram; acceptance walk-through; study sources.

## 5. First-pass review checkpoint

- [x] 5.1 Render and verify all Mermaid diagrams of pass 1 with the drawing/rendering tooling (no render errors).
- [x] 5.2 Run `mdbook build docs` locally; fix all warnings/errors introduced so far.
- [x] 5.3 Review pass-1 chapters against the content contract (design.md) before writing pass 2.

## 6. Streaming model case study (second writing pass)

- [x] 6.1 Write `docs/src/tutorials/adding-a-streaming-model.md` — `SKSGDClassifier` + `SKSGDRegressor`: decision table; gradient descent from zero (convexity, losses incl. log-loss and squares, learning-rate schedules constant/inv-scaling; analysis of why scikit-learn's `optimal` heuristic is non-trivial, source-cited); combinable-accumulator state; full reference code (batch SGD update as the driver's `update` closure, in-memory as degenerate single-batch case, `SKOperationKind::PartialFit` spans, `SKPredictor` on model, label canonicalization via `sk_canonicalize_labels` for the classifier); pros/cons vs full-batch regression; both regimes Mermaid diagram; acceptance walk-through; study sources (Bottou, Ruder overview).

## 7. Neighbors case study (second writing pass)

- [x] 7.1 Write `docs/src/tutorials/adding-a-neighbors-model.md` — `SKKNeighborsClassifier` + `SKKNeighborsRegressor`: decision table; neighbor search from zero (brute force, curse of dimensionality, kd-tree vs brute-force trade-off with sources); the honest exception — sequential streaming does not fit kNN, `fit` stores the base, `predict` does the work; full reference code (base storage as `SKMappableSource`-compatible view / memory-mapped option, distance matrix via `sk_squared_euclidean_distance_matrix`, top-k selection, majority/ranking rules via label table); both regimes Mermaid diagram (in-memory vs memory-mapped); acceptance walk-through; study sources.

## 8. Final build and gates

- [x] 8.1 Re-render and verify every Mermaid diagram in the part (no errors).
- [x] 8.2 Run `mdbook build docs` locally; book must build clean.
- [x] 8.3 Update `docs/src/SUMMARY.md` final state and `README.md` pointer if needed per AGENTS.md documentation rules.

## 9. Push, PRs, review

- [x] 9.1 Push the docs branch and the openspec branch; open the docs PR (→ `docs/documentations`) and the openspec PR (→ `main`), both referencing the ADR issue with closing keywords.
- [ ] 9.2 Independent agent review validating the tutorial effectively teaches a collaborator to write `SKStandardScaler` end-to-end by hand; fix findings; final merge.
