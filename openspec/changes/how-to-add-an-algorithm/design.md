# Design: How to add an Algorithm (Tutorials)

## Context

All building blocks the tutorial teaches already exist on the merged/shared crate state (base level of `feat/parallel-kernel-execution`):

- Contracts: `SKSupervisedFit`, `SKUnsupervisedFit`, `SKFeatureTransformer` (with `type Output`), `SKPredictor` (on the model), `SKSupervisedScorer`/`SKUnsupervisedScorer`.
- Builders: `SKBuilder<Model>`, `SKBuilderState` (default `SKExecutionMode::Automatic`), `sk_validate_hyperparameter`, `SKReferenceEstimator` as the canonical shape.
- Execution: `SKExecutionContext` (grain, access pattern, memory, cores), `sk_resolve_execution_plan`, `sk_run_streaming_driver(source, state, update, &plan)`, `SKAccessPattern`, `SKExecutionMode` (five variants).
- Data: `SKDataView` (dense/sparse, `TryInto`-based zero-copy), `SKTargetView` (continuous/integer/nominal), `SKDataBatch`, `SKLazySource`, `SKMappableSource`.
- Math: kernels with measured grain (`sk_axis_sum`, `sk_scale_in_place`, `sk_elementwise_transform`, pairwise distance matrices), `SKMathBackend` (`lstsq` via SVD on the faer default backend), memory layout helpers.
- Observability: `sk_run_operation` with `SKOperationKind::{Fit, PartialFit, Transform, Predict, ...}`.

None of the seven reference algorithms exist as code anywhere yet; the library is at the "first algorithm crates" frontier. The book chapter code is therefore **the reference implementation the author and collaborators will re-type by hand** — exact signatures, real APIs, no pseudocode.

## Goals

1. A contributor with no prior exposure to `sciencekit` can, using only this part of the book, write a complete new algorithm (builder, estimator, model, both execution regimes, tests, observability) without agent assistance.
2. Every algorithm chapter demonstrates the **same state-is-combinable principle**: one formulation, two (or three) execution regimes, regime chosen by the resolved plan — never by branching on user type.
3. Numerical-computing concepts are taught from zero in every chapter (why Welford, why SVD not normal equations, why quantiles are hard in a stream, why SGD in-memory is a degenerate batch); Rust language concepts are referenced, not re-taught (the Development Explained book already carries those lessons).
4. Complete and efficient: the reference code must be the *optimal* version (parallel partials without races, Welford combiner, SVD/pinv path, vectorized batch SGD), each with pros/cons and when the obvious-but-wrong solution fails.

## Decisions

| Decision | Rationale | Rejected alternatives |
|---|---|---|
| One `## Tutorials` part, five files, single change | The value is the cross-linked recipe; per-family chapters reference the anchor's vocabulary instead of repeating it. | One giant chapter (unwieldy), per-chapter changes (loses coherence). |
| Reference code written against `feat/parallel-kernel-execution`-level APIs (merged master contracts) | Those are the contracts every current and future workspace member shares; no newer/older API mix. | Writing against `main` (further behind; awaiting merge). |
| Dual-regime exposition with `Automatic` default everywhere | PRD mandate (`execution_mode(...)` defaults to `Automatic`) and the user's correction: every model gets presented with and without streaming, per configuration. | Per-algorithm "streaming vs not" taxonomy (wrong mental model). |
| kNN chapter is explicitly the "exception" chapter | Honest pedagogy: sequential streaming does not fit kNN (fit is O(1), predict needs random access); `SKMappableSource`/`OutOfCoreMemoryMapped` is its version of out-of-core. Explaining *why* is the lesson. | Forcing kNN through the streaming driver (artificial, misleading). |
| SGD taught with constant and inverse-scaling schedules; `optimal` analyzed, not implemented | Reproducing scikit-learn's `optimal` heuristic faithfully is subtle (it hinges on a theoretical loss-concavity constant); the chapter explains why and cites the source, keeping the hand-written path tractable. | Implementing `optimal` in reference code (large, fragile surface now). |
| Machine-precision choices left explicit (`SKFloat` over f32/f64) | Teaching dtype policy is part of "adding an algorithm" (numerics differ between regimes); reference code is generic over `SKFloat` like the kernels. | Hardcoding f64. |
| Writing in two passes (infra + linear-family → SGD + kNN) | Quality guard for ~5k lines of prose+code; review checkpoint between passes. | Single pass. |
| Mermaid diagrams rendered and checked (via the drawing/rendering tools) before commit; no ASCII-art in the book | AGENTS.md documentation rule. | ASCII sketches in the book (forbidden). |

## Content contract per case-study chapter

Each of the four case-study chapters contains, in this order:

1. **Story & decision table** — what problem the algorithm solves, decisions made and roads not taken (Dev Book format).
2. **Theory from zero** — the numerical concept (moments, OLS/SVD, SGD/convexity, neighbor search) with references and pros/cons of alternatives, including the non-obvious failure modes of the obvious solution.
3. **State that combines** — the internal state type for naive/in-memory and the streaming combiner; why they share the same mathematical object.
4. **Full reference code** — file by file, matching the anchor chapter's skeleton: folder module layout, builder, estimator, model, fit/transform/predict impls, plan wiring, streaming driver wiring, `sk_run_operation` spans, companion `*_tests.rs` (condensed but real, as the executable source of truth walk-through).
5. **Both regimes side by side** — a Mermaid diagram of the data flow for in-memory vs streaming, and a table of when to choose which (and why `Automatic` usually suffices).
6. **Acceptance walk-through** — PRD §8.7: little and lots of data, concurrency, model export, metrics — mapped to the tests just written.
7. **Study sources** — books/papers/docs per theory topic.

## Risks / mitigations

- **API drift before the algorithms are implemented** → the anchor chapter states the contract level it targets (shared master contracts) and the Dev Book's `maintaining.md` gains the lockstep duty (new code chapter already required by AGENTS.md; when a real algorithm lands and diverges, maintenance edits the tutorial).
- **Large docs PR hard to review** → two-pass writing, `mdbook build docs` green locally, one PR per worktree pair (docs + openspec) referencing the ADR issue.

## Migration plan

Not applicable (docs-only; no runtime behavior). Publication is automatic via the `docs/documentations` deploy workflow.

## Open questions

None remaining — resolved during exploration: single change; docs-only with `skip_specs: true`; reference code is final-form text against current contracts; all seven algorithms dual-regime except kNN (explicit exception chapter).
