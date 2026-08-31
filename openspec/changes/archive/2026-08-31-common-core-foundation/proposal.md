# Proposal: common-core-foundation

## Why

Every algorithm on the roadmap depends on the central vocabulary defined in the PRD (§3.3 fundamental traits, §4 memory, §5 concurrency, §9 errors). An in-depth design exploration decided the contracts that cross the entire library — generic dtypes, estimator/trained-model separation, dense/sparse/target views with a conversion seam open to third parties, execution mode resolution and owned streaming batches. These contracts must exist, tested and stable, before the first estimator (Phase 1); changing them later would cost breaking changes across the whole surface.

## What Changes

- Creation of the **`crates/sciencekit_common`** crate (first sub-crate of the workspace), containing:
  - Sealed trait **`SKFloat`** — the library's single numeric bound (generic dtype as a trait parameter).
  - **Central error `SKError`** (`thiserror`) with the agreed taxonomy; per-algorithm error enums derive via `From<SKError>`.
  - Zero-copy boundary views: **`SKDataView<'_, F>`** (dense/sparse, `#[non_exhaustive]`) and **`SKTargetView`** (continuous/integer/nominal), both reachable through the `TryInto` seam, with precise runtime rejection of unsupported representations.
  - Label table (**`SKLabelTable`**) + canonicalization helpers used by classifiers (foundation of the Phase 1 explicit codecs).
  - Fit contracts: **`SKUnsupervisedFit`** / **`SKSupervisedFit`** (configured estimator → fitted model as a distinct type) and **`SKFeatureTransformer`** with an associated output type (anticipated requirement of the Phase 6 type-safe pipeline).
  - Evaluation contracts: **`SKSupervisedScorer`** / **`SKUnsupervisedScorer`** with pure input (`score_from_predictions`/equivalent) + convenient provided method (`score`) that runs inference.
  - Execution: enum **`SKExecutionMode`** (intent) ≠ struct **`SKExecutionPlan`** (resolved plan), injectable context and a pure resolution function (`sk_resolve_execution_plan`), with a **hard error** for explicit modes incompatible with the declared access pattern.
  - Streaming: owned struct **`SKDataBatch<F>`** + traits **`SKLazySource`** (sequential batch iterator) and **`SKMappableSource`** (O(1) random access).
- Organization into standardized folder modules from the start (200-line-per-file limit).

Out of scope: any algorithm, estimator builders (born in Phase 1 with the first estimator), complete encoder/decoder codecs (Phase 1), concrete metrics (own crates), Python bindings and GPU backends (separate changes, after CPU validation).

## Capabilities

### New Capabilities

- `scalar-typing`: sealed trait `SKFloat` — the single numeric contract used by views and traits.
- `error-model`: central `SKError` taxonomy, agreed variants and uniform propagation across algorithms via `From`.
- `data-view-boundary`: `SKDataView`/`SKTargetView`, `TryInto` seam (compatible with `Into` via the std blanket), mandatory zero-copy in native impls, non-breaking evolution through new variants (`#[non_exhaustive]`), target canonicalization and label table.
- `estimator-contracts`: semantics of the two fit traits (distinct model-type return, immutable, `Send + Sync` by construction) and of the transformer with output typed by an associated type.
- `scoring-contracts`: supervised/unsupervised scorers with dual input (existing predictions vs built-in inference).
- `execution-planning`: intent/plan separation, deterministic per-operation resolution with injectable context, hard error on explicit incompatibility.
- `streaming-batches`: owned batches with minimal metadata and the two source traits (sequential/memmap-abstract).

### Modified Capabilities

(none — no existing specs)

## Impact

- **Code:** new crate `crates/sciencekit_common`; no existing crate is altered (there are none).
- **Dependencies:** `ndarray`, `sprs`, `num-traits`, `thiserror`, `sysinfo` in `sciencekit_common`. No `memmap2` (the mappable source trait remains abstract — the concrete implementation arrives with interop) and no Tokio/rayon in this change.
- **Downstream:** all future algorithm changes consume these contracts; consciously deferred decisions stay recorded in design.md (internal nominal representation, binding continuous target dtype to `F`, 2D multi-target, type-inference caveat with `TryInto`).
- **Acceptance criteria (PRD §8.7/§10.3):** there are no trainable models or export yet in this change — the full criteria activate with the first estimator. Acceptance here is: all contracts compile under `Send`/`Sync` where promised, companion tests cover view conversions, canonicalization, deterministic resolution (simulated context), hard incompatibility error and the batch iteration contract, all with mock data in `ndarray`/`sprs` and TDD.
