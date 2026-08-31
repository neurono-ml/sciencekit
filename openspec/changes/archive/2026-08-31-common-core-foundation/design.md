# Design — common-core-foundation

## Context

The `sciencekit_common` crate is born on top of the workspace from the `bootstrap-workspace` change. The contracts distilled here were decided in a previous design exploration and are formalized in the specs of this directory (see proposal — Capabilities). PRD structural constraints shaping the design: zero-copy on the public surface (§4.1), trait composition (§3.3), complete naming with the `sk`/`SK` prefix (§3.4), files up to 200 lines becoming standardized folder modules (§3.2), companion `*_tests.rs` tests (§3.5).

## Goals / Non-Goals

**Goals:**
- Freeze the vocabulary of types and traits every future crate compiles against.
- Guarantee by construction (`Send`/`Sync`, distinct estimator/model types) the promised concurrency properties.
- Keep the boundary open to external integrators (a single fallible conversion seam).
- Testable execution resolution without a real machine (injectable context).

**Non-Goals:**
- Any algorithm or estimator builder (Phase 1+).
- Complete encoder/decoder codecs (Phase 1 — here only the label table and canonicalization).
- Concrete implementation of memmap (interop), parallelism (rayon) and async I/O (tokio).

## Decisions

1. **Generic dtype as a trait parameter, not an associated type; sealed `SKFloat` trait as the single numeric bound.**
   - *Why:* each algorithm will have distinct kernels per dtype (SIMD `wide` uses concrete types); multiple impls per dtype are natural, and a single sealed bound prevents bound fragmentation.
   - *Alternatives:* associated type (prevents multiple dtypes per concrete type without extra machinery); concrete `f64` (simpler, but closes the door to the f32 demanded by performance/future GPU work).

2. **Two fit traits separated by supervision; fit returns a distinct model; fit operates on a shared reference.**
   - *Why:* post-fit immutability gives `Send`/`Sync` by construction, cross-validation and grid search parallelize without locks, predict-without-fit does not compile, and type-safe pipelines (PRD §2.1) require statically chainable types. A shared reference in fit allows reusing the same estimator across multiple folds without cloning.
   - *Alternatives:* sklearn-style mutable object (`&mut self`) — worse for concurrency and runtime usage errors; single generic trait over targets with `()` — uniform, but the `fit(x, ())` call site was rejected by the product owner.
   - *Naming convention:* configured estimator `SKXxx`; trained model `SKXxxModel` — aligned with export (the model is what exports Safetensors; hyperparameters go to the JSON header, PRD §8.2). The example in PRD §2.1 will be updated when this convention is formalized in a release.

3. **Data boundary: `#[non_exhaustive]` wrapper enums with a single dispatch at input; universal seam via the std's fallible bound (`TryInto`).**
   - *Why:* the seam gives third parties ONE integration point (implementing conversion into our views) valid for both infallible conversions (automatically promoted by the std) and fallible ones; the error flows into the `Result` already returned by operations. Dense/sparse dispatch once per operation keeps kernels clean; partially supporting algorithms reject the variant with a specific error before processing elements.
   - *Alternatives:* traits with fixed input per representation (marginal type safety, kills the extension point); infallible bound only (no door for third-party validation).
   - *Zero-copy contract:* native conversion implementations never copy matrix data; copies exist only as explicit materialization methods.
   - *Pipeline intermediates:* conversion from owned blocks borrows the data within the call scope — an anticipated requirement of Phase 6 static chaining.

4. **Targets: storage ≠ interpretation.**
   - *Why:* `[1,2,3]` is continuous for the regressor and categorical for the classifier; the view describes how data is stored (continuous/integer/nominal), the algorithm decides the meaning. Lossless integer→continuous elevation is allowed; numeric arithmetic over canonical labels never happens (canonicalization produces symbolic indices).
   - *Canonicalization:* deterministic free function → compact indices + reversible table; it is the foundation of classifier automatic encoding (Phase 1+) and of the explicit codecs whose training will be atomic with derived halves (encoder/decoder sharing an immutable table).
   - *Continuous target dtype decoupled from feature dtype:* the continuous view stores `f64` independently of the features' `F` parameter; regressors convert once at input. Avoids contaminating the whole target machinery with the generic parameter. Alternative (binding to `F`) reopened if benchmarks show real cost.

5. **Scorers with dual input and provided method; they live in the core vocabulary.**
   - *Why:* pure form (true targets × existing predictions) avoids re-inference; convenient form (model × features × targets) feeds GridSearchCV/pipelines; the provided method delegates to the pure one after inferring, so scorer authors implement only the metric. Both return a fallible result (inference can fail). Defined in the common crate from the start because they are part of the stable public contract; concrete implementations arrive with metrics.
   - *Recorded boundary:* optimization loss functions (boosting, Phase 4) are internal ensemble machinery — they do NOT use the scorers.

6. **Execution: intent (`SKExecutionMode`) ≠ plan (`SKExecutionPlan`); pure resolution per operation; hard error on explicit incompatibility.**
   - *Why:* dataset size only becomes known at operation input — fit and prediction resolve independent plans seeded by the same intent stored on the estimator/model. Resolution as a pure function over an injectable context makes behavior testable and deterministic. Hard failure preserves explicit-request semantics; automatic always picks a compatible mode and never fails on incompatibility.
   - *Context:* available memory, cores, dataset size, access pattern declared by the algorithm, batch hint. Physical machine reading stays confined to the default context constructor (`sysinfo`); decision logic never reads the environment.

7. **Streaming: owned batches with minimal metadata; iterable fallible sequential source; abstract random-access source without mapping dependency.**
   - *Why:* an owned block crosses thread boundaries (PRD §5.1's I/O ∥ CPU pipeline); borrowing would tie processing to the reader. The random contract stays independent from `memmap2` — mapped implementations belong to interop.

8. **Modular organization from the first commit:** folders per concept (`sk_float/`, `errors/`, `data_view/`, `target_view/`, `label_table/`, `fit_traits/`, `scorer_traits/`, `execution/`, `batching/`), `mod.rs` with public re-exports, companion tests alongside, no file beyond 200 lines.

9. **Minimal dependencies:** `ndarray`, `sprs`, `num-traits`, `thiserror`, `sysinfo`. No tokio/rayon/memmap2/serde in this change.

## Risks / Trade-offs

- [Fallible bound may degrade type inference at some call sites] → mitigated: native cases have direct impls; if real friction appears, own constructor sugar can be added without contract changes.
- [Per-dtype monomorphization multiplies generated code] → accepted: only instantiated dtypes compile; build-time review becomes a criterion in algorithm changes.
- [`sysinfo` weighs on the common crate's dependency graph] → mitigated: physical reading isolated in the default context constructor; test paths inject simulated values.
- [`#[non_exhaustive}` variants require a wildcard case in external matches] → intentional: it is the price of non-breaking evolution; documented as a consumption pattern.
- [Deferred decisions may cost localized rework] → explicit list below with default direction; all are internal, no impact on the public contract.

## Consciously deferred decisions (with default direction)

Internal representation of the nominal variant (direction: tiny internal enum accepting a slice of references and a slice of owned strings, exposed as a single public variant); 2D multi-target/multi-label (deferred — future new variant, non-breaking); curated conversions with feature flags for `chrono`/`uuid`/`arrow` and a `#[derive(SKTarget)]` macro (future changes); fine binding between execution plan and allocator choice (arrives with allocators).

## Migration Plan

New crate; no migration. Future consumers start against this API; rollback = revert the code branch merge.

## Open Questions

None blocking task detailing — the micro-decisions listed above have a defined default direction and will be confirmed during each module's TDD.
