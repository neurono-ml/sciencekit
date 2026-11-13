# Design

## Context

See `proposal.md` — *Why*. The constraints that shape the approach:

- `SKLazySource<F>` and `SKSupervisedLazySource<F>` are sequential, fallible, `&mut`
  consumers (`next_batch` / `next_supervised_batch`). A source cannot rewind.
- Eleven estimators currently expose `pub fn fit_streaming<S>(&self, source: &mut S)` where
  `S: SKLazySource<F>` (or the supervised twin). `SKOneHotEncoder` is the exception: it takes
  `IntoIterator<Item = C>` over categorical views.
- `sciencekit_common` owns the streaming vocabulary (`SKDataBatch`, `SKSupervisedBatch`,
  the sources, the prefetch driver, `sk_run_supervised_streaming_driver`).
- MSRV 1.85 supports generic associated types and `const` items in traits.
- Exact-accumulator estimators (Gram matrices, running moments, category tables) are
  mathematically single-pass; stochastic estimators (SGD today, mini-batch k-means and
  incremental PCA later) are iterative and need epochs.

## Goals / Non-Goals

**Goals:**

- One seam that accepts a concrete source unchanged and a factory without wrappers at the
  call site (factory wrapped once at construction, not per call).
- One place that decides how many passes an estimator may take.
- Zero dynamic dispatch, zero new dependency.
- No silent degradation: a request the provider cannot satisfy is an explicit error or a
  warning that the model makes observable.

**Non-Goals:**

- Sparse streaming (batches are dense; recorded in the PRD as a future change).
- Early stopping over a stream (needs a held-out slice the provider cannot guarantee).
- Parallel epochs (SGD's trajectory is sequential; parallel passes would change the answer).
- Unifying the categorical `SKOneHotEncoder` seam in this change.

## Decisions

### 1. Why not `Into<Source>`

`fit_streaming(source: impl Into<S>)` cannot work:

- `S` would appear only in the bound, so the compiler has no way to infer the target of the
  conversion; several `S` are plausible for the same argument.
- `impl<T> From<T> for T` already gives every concrete source `Into<Self>`; adding factory
  conversions for foreign types violates the orphan rule, and per-source impls would still
  leave the target type unconstrained.
- A closure factory has an unnameable type, so it cannot implement `From`/`Into` at all.

A dedicated trait is the only seam that covers both shapes. Alternatives considered:
`Iterator<Item = Batch>` (what one-hot uses today) — expresses one pass only and forces
every source to be re-shaped per estimator; requiring `Clone` on sources — a file-backed
source would clone-handle and replay, but memory-backed sources pay for a deep copy;
`Box<dyn Source>` — loses the error type and the static dispatch the crate relies on.

### 2. Provider shape: GAT + reopenability constant

```rust
pub trait SKSupervisedSourceProvider<F: SKFloat> {
    type Source<'pass>: SKSupervisedLazySource<F> where Self: 'pass;
    const IS_REOPENABLE: bool;
    fn open(&mut self) -> Result<Self::Source<'_>, SKError>;
}
```

The generic associated type lets the one-shot implementation return a reborrow
(`&'pass mut S`) while the reopenable implementation returns an owned source, under the same
trait. `IS_REOPENABLE` is a compile-time fact, so the driver's branch is free and an
estimator can reject an impossible request at the type level's discretion (it is a runtime
`if` on a constant, folded away). The feature-only provider is the same shape without
targets. Alternatives: two methods (`open` / `reset`) — the reset step forces sources to
implement rewinding they cannot; an enum of `Once | Factory` — closes the set and prevents a
future memory-mapped reopenable source from joining without editing the enum.

### 3. The residual `&mut S` implementation keeps call sites

```rust
impl<'s, F, S> SKSupervisedSourceProvider<F> for &'s mut S
where S: SKSupervisedLazySource<F>
{
    type Source<'pass> = &'pass mut S where Self: 'pass;
    const IS_REOPENABLE: bool = false;
    fn open(&mut self) -> Result<&mut S, SKError> { Ok(&mut **self) }
}
```

`estimator.fit_streaming(&mut source)` therefore keeps compiling and consumes the source
once. The reopenable path is explicit:
`SKReopenableSource::new(|| FileSource::open(path))`. Rationale: the migration must not break
every existing test and example; a wrapper type for the common case would be churn with no
safety gain.

### 4. One epoch rule, classified by estimator

| Class | Estimators | Passes |
|---|---|---|
| Accumulating (exact statistics) | linear regression, ridge, lasso, elastic net, logistic regression, standard/min-max/robust scaler, polynomial features, simple imputer | exactly 1, regardless of budget |
| Iterative (stochastic) | SGD classifier/regressor | up to `maximum_number_of_iterations` from a reopenable provider; 1 from a one-shot provider with a `tracing::warn!` |

The estimator declares its class; the executor enforces the rule. A reassembled pass over an
accumulator would double-count `n`, `Σx`, `XᵀX` and corrupt the model, so "exactly one" is a
correctness requirement, not a default. For an iterative estimator, `maximum_number_of_iterations`
is already the epoch budget of `fit`, so the same number governs both regimes — no new knob.

### 5. Driver integration

`sk_run_streaming_driver` / `sk_run_supervised_streaming_driver` gain provider-based entry
points that loop:

```
for pass in 0..budget {
    let mut source = provider.open()?;
    run_one_pass(&mut source, &mut state)?;   // existing prefetch driver
}
```

The one-pass entry points stay as thin adapters so the existing tests and non-migrated call
sites keep working until every estimator is migrated. The model reports `epochs_taken`, which
is how a caller detects the one-shot clamp without relying on logs.

### 6. Errors and warnings

- A factory or source failure aborts with the wrapped `SKError`, never a partial model.
- A one-shot provider with a budget above one: one pass plus `tracing::warn!` — a structured
  error would break every existing call that leaves the default budget. An iterative
  estimator under a strict convergence policy does not treat this as budget exhaustion: the
  model reports `converged() == false` and `epochs_taken() == 1` instead of failing.
- Early stopping over a stream: `SKError::UnsupportedOperation`. The validation slice would
  have to be retained across passes, which the provider contract does not guarantee.

### 7. Feature-only and supervised symmetry

Both traits and both reopenable wrappers mirror each other line for line, so the transformers,
imputers and supervised models share one mental model. The supervised provider carries the
targets inside `SKSupervisedBatch`; nothing else differs.

### 8. One-hot stays on its iterator seam (for now)

`SKOneHotEncoder::fit_streaming` consumes categorical views, not numeric blocks. Forcing it
into the provider trait would mean defining a categorical source contract (owned categorical
batches, positions, final flag) — a real design of its own. The change migrates it to the
same *shape* only if that contract is trivial; otherwise the exception is documented in the
book and tracked. Decision: keep the iterator seam, document it, and note the categorical
provider as a follow-up.

### 9. Migration order

1. Land the provider traits and the reopenable wrapper with their tests.
2. Add the provider-based driver entry points and keep the one-pass adapters.
3. Migrate the exact accumulators (mechanical, one per crate), each with a
   `reopenable ≡ dense` test.
4. Migrate SGD and turn on multi-epoch streaming (depends on `sgd-classifier-regressor`).
5. Update the book and `CHANGELOG.md`.

Each step is green before the next: the adapters keep the un-migrated estimators working
throughout, so the workspace never sits in a broken state.

## Risks / Trade-offs

- **GAT borrow friction** (the one-shot impl returns a reborrow from `&mut self`) → the
  implementation is tiny and covered by a dedicated test; if the borrow checker fights the
  driver loop, the driver opens the source and fully consumes it inside the same scope.
  Implementation note: the shipped contract bounds the GAT with `Send`, keeps the source
  iteration error at `From<SKError>` (the trait bound stays additive) and requires
  `Into<SKError>` on the provider traits' `Error` associated type — the split that lets a
  concrete source keep its own error type. The prefetch driver moves the opened
  source to its I/O thread, and higher-ranked bounds on the source's `Send`/`Error` otherwise
  hit a borrow-checker limitation that forces a `'static` borrow. Forwarding
  `SKLazySource`/`SKSupervisedLazySource` impls for `&mut S` keep the one-shot reborrow a
  valid source.
- **Coherence between the `&mut S` impl and the reopenable wrapper** → the wrapper is a
  concrete local type that does not implement the source trait; if rustc still flags
  potential overlap, the trait is sealed and the wrapper's impl moves to a private module.
- **Two ways to call the same method** (`&mut source` vs `SKReopenableSource`) → documented
  in the book; the model's `epochs_taken` makes the difference observable.
- **Silent one-pass clamp on a one-shot provider** → `tracing::warn!` plus `epochs_taken`;
  the SGD change's docs state it explicitly.
- **Non-iterative estimators must not take two passes** → the class is declared by the
  estimator and asserted by a shared test harness that attempts a second pass and expects a
  single-pass result.
- **Categorical seam remains divergent** → accepted and documented; tracked as a follow-up.

## Open Questions

- Whether the categorical provider contract should be designed together with the sparse
  streaming change (both generalize a batch), rather than as a separate effort.
