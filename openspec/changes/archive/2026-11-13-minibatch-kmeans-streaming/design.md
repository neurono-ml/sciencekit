# Design

## Context

See `proposal.md` — Why. `SKKMeans` (W5.1) and the shared assignment contract exist.
`sciencekit_common` provides `SKLazySource`, `SKSourceProvider` and the prefetch executor;
`SKFeatureBlock` carries dense or CSR batches. The estimator must keep the same model
surface as KMeans so downstream code can swap one for the other.

## Goals / Non-Goals

**Goals:**
- Exact MiniBatch update with per-center learning rates and reassignment.
- One-pass and multi-epoch streaming through the existing executor.
- Dense/sparse parity and deterministic results.

**Non-Goals:**
- Elkan pruning on streaming batches (the batch update is approximate by design).
- Missing values, GPU, `memmap2`, export.
- A separate model type hierarchy; reuse the KMeans center/assignment model shape.

## Decisions

### Decision 1 — MiniBatchKMeans lives in `sciencekit_cluster` beside KMeans

The estimator shares the center matrix, assignment kernel and model accessors with
KMeans. Its model additionally stores per-center counts and batch/epoch progress. The
configured estimator keeps the same builder shape so `Auto` selection between KMeans and
MiniBatch can be added later without changing the model contract.

### Decision 2 — Initialization on a bounded sample

Initialization uses at most `initialization_batch_size` rows, drawn deterministically from
the first available batch (or from the full data for `fit`). It runs the same
k-means++/random initializer as KMeans; `number_of_initializations` runs independent
seeds and selects the smallest inertia on the initialization sample.

### Decision 3 — Per-center learning rate

For a row assigned to center `c`, the update is:

```
center[c] += (row - center[c]) / (count[c] + 1)
count[c] += 1
```

This is the standard MiniBatchKMeans update and converges to the batch mean as the count
grows. The update is vectorized over the batch where possible and deterministic in row
order; parallel row updates would race on the same center, so the batch loop is sequential
per center but the assignment step is parallel.

### Decision 4 — Reassignment

After each batch, compute `mean_count`; any center with `count < reassignment_ratio *
mean_count` is a candidate. The point with the largest distance to its assigned center is
moved to the candidate center and its count is reset to one. The choice is deterministic
(largest distance, then smallest row index). This prevents dead centers without a full
reinitialization.

### Decision 5 — Partial fit and streaming

`partial_fit` processes one owned batch and returns the updated estimator state (the
configured estimator is immutable per the fit contract; the streaming driver carries a
mutable state internally). `fit` chunks the input into `batch_size` blocks and calls the
same update. `fit_streaming` drives `SKSourceProvider` through
`sk_run_streaming_driver`, keeping a state across epochs; a reopenable provider opens a
fresh source per epoch, a one-shot provider is consumed once. The driver's early-stop
signal is used when the center movement falls below `tolerance` (when `tolerance > 0`).

### Decision 6 — Convergence and epoch reporting

`tolerance = 0` means "fit the configured batches/epochs and do not stop early"; a
positive tolerance stops when the maximum center movement is at or below it. The model
reports `batches_processed`, `epochs_consumed` and `converged()`.

### Decision 7 — Representation parity

Dense and sparse batches share the assignment kernel from KMeans; sparse batches are read
through their stored entries and never densified. The centers stay dense.

### Decision 8 — Validation and observability

Builder/fit validation follows the proposal. Fit/partial-fit spans record the backend;
the model carries it. Errors: `EmptyInput`, `InvalidHyperparameter`, `ShapeMismatch`,
`NonFiniteValue` and the streaming source error converted into the central taxonomy.

### Decision 9 — File layout

```
crates/sciencekit_cluster/src/minibatch_kmeans/
  mod.rs, builder.rs, config.rs, init.rs, update.rs, reassign.rs, state.rs,
  model.rs, fit.rs, streaming.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **MiniBatch is approximate** → documented; the batch size and initialization size
  control the trade-off, and tests compare against full KMeans within tolerance.
- **Center updates are sequential per batch** → assignment is parallel; the update is
  cheap relative to assignment.
- **Reassignment can be sensitive to outliers** → the ratio is configurable and the
  choice is deterministic.
- **Streaming epochs require a reopenable provider** → the one-shot case is explicit.
- **No Elkan** → the streaming path is inherently approximate; Elkan is not applicable per
  batch.

## Migration Plan

Additive: new estimator in `sciencekit_cluster`. No existing API changes.

## Open Questions

None blocking. An automatic KMeans/MiniBatch selection is a future convenience.
