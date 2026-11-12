# Design

## Context

See `proposal.md` — Why. `grid-search-cv` supplies candidate evaluation, scoring policies,
fold gathering and refit. `model-selection-core` supplies splitters and seeded subsampling.
The halving search is a scheduling layer over the same evaluation machinery.

## Goals / Non-Goals

**Goals:**
- Successive-halving scheduling with a deterministic trace.
- Reuse grid-search scoring, fold evaluation and refit.
- Stratified subsampling for classification.

**Non-Goals:**
- Bayesian/randomized search.
- Multi-resource (sample + feature) schedules.
- `memmap2` and export.

## Decisions

### Decision 1 — Resource is the number of samples

The first version supports `resource = NumberOfSamples`. A round with resource `r` draws a
deterministic subset of `r` rows (seeded; stratified when the splitter requires it) and
evaluates every survivor on the folds. Future resources (features, boosting rounds) can be
added as an enum without changing the estimator.

### Decision 2 — Round loop

```
alive = all candidates
resource = minimum_resource
while alive.len() > 1 && resource <= maximum_resource:
    scores = evaluate(alive, resource)
    alive = top ceil(alive.len() / factor) by mean score, ties by index
    record round(alive, resource, scores)
    resource *= factor
```

The winner is the single surviving candidate; if the loop ends at the maximum resource with
more than one candidate, the highest mean score wins.

### Decision 3 — Subsample and folds

The resource subset is drawn once per round with `SKRandomGenerator` (seed + round index)
and then split by the configured splitter. For `SKStratifiedKFold`, the subset preserves
class proportions; the splitter receives the subset targets. Groups are respected by the
group splitter.

### Decision 4 — Aggressive elimination

When enabled, a round may eliminate candidates whose mean score is below the worst score
that could still survive at the next resource, using the same deterministic tie rule. The
first version implements the standard "prune the bottom half early" rule and records every
pruned candidate in the trace.

### Decision 5 — Reuse and model

The classifier/regressor wrappers share the grid-search scoring enums, error-score policy
and refit path. The model exposes `best_candidate_index`, `best_score`, `elimination_trace`
(round, resource, surviving indices, scores), the refitted best model and the wrapper's
prediction methods.

### Decision 6 — Validation and observability

Validate factor ≥ 2, `minimum_resource ≥ 1`, `minimum_resource ≤ maximum_resource`,
non-empty candidates and scorer/estimator compatibility. Spans record the round and
resource; candidate estimators record their backend.

### Decision 7 — File layout

```
crates/sciencekit_model_selection/src/halving/
  mod.rs, classifier.rs, regressor.rs, builder.rs, scheduler.rs,
  subsample.rs, model.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Early rounds see few samples** → the configurable minimum resource and the trace make
  the trade-off explicit.
- **Aggressive elimination can discard a late bloomer** → off by default; the trace shows
  every prune.
- **Subsampling adds variance** → seeded and documented; the final winner is refitted on
  all data.
- **Only sample resources** → a future enum can add feature/iteration resources without
  breaking the API.

## Migration Plan

Additive: new estimators in `sciencekit_model_selection`. No existing API changes.

## Open Questions

None blocking. Multi-resource and Bayesian search are future changes.
