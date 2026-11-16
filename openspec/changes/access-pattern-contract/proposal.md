# Proposal

## Why

The library already has an `SKAccessPattern` enum and the execution resolver already
rejects an explicit mode incompatible with "the algorithm's declared access pattern", but
no estimator actually declares one: every operation hardcodes the value it passes into the
resolution context (`SKAccessPattern::Sequential` in the scalers, `RandomAccess` in
truncated SVD, and so on), there is no trait or associated constant that a consumer or a
test can read, and the enum has no variant for algorithms that cannot run out-of-core.
Meanwhile PRD §4.4 promises that every algorithm declares its pattern and implements the
matching source trait, and no estimator in the workspace consumes `SKMappableSource`.

Without a single declaration this cannot be enforced: the parity program must classify 244
scikit-learn entries by memory tier (`streaming`, `mappable`, `chunked`, `full-memory`) and
cannot do so against values scattered inside `fit` bodies. The 181 missing algorithms would
each re-invent their own pattern handling, and "out-of-core for every algorithm" stays
unfalsifiable.

## What Changes

- Extend `SKAccessPattern` with a fourth variant, `FullMemory`, and document the mapping to
  the parity tiers: `Sequential` = streaming, `RandomAccess` = mappable, `Iterative` =
  chunked, `FullMemory` = in-memory only.
- Add the declaration contract: a trait every public estimator implements, exposing its
  access pattern before fit, immutable for the estimator's lifetime, and independent of the
  operation being performed.
- Make every operation read the pattern from the declaration instead of hardcoding it, and
  record the declared pattern in the operation's observability attributes.
- Define source-kind compatibility and enforce it before any element is processed:
  `Sequential` accepts sequential lazy sources; `RandomAccess` accepts mappable sources;
  `Iterative` accepts reopenable sources of either kind; `FullMemory` accepts in-memory
  views only.
- Define the full-memory doctrine: a `FullMemory` estimator MUST name a documented
  approximate alternative and MUST reject a non-in-memory source with the existing
  `UnsupportedOperation` error carrying that alternative; it never attempts the out-of-core
  path.
- Add a conformance test that enumerates every public estimator, asserts a declaration
  exists, asserts the declared pattern is compatible with the source kinds the estimator
  accepts, and asserts the declared pattern matches the parity matrix tier.
- Explicit non-goals: implementing the mappable fit path (separate change
  `mappable-fit-path`), adding streaming implementations, changing the resolution
  arithmetic, and introducing per-estimator memory budgets (each algorithm change owns its
  budget and its refusal threshold).

No estimator results change: the resolved plans and the observable outputs stay identical
for every existing estimator.

## Capabilities

### New Capabilities

- `access-pattern`: the declaration contract, the four-tier vocabulary, the source-kind
  compatibility matrix, the full-memory doctrine, and the conformance requirement that
  keeps declarations, implementations and the parity matrix aligned.

### Modified Capabilities

None. `execution-planning` already consumes a declared pattern; this change supplies the
declaration it references without changing the resolution requirements.

## Impact

- **`sciencekit_common`:** `SKAccessPattern::FullMemory`, the declaration trait, the
  operation attribute carrying the pattern, and the conformance harness.
- **Every estimator crate:** each estimator declares its pattern and replaces the
  hardcoded per-operation assignment with the declared value.
- **`mappable-fit-path`:** depends on this change; it can assume every random-access
  estimator declares `RandomAccess`.
- **`sklearn-parity-matrix`:** every matrix row's `tier` must equal the declared pattern's
  tier; the conformance test enforces it.
- **Docs:** a chapter in the Development Explained book on why a declaration replaces a
  hardcoded value.
