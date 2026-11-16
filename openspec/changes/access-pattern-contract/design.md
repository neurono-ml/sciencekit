# Design

## Context

`SKAccessPattern` already exists in `sciencekit_common::execution` with three variants
(`Sequential`, `RandomAccess`, `Iterative`), and `execution-planning` already resolves a
plan against "the access pattern declared by the algorithm". In practice the pattern is not
declared by the algorithm at all: each operation builds an `SKExecutionContext` and writes
the value inline, so the same estimator can silently disagree with itself, and nothing can
read the declaration from outside. `FullMemory` does not exist, so algorithms whose
mathematics needs the whole design resident have no honest way to say so. No estimator
consumes `SKMappableSource`, so `RandomAccess` today only means "I asked the resolver for
that mode", not "I can fit from a mapped file".

## Goals / Non-Goals

**Goals**

- One immutable declaration per estimator, readable before fit and by tooling.
- Closed vocabulary that lines up 1:1 with the parity memory tiers.
- Source-kind compatibility enforced before the first element is read.
- An honest `FullMemory` declaration with a named alternative and a typed refusal.
- A conformance harness that makes a missing or drifting declaration a test failure.

**Non-Goals**

- Implementing `fit` from a mappable source (that is `mappable-fit-path`).
- Implementing streaming versions of estimators that do not have one.
- Changing the resolution arithmetic or the existing plan shape.
- Per-estimator memory budgets and thresholds (each algorithm change owns its budget).

## Declaration shape

A trait, not a free constant, so the declaration is reachable through `dyn` trait objects
(the pipeline graph and the stacking/multioutput meta-estimators hold boxed stages):

```rust
pub trait SKDeclaredAccessPattern {
    fn access_pattern(&self) -> SKAccessPattern;
    fn approximate_alternative(&self) -> Option<&'static str> {
        None
    }
}
```

- `access_pattern` is pure and `&self`: no state, no data, no operation.
- `approximate_alternative` is `Some` exactly when the pattern is `FullMemory`; the
  conformance harness asserts that pair.
- Estimators implement it next to `SKUnsupervisedFit`/`SKSupervisedFit`; builders do not
  expose a setter, because the pattern is a property of the algorithm, not a knob.

## Vocabulary and tier mapping

| `SKAccessPattern` | Parity tier | Meaning | Accepted sources |
|---|---|---|---|
| `Sequential` | `streaming` | single pass or incremental update | sequential lazy source, or an in-memory view |
| `RandomAccess` | `mappable` | positional access by index | mappable source, or an in-memory view |
| `Iterative` | `chunked` | several passes over the same logical data | reopenable lazy source, mappable source, or an in-memory view |
| `FullMemory` | `full-memory` | whole design resident (quadratic/cubic) | in-memory view only |

`FullMemory` is added to the existing enum. In-memory views remain accepted by every
pattern: the declaration describes the out-of-core capability, not a restriction on the
classic path.

## Enforcement point

A single shared helper validates compatibility at the operation boundary, before resolution
and before any read:

```text
operation(features)
  ├─ declared = estimator.access_pattern()
  ├─ sk_validate_source_kind(declared, source_kind)?   // before any element is read
  ├─ plan = sk_resolve_execution_plan(intent, context_with(declared))
  └─ run(plan, features)
```

`FullMemory` plus a non-in-memory source returns the existing
`SKError::UnsupportedOperation { operation, reason }` with the reason naming
`approximate_alternative()`. No new error variant is introduced.

## Retrofit inventory

The hardcoded assignments to replace (one declaration per estimator, one read per
operation):

- `sciencekit_preprocessing`: standard, min-max and robust scalers, one-hot encoder,
  polynomial features, missing indicator → `Sequential`.
- `sciencekit_impute`: simple imputer, KNN imputer → `Iterative` (KNN imputer needs the
  training design; simple imputer streams).
- `sciencekit_linear_model`: the OLS/ridge/lasso/elastic-net/logistic/SGD family →
  `Sequential` where a streaming path exists, `Iterative` where the solver revisits the
  data.
- `sciencekit_decomposition`: PCA and truncated SVD → `Iterative`/`RandomAccess` per solver,
  incremental PCA → `Sequential`.
- `sciencekit_cluster`: KMeans → `Iterative`, MiniBatchKMeans → `Sequential`, DBSCAN →
  `RandomAccess`.
- `sciencekit_neighbors`, `sciencekit_tree`, `sciencekit_ensemble`: `RandomAccess`
  (random sampling for bootstrap is positional over a mappable source).
- `sciencekit_svm`: `Iterative`; `sciencekit_pipeline`: the pipeline declares the union of
  its stages, exposed as the weakest common pattern.

The retrofit is mechanical but touches every crate; the conformance harness is what proves
it complete.

## Conformance harness

Rust has no runtime enumeration of types, so the workspace test uses an explicit registry
macro invoked beside each estimator:

```rust
sk_declare_access_pattern!(SKStandardScaler<F>, Sequential);
sk_declare_access_pattern!(SKAgglomerativeClustering<F>, FullMemory, "Nystroem + KMeans");
```

A workspace test (the umbrella crate already aggregates every model crate as a dev
dependency for `export_coverage`) asserts:

1. every registry entry's pattern matches `approximate_alternative` presence;
2. every estimator exported from each crate appears in the registry (the export-coverage
   test pattern is reused);
3. every entry declares a tier that equals its `parity.yaml` row.

## Observability

`SKOperationAttributes` gains `access_pattern: SKAccessPattern`; the span records
`sk.access_pattern`. This is a struct-literal change at every operation call site, so it
lands together with the declaration retrofit in the same change.

## Risks

- **Enum growth is breaking** for exhaustive `match` on `SKAccessPattern`; the enum is
  internal-facing and pre-1.0, and every match site is updated in this change.
- **Retrofit breadth:** ~20 estimator families. Mitigated by per-crate tasks and the
  conformance test.
- **Pipeline composition:** a pipeline of stages with different patterns needs a documented
  join rule (weakest common pattern); the rule is part of this change, not left to the
  pipeline change.
