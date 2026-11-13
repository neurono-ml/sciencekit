# Proposal

## Why

A typed linear pipeline cannot express a model that consumes several feature views at once:
scaled numeric columns, one-hot categoricals and text features concatenated side by side.
`SKFeatureUnion` is the standard transformer for that, and it composes with `SKPipeline` as
a single stage.

## What Changes

- `SKFeatureUnion` in `sciencekit_pipeline`: fit several transformers on the same input,
  transform each and concatenate their outputs column-wise.
- Heterogeneous stages through a tuple-based `SKFeatureUnionStages` trait (implemented for
  tuples up to a documented arity) so each member keeps its own type; a runtime list of
  boxed members is a documented fallback for dynamic construction.
- Output representation policy: all-dense outputs concatenate to a dense block; any sparse
  output concatenates to CSR; no member densifies its input.
- The union implements the pipeline transformer contract, so it can be a stage in
  `SKPipeline` and receive the same fit/transform treatment.
- Validation: empty member list, duplicate member names (if names are exposed), feature
  count consistency, mismatched row counts, unsupported output representations and member
  errors.
- Accessors: member count, per-member output widths and the concatenated feature count.
- Explicit non-goals: column-name transforms, weights, drop/remainder policies, DAG
  unions, `memmap2` and export.
- Companion tests: hand-checkable concatenation, dense/sparse/mixed members, tuple and
  boxed construction, nested pipeline composition, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `feature-union`: parallel transformer fitting, output concatenation, the representation
  policy and the union's pipeline-stage contract.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_pipeline` gains the feature-union modules.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `ndarray`, `sprs`, `rayon`.
- **Downstream:** enables heterogeneous feature engineering inside a typed pipeline.
- **Acceptance criteria:** small and large data, dense/sparse/mixed members, concurrency,
  companion `*_tests.rs` modules. Export is deferred to Wave 7.
