# Design

## Context

See `proposal.md` — Why. `sciencekit_interop` owns the format adapters; `SKModelState`
comes from `sciencekit_common`. The JSON format is debug-oriented, so fidelity and
readability matter more than compactness.

## Goals / Non-Goals

**Goals:**
- A readable schema with stable ordering.
- Exact round-trip into `SKModelState`.
- Pretty and compact modes.

**Non-Goals:**
- Binary/ONNX formats and Python bindings.
- Streaming JSON for enormous models.
- Cross-version migration.

## Decisions

### Decision 1 — Schema

```json
{
  "format_version": 1,
  "estimator": "SKStandardScaler",
  "dtype": "f64",
  "feature_count": 4,
  "class_labels": null,
  "hyperparameters": { "with_mean": true, "with_std": true },
  "tensors": {
    "standard_scaler.mean": { "dtype": "F64", "shape": [4], "values": [1.0, 2.0, 3.0, 4.0] }
  }
}
```

`BTreeMap` ordering keeps output deterministic. Values are flat arrays with an explicit
shape; the reader validates the product.

### Decision 2 — Writer/reader

The writer serializes the state with `serde_json`; pretty is the default and compact is an
option. The reader parses into a private schema, validates version/dtypes/shapes/names and
converts to `SKModelState`. Unknown fields are ignored within a known major version.

### Decision 3 — Large tensors

The format is not intended for huge production models; it documents that JSON expands
values and should be used for inspection/fixtures. A future streaming writer can be added
without changing the schema.

### Decision 4 — Validation

Reject malformed JSON, unknown major versions, duplicate names, dtype mismatches and
shape/value-count mismatches. Errors use the central taxonomy.

### Decision 5 — File layout

```
crates/sciencekit_interop/src/json/
  mod.rs, schema.rs, writer.rs, reader.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **JSON is large** → debug-only and documented.
- **Schema drift** → versioned and tested with a fixture.
- **Ordering** → `BTreeMap` everywhere.
- **Unknown fields** → ignored for forward compatibility within a major version.

## Migration Plan

Additive: new module in `sciencekit_interop`. No existing API changes.

## Open Questions

None blocking. Streaming JSON is a future convenience.
