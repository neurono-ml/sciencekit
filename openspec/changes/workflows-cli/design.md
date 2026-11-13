# Design

## Context

See `proposal.md` — Why. The umbrella crate re-exports the library; interop crates read/write
model artifacts; pipelines/model-selection/metrics supply the workflow steps. `clap 4.6`
has MSRV 1.85. The CLI is a thin orchestration layer, not a second implementation.

## Goals / Non-Goals

**Goals:**
- A small, predictable CLI surface.
- A versioned workflow schema that composes existing APIs.
- Reproducibility and clear errors.

**Non-Goals:**
- A server, scheduler, distributed execution or GUI.
- Reimplementing estimators/metrics in the CLI.
- Cloud storage.

## Decisions

### Decision 1 — `sciencekit-cli` crate and binary

The crate depends on the umbrella crate and optional interop features. `main` parses
`clap` commands, loads configuration, initializes telemetry, runs the command and maps
errors to exit codes.

### Decision 2 — Workflow schema

```json
{
  "version": 1,
  "dataset": { "format": "csv", "path": "data.csv", "target": "y" },
  "pipeline": [ { "transformer": "standard_scaler", "with_mean": true } ],
  "estimator": { "name": "ridge", "alpha": 1.0 },
  "search": { "kind": "grid", "folds": 5, "scoring": "r2" },
  "metrics": ["r2", "rmse"],
  "output": { "model": "model.safetensors", "predictions": "pred.csv" }
}
```

The schema is versioned; the first version supports JSON and TOML. YAML is an optional
feature after a `serde_yaml_ng` spike. Unknown fields within a known version are rejected
to catch typos.

### Decision 3 — Resolution and precedence

A `SKCliConfig` merges defaults, workflow file, environment (`SCIENCEKIT_*`) and CLI flags
in that order. `--dry-run` prints the effective config and validates paths/features without
writing artifacts.

### Decision 4 — Data and model adapters

CSV/Parquet via the interop sources; model artifacts via Safetensors/JSON/ONNX features.
Each adapter validates the feature flag and returns a clear error when missing. Prediction
outputs are CSV/Parquet/JSON as configured.

### Decision 5 — Telemetry and exit codes

If an OTLP endpoint is configured, initialize `sciencekit_telemetry` and flush on exit.
Exit codes: `0` success, `2` usage/validation, `3` data/model I/O, `4` unsupported feature,
`1` runtime. Logs go to stderr; command output/artifacts to stdout/files.

### Decision 6 — File layout

```
crates/sciencekit_cli/src/
  main.rs
  cli/{mod.rs, commands.rs, args.rs, exit_codes.rs}
  config/{mod.rs, precedence.rs, env.rs, dry_run.rs}
  workflow/{mod.rs, schema.rs, version.rs, runner.rs, steps.rs}
  io/{mod.rs, dataset.rs, model.rs, predictions.rs}
  telemetry.rs
  tests/ (integration)
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines. Integration
tests use `assert_cmd`-style temp-dir fixtures.

## Risks / Trade-offs

- **Workflow schema can become a language** → versioned, small first version, reject
  unknown fields.
- **Feature matrix for formats** → explicit errors and a capability matrix in `--help`.
- **CLI output stability** → documented exit codes and machine-readable JSON output.
- **YAML dependency** → optional feature and spike.
- **Telemetry lifecycle** → flush on every exit path.

## Migration Plan

Additive: new crate and binary. No existing API changes.

## Open Questions

None blocking. A server/serve command and workflow scheduling are future changes.
