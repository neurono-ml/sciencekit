# Proposal

## Why

The library is a Rust API only. The PRD records a workflows CLI as the missing production
entry point: fit, predict, evaluate, export and reproducible multi-step workflows without
writing Rust. The CLI should reuse the umbrella crate, pipelines, model selection and
interop formats rather than reimplement them.

## What Changes

- New `sciencekit-cli` crate with a `sciencekit` binary (`clap 4.6`, MSRV 1.85).
- Commands: `fit`, `predict`, `transform`, `evaluate`, `export`, `inspect` and `run`.
- A versioned workflow schema (JSON first; TOML and optional YAML) describing dataset
  sources, feature pipeline, estimator, search, metrics and output artifacts.
- Data sources: CSV and Parquet (via the interop sources); optional Polars/DataFusion
  features.
- Model output/input through Safetensors/JSON/ONNX; metrics through `sciencekit_metrics`.
- Configuration precedence: CLI flags > environment > workflow file > defaults.
- Structured logging/telemetry integration and stable exit codes.
- Validation/dry-run mode that resolves a workflow without executing it.
- Explicit non-goals: a long-running model server, distributed execution, GUI, workflow
  scheduling and cloud storage.
- Companion tests: CLI integration tests with temporary files, golden workflows, error/
  exit-code cases, dry-run and output artifact checks.

## Capabilities

### New Capabilities

- `workflows-cli`: commands, workflow schema, configuration precedence, dataset/model
  adapters, telemetry and exit codes.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_cli` workspace member with a binary.
- **APIs:** additive CLI.
- **Dependencies:** `clap 4.6`, `serde`/`serde_json`/`toml`, optional `serde_yaml_ng`,
  `sciencekit` umbrella, interop, pipelines and metrics.
- **Downstream:** gives users a reproducible command-line entry point and a workflow format
  for automation.
- **Acceptance criteria:** every command runs on small fixtures, workflows are
  reproducible, errors map to non-zero exit codes and dry-run resolves without side
  effects.
