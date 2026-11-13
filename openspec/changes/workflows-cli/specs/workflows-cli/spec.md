# Spec Delta

## Purpose

Defines the `sciencekit` CLI: commands, the workflow schema, configuration precedence,
dataset/model adapters, telemetry and exit codes.

## ADDED Requirements

### Requirement: Commands

The CLI SHALL expose `fit`, `predict`, `transform`, `evaluate`, `export`, `inspect` and
`run` commands, each with a documented argument surface.

#### Scenario: Fit writes a model

- **WHEN** `sciencekit fit` runs with a dataset and estimator configuration
- **THEN** it writes a model artifact and prints the chosen output path

#### Scenario: Predict reads a model

- **WHEN** `sciencekit predict` runs with a model artifact and input data
- **THEN** it writes predictions in the requested format

#### Scenario: Run executes a workflow

- **WHEN** `sciencekit run workflow.json` runs
- **THEN** every step executes in order and the declared outputs are written

### Requirement: Workflow schema

The workflow file SHALL carry a version, dataset source(s), feature pipeline, estimator or
search configuration, metric selection and output artifacts, and the CLI SHALL reject an
unknown major version.

#### Scenario: Versioned workflow

- **WHEN** a workflow with a supported version is loaded
- **THEN** it resolves and executes

#### Scenario: Future version

- **WHEN** a workflow declares an unsupported major version
- **THEN** the CLI exits with a structured error

### Requirement: Configuration precedence

Configuration SHALL resolve as CLI flags > environment variables > workflow file >
defaults, with the effective value visible in dry-run output.

#### Scenario: CLI overrides file

- **WHEN** a flag and a workflow value conflict
- **THEN** the flag wins

#### Scenario: Dry run

- **WHEN** `--dry-run` is passed
- **THEN** the CLI prints the resolved configuration and writes no artifacts

### Requirement: Data and model adapters

The CLI SHALL read CSV and Parquet sources and read/write Safetensors, JSON and ONNX model
artifacts through the interop crates, behind the corresponding features.

#### Scenario: Parquet input

- **WHEN** a workflow references a Parquet source with the feature enabled
- **THEN** the dataset is loaded through the interop source

#### Scenario: Missing feature

- **WHEN** an artifact format feature is disabled
- **THEN** the CLI returns a clear error naming the feature

### Requirement: Telemetry and exit codes

The CLI SHALL initialize telemetry when configured and SHALL map success, validation errors
and runtime errors to stable exit codes.

#### Scenario: Validation error

- **WHEN** a workflow is invalid
- **THEN** the CLI exits non-zero with a validation message and no partial artifact

#### Scenario: Telemetry configured

- **WHEN** an OTLP endpoint is configured
- **THEN** the CLI initializes the telemetry pipeline and flushes on exit
