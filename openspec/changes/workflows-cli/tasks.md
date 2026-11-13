# Tasks

## 1. Crate and command surface

- [x] 1.1 Create `crates/sciencekit_cli` with `main.rs`, `clap` commands and exit-code
  mapping; verify `cargo test -p sciencekit_cli cli`.
- [x] 1.2 Implement `fit`, `predict`, `transform`, `evaluate`, `export` and `inspect` on
  small CSV fixtures; verify with integration tests.
- [ ] 1.3 Implement the `run` command and workflow step dispatch; verify with a golden
  workflow.

## 2. Workflow schema and config

- [ ] 2.1 Implement the versioned workflow schema (JSON/TOML) and unknown-field/version
  validation; verify with `cargo test -p sciencekit_cli workflow_schema`.
- [ ] 2.2 Implement configuration precedence (flags > env > file > defaults) and dry-run
  output; verify with `cargo test -p sciencekit_cli config`.
- [ ] 2.3 Add the optional YAML feature after the `serde_yaml_ng` spike; verify with a
  YAML fixture.

## 3. Adapters and telemetry

- [ ] 3.1 Implement CSV/Parquet dataset adapters and clear missing-feature errors; verify
  with feature-gated fixtures.
- [ ] 3.2 Implement Safetensors/JSON/ONNX model and prediction adapters; verify with
  round-trip fixtures.
- [ ] 3.3 Integrate `sciencekit_telemetry`, flushing on every exit path; verify with an
  in-memory endpoint test.

## 4. Acceptance and review

- [ ] 4.1 Add CLI integration tests for every command, invalid workflows, missing
  features, dry-run and exit codes; verify with `cargo test -p sciencekit_cli`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on the workflow schema and config precedence); verify
  with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `workflows-cli`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
