# Tasks

## 1. Crate and configuration

- [x] 1.1 Create `crates/sciencekit_telemetry` with `Cargo.toml`, `lib.rs` and the config/
  resource/sampler modules; verify `cargo test -p sciencekit_telemetry config`.
- [x] 1.2 Implement the builder and validation (endpoint/transport/service/sampler/batch)
  with invalid-value tests; verify with `cargo test -p sciencekit_telemetry config`.

## 2. Pipelines

- [x] 2.1 Implement the trace bridge and in-memory exporter test for span names/attributes/
  errors; verify with `cargo test -p sciencekit_telemetry traces`.
- [x] 2.2 Implement the log bridge and capture structured events; verify with
  `cargo test -p sciencekit_telemetry logs`.
- [x] 2.3 Implement the metric bridge (duration/count/rows/columns/errors) and capture
  observations; verify with `cargo test -p sciencekit_telemetry metrics`.
- [x] 2.4 Add resource-attribute, sampler and batch configuration tests; verify with
  `cargo test -p sciencekit_telemetry pipeline`.

## 3. Lifecycle and acceptance

- [x] 3.1 Implement `init`, `force_flush`, `shutdown`, double-init error and drop behavior;
  verify with `cargo test -p sciencekit_telemetry lifecycle`.
- [x] 3.2 Add a no-op default test proving no exporter/traffic without initialization;
  verify with `cargo test -p sciencekit_telemetry noop`.
- [ ] 3.3 Add concurrent operation/export and shutdown tests; verify with
  `cargo test -p sciencekit_telemetry concurrency`.
- [ ] 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on OTLP pipelines and lifecycle); verify with
  `mdbook build docs`.
- [ ] 3.6 Independent review by a separate agent validating every scenario in
  `telemetry-export`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
