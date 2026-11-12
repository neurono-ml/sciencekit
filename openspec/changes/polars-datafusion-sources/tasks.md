# Tasks

## 1. Tabular contract

- [x] 1.1 Implement `SKTabularSchema`, `SKTabularSource`, the owned nominal block and the
  column-selection config, with hand-built schema/feature/categorical tests; verify with
  `cargo test -p sciencekit_interop tabular_source`.
- [x] 1.2 Implement missing-value mapping and unsupported-dtype errors; verify with
  `cargo test -p sciencekit_interop tabular_missing`.

## 2. Version spike and Polars

- [x] 2.1 Spike `polars 0.46/0.48` on MSRV 1.85 and record the chosen pin; verify with a
  minimal build.
- [x] 2.2 Implement `TryFrom<&DataFrame>`/`TryFrom<LazyFrame>` with numeric, categorical,
  missing, column-selection and batch tests; verify with
  `cargo test -p sciencekit_interop --features polars`.

## 3. DataFusion, Arrow and Parquet

- [ ] 3.1 Spike `datafusion 46` (and 44 fallback) plus matching `arrow`/`parquet` on MSRV
  1.85; record the pins.
- [ ] 3.2 Implement `TryFrom<&DataFrame>`/`RecordBatch` conversions and batch streaming;
  verify with `cargo test -p sciencekit_interop --features datafusion,arrow`.
- [ ] 3.3 Implement Parquet-through-DataFusion conversion and a fixture test; verify with
  `cargo test -p sciencekit_interop --features datafusion,parquet`.

## 4. Acceptance and review

- [ ] 4.1 Add feature-gated acceptance tests for tiny/large tables, numeric/categorical/
  missing schemas, streaming batch order, column selection and concurrent reads; verify
  with the matching feature combinations.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on tabular schemas, missing mapping and feature-gated
  integrations); verify with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `tabular-sources`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
