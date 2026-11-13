# Tasks

## 1. Crate and re-exports

- [x] 1.1 Create `crates/sciencekit` with `Cargo.toml`, `lib.rs` and the family modules;
  add it to `[workspace.members]`; verify `cargo test -p sciencekit` compiles.
- [x] 1.2 Re-export every sub-crate's stable public surface and add compile tests for
  common types and family modules; verify with `cargo test -p sciencekit reexports`.
- [x] 1.3 Implement the prelude and a basic build→fit→predict example test; verify with
  `cargo test -p sciencekit prelude`.

## 2. Features and docs

- [x] 2.1 Add the optional integration features and a feature-matrix build test; verify
  with `cargo check -p sciencekit --no-default-features` and each feature.
- [x] 2.2 Write crate-level documentation and docs.rs metadata; verify with
  `cargo doc -p sciencekit --no-deps`.
- [x] 2.3 Add a `Development Explained` subsection on the umbrella and prelude; verify with
  `mdbook build docs`.

## 3. Gates and review

- [x] 3.1 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 3.2 Update `CHANGELOG.md` under `Unreleased → Added` for the umbrella crate.
- [ ] 3.3 Independent review by a separate agent validating every scenario in
  `umbrella-crate`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
