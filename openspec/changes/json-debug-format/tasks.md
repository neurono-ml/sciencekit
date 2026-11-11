# Tasks

## 1. Schema and writer

- [x] 1.1 Add the JSON schema types and a fixture test for a hand-written state; verify
  with `cargo test -p sciencekit_interop json_schema`.
- [x] 1.2 Implement pretty and compact writing with deterministic key ordering; verify
  with `cargo test -p sciencekit_interop json_writer`.

## 2. Reader and validation

- [x] 2.1 Implement reading back into `SKModelState` and a round-trip test; verify with
  `cargo test -p sciencekit_interop json_round_trip`.
- [x] 2.2 Add malformed JSON, unknown major version, duplicate name, dtype and shape
  mismatch tests; verify with `cargo test -p sciencekit_interop json_validation`.
- [x] 2.3 Add pretty-vs-compact equivalence, large-tensor and concurrent-read tests; verify
  with `cargo test -p sciencekit_interop json_concurrency`.

## 3. Acceptance and review

- [ ] 3.1 Add `acceptance_tests/` covering tiny and large states, all dtypes, class tables,
  metadata and corrupt inputs; verify with `cargo test -p sciencekit_interop`.
- [ ] 3.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on the JSON schema and its debug purpose); verify with
  `mdbook build docs`.
- [ ] 3.4 Independent review by a separate agent validating every scenario in
  `json-debug-format`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
