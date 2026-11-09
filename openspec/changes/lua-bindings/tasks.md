# Tasks

## 1. Module and userdata

- [ ] 1.1 Create `crates/sciencekit_lua` with `mlua 0.10`, the module entry point and a
  `require` smoke test; verify with `cargo test -p sciencekit_lua module`.
- [ ] 1.2 Implement builder/model userdata, setters and `__gc`/`close`; verify with
  `cargo test -p sciencekit_lua userdata`.

## 2. Arrays and operations

- [ ] 2.1 Implement Lua table→array conversion and rectangularity validation; verify with
  `cargo test -p sciencekit_lua convert`.
- [ ] 2.2 Implement fit/predict/transform/score and array→table output; verify with
  `cargo test -p sciencekit_lua operations`.
- [ ] 2.3 Implement error mapping and double-free/GC tests; verify with
  `cargo test -p sciencekit_lua errors`.

## 3. Packaging and acceptance

- [ ] 3.1 Add the rockspec and a Lua 5.4 script covering every family; verify with the Lua
  test command.
- [ ] 3.2 Add large-array and concurrent-use tests; verify with
  `cargo test -p sciencekit_lua --release`.
- [ ] 3.3 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.4 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on userdata lifecycle and table conversion); verify
  with `mdbook build docs`.
- [ ] 3.5 Independent review by a separate agent validating every scenario in
  `lua-bindings`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
