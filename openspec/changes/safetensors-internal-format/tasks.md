# Tasks

## 1. Crate and standard format

- [x] 1.1 Create `crates/sciencekit_interop` with `Cargo.toml`, `lib.rs` and the
  `safetensors` dispatcher; add it to `[workspace.members]`; verify
  `cargo test -p sciencekit_interop` compiles.
- [x] 1.2 Implement metadata header encode/decode and a single-file round-trip for a mock
  state; verify with `cargo test -p sciencekit_interop safetensors_round_trip`.
- [x] 1.3 Add dtype/shape/offset validation and corrupt-file tests (truncated, wrong
  offset, duplicate name); verify with
  `cargo test -p sciencekit_interop safetensors_validation`.

## 2. Sharding and padding

- [x] 2.1 Implement `SKShardPlan`, shard writing and the index reader, with a
  large-state round-trip; verify with `cargo test -p sciencekit_interop safetensors_shards`.
- [x] 2.2 Implement header padding, in-place append and the overflow-to-new-shard path;
  verify with `cargo test -p sciencekit_interop safetensors_padding`.
- [ ] 2.3 Add missing-shard, duplicate-shard and future-version tests; verify with
  `cargo test -p sciencekit_interop safetensors_shards`.

## 3. Compression

- [ ] 3.1 Add optional gzip, Brotli and Snappy features and transparent compressed
  round-trips; verify with `cargo test -p sciencekit_interop --features compression-gzip`
  (and the other features).
- [ ] 3.2 Add a missing-feature error test and magic-byte detection tests; verify with
  `cargo test -p sciencekit_interop safetensors_compression`.

## 4. Acceptance and review

- [ ] 4.1 Add `acceptance_tests/` covering tiny and large models, sharded and padded
  files, every compression feature, concurrent reads and corruption cases; verify with
  `cargo test -p sciencekit_interop`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on the extended Safetensors layout, sharding and
  padding); verify with `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `safetensors-format`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
