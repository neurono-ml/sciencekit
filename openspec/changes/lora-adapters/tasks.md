# Tasks

## 1. Adapter core

- [x] 1.1 Create the LoRA adapter module with `SKLoraAdapter`, builder, rank/alpha and
  scaling tests; verify with `cargo test -p sciencekit_linear_model lora_adapter`.
- [x] 1.2 Implement `A`/`B` shape/initialization and frozen-base attachment, with a
  zero-adapter prediction test; verify with
  `cargo test -p sciencekit_linear_model lora_attach`.

## 2. Fitting and prediction

- [x] 2.1 Implement linear-regression adapter fitting and a training-loss reduction test;
  verify with `cargo test -p sciencekit_linear_model lora_fit`.
- [x] 2.2 Implement logistic adapter fitting and a classification-loss test; verify with
  `cargo test -p sciencekit_linear_model lora_logistic`.
- [x] 2.3 Implement adapter prediction and base-frozen invariance tests; verify with
  `cargo test -p sciencekit_linear_model lora_predict`.

## 3. Merge and persistence

- [ ] 3.1 Implement merge/unmerge with prediction-parity tests; verify with
  `cargo test -p sciencekit_linear_model lora_merge`.
- [ ] 3.2 Implement adapter export/import through the interop state and round-trip tests;
  verify with `cargo test -p sciencekit_linear_model lora_state`.
- [ ] 3.3 Add unsupported-family, shape, non-finite and concurrency tests; verify with
  `cargo test -p sciencekit_linear_model lora_validation`.

## 4. Acceptance and review

- [ ] 4.1 Add tiny/large data acceptance tests, deterministic fit and `Send + Sync` checks;
  verify with `cargo test -p sciencekit_linear_model`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on low-rank adapters and frozen bases); verify with
  `mdbook build docs`.
- [ ] 4.4 Independent review by a separate agent validating every scenario in
  `lora-adapters`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
