# Tasks — bootstrap-workspace

## 1. Workspace foundation

- [ ] 1.1 Create root `Cargo.toml` with `[workspace]` (empty members), the resolver appropriate to the edition, and `[workspace.package]` declaring the Apache-2.0 license
- [ ] 1.2 Create `rust-toolchain.toml` pinning the exact `1.85` channel
- [ ] 1.3 Add `LICENSE` with the full Apache-2.0 text
- [ ] 1.4 Update `.gitignore` with Rust build artifacts (`target/`, among others)

## 2. CI workflow

- [ ] 2.1 Create the CI workflow triggered on pull requests, with a formatting job (`cargo fmt --check`)
- [ ] 2.2 Add a static analysis job with warnings promoted to errors (`cargo clippy --workspace --all-targets -- -D warnings`)
- [ ] 2.3 Add a workspace test job including documentation tests
- [ ] 2.4 Add an example build-and-test job
- [ ] 2.5 Add a dedicated MSRV job compiling with the exact 1.85 toolchain

## 3. Acceptance validation

- [ ] 3.1 On a clean clone of the worktree, run all gate commands locally and confirm green end-to-end (build, fmt, clippy, tests, examples)
- [ ] 3.2 Confirm edition 2024 is in force and that the automatically selected toolchain is the pinned 1.85
- [ ] 3.3 Open a PR and verify that all CI jobs run and pass on the pull request
