# Proposal: bootstrap-workspace

## Why

The repository is freshly initialized and has no Cargo workspace, pinned toolchain or CI — the structural prerequisite of Phase 0 item 1 of the PRD (§12). Every subsequent change (starting with `sciencekit_common`) needs a reproducible build environment (Rust 1.85 / edition 2024), automated quality gates and the Apache-2.0 license defined before any code exists.

## What Changes

- Creation of the **root Cargo workspace** (`Cargo.toml` with `[workspace]`, empty members initially — each sub-crate is born in its own change, keeping PRs small).
- Toolchain pinning via `rust-toolchain.toml`: exact channel **1.85**, minimal profile, host target — guaranteeing edition 2024 and a reproducible MSRV.
- Addition of the **Apache-2.0 license** (`LICENSE`).
- Creation of the **CI workflow** (GitHub Actions) with the agreed gates:
  - `cargo fmt --check`;
  - `cargo clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace` (includes companion `*_tests.rs` modules and doctests);
  - example build + test (`cargo build --examples && cargo test --examples`);
  - **MSRV gate**: separate job compiling with the exact 1.85 toolchain.
- Point adjustments to `.gitignore` for Rust build artifacts.

Out of scope: creating sub-crates, configuring branch protection on GitHub (remote administrative decision), release profiles and auxiliary tooling (`cargo-deny`, coverage).

## Capabilities

### New Capabilities

- `workspace-bootstrap`: verifiable behaviors of the repository foundation — pinned, reproducible toolchain; mandatory CI gates for any PR; Apache-2.0 license present and declared in manifests.

### Modified Capabilities

(none — no existing specs)

## Impact

- **Files:** repository root (`Cargo.toml`, `rust-toolchain.toml`, `LICENSE`, `.gitignore`) and `.github/workflows/ci.yml`.
- **Dependencies:** no code dependencies in this change (empty workspace).
- **Systems:** GitHub Actions starts running on pull requests; all future changes assume these gates as preconditions.
- **Acceptance criteria (PRD §8.7/§10.3):** the algorithmic criteria (lots/little data, concurrency, export + metrics) do not apply yet since no algorithms exist; they apply starting from the first estimator change (Phase 1). Acceptance for this change is: a clean clone compiles, tests and passes all gates on the exact 1.85 toolchain, with CI running them automatically on PRs.
