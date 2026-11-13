# Tasks

## 1. Spike and crate

- [ ] 1.1 Spike `hip-sys` and maintained HIP wrappers on MSRV 1.85; record the choice.
- [ ] 1.2 Create `crates/sciencekit_gpu_rocm` with the `gpu-rocm` feature and dispatcher
  modules; verify `cargo check -p sciencekit_gpu_rocm --features gpu-rocm`.
- [ ] 1.3 Implement device enumeration, architecture filtering and deterministic selection
  with a no-runtime test; verify with
  `cargo test -p sciencekit_gpu_rocm device`.

## 2. Buffers and kernels

- [ ] 2.1 Implement buffers, stream upload/download and transfer errors; verify with
  `cargo test -p sciencekit_gpu_rocm buffer`.
- [ ] 2.2 Implement pairwise-distance kernels and CPU-parity tests; verify with
  `cargo test -p sciencekit_gpu_rocm pairwise`.
- [ ] 2.3 Implement GEMM, elementwise and tree-prediction kernels with parity tests; verify
  with `cargo test -p sciencekit_gpu_rocm kernels`.
- [ ] 2.4 Implement kernel compilation/caching and build-log errors; verify with
  `cargo test -p sciencekit_gpu_rocm compile`.

## 3. Fallback and acceptance

- [ ] 3.1 Implement feature-disabled/no-runtime/unsupported-kernel fallback and reason
  reporting; verify with `cargo test -p sciencekit_gpu_rocm fallback`.
- [ ] 3.2 Add small/large parity, concurrency and `Send + Sync` tests; verify with
  `cargo test -p sciencekit_gpu_rocm concurrency`.
- [ ] 3.3 Add hardware-gated acceptance tests that skip cleanly without a ROCm GPU; verify
  with `cargo test -p sciencekit_gpu_rocm --features gpu-rocm`.
- [ ] 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on HIP kernels and streams); verify with `mdbook build
  docs`.
- [ ] 3.6 Independent review by a separate agent validating every scenario in
  `rocm-backend`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
