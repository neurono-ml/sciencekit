# Tasks

## 1. Spike and crate

- [ ] 1.1 Spike `cudarc` on MSRV 1.85 and driver/runtime behavior; record the chosen pin.
- [ ] 1.2 Create `crates/sciencekit_gpu_cuda` with the `gpu-cuda` feature and dispatcher
  modules; verify `cargo check -p sciencekit_gpu_cuda --features gpu-cuda`.
- [ ] 1.3 Implement device enumeration, compute-capability filtering and deterministic
  selection with a no-driver test; verify with
  `cargo test -p sciencekit_gpu_cuda device`.

## 2. Buffers and kernels

- [ ] 2.1 Implement buffers, async upload/download and transfer errors; verify with
  `cargo test -p sciencekit_gpu_cuda buffer`.
- [ ] 2.2 Implement pairwise-distance kernels and CPU-parity tests; verify with
  `cargo test -p sciencekit_gpu_cuda pairwise`.
- [ ] 2.3 Implement GEMM, elementwise and tree-prediction kernels with parity tests; verify
  with `cargo test -p sciencekit_gpu_cuda kernels`.
- [ ] 2.4 Implement NVRTC/PTX compilation and caching with build-log errors; verify with
  `cargo test -p sciencekit_gpu_cuda compile`.

## 3. Fallback and acceptance

- [ ] 3.1 Implement feature-disabled/no-driver/unsupported-kernel fallback and reason
  reporting; verify with `cargo test -p sciencekit_gpu_cuda fallback`.
- [ ] 3.2 Add small/large parity, concurrency and `Send + Sync` tests; verify with
  `cargo test -p sciencekit_gpu_cuda concurrency`.
- [ ] 3.3 Add hardware-gated acceptance tests that skip cleanly without a CUDA GPU; verify
  with `cargo test -p sciencekit_gpu_cuda --features gpu-cuda`.
- [ ] 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on CUDA kernels and streams); verify with `mdbook build
  docs`.
- [ ] 3.6 Independent review by a separate agent validating every scenario in
  `cuda-backend`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
