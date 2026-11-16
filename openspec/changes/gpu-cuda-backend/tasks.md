# Tasks

## Status

**Deferred — see ADR #408.** No CUDA toolkit (including NVRTC) and no machine able to
compile or verify a kernel is available to this project: the development hosts carry only
the driver (`libcuda.so.1` 580.178.04), with no `nvcc`, no NVRTC headers and no CUDA
development package. The plan below is preserved as design material, not as committed
near-term work. Tracking issue #409 states what a future contribution must satisfy before
this change is reactivated.

## 1. Spike and crate

- [ ] DEFERRED (2026-11-16): 1.1 Spike `cudarc` on MSRV 1.85 and driver/runtime behavior; record the
  chosen pin.
- [ ] DEFERRED (2026-11-16): 1.2 Create `crates/sciencekit_gpu_cuda` with the `gpu-cuda` feature and
  dispatcher modules; verify `cargo check -p sciencekit_gpu_cuda --features gpu-cuda`.
- [ ] DEFERRED (2026-11-16): 1.3 Implement device enumeration, compute-capability filtering and
  deterministic selection with a no-driver test; verify with `cargo test -p sciencekit_gpu_cuda
  device`.

## 2. Buffers and kernels

- [ ] DEFERRED (2026-11-16): 2.1 Implement buffers, async upload/download and transfer errors;
  verify with `cargo test -p sciencekit_gpu_cuda buffer`.
- [ ] DEFERRED (2026-11-16): 2.2 Implement pairwise-distance kernels and CPU-parity tests; verify
  with `cargo test -p sciencekit_gpu_cuda pairwise`.
- [ ] DEFERRED (2026-11-16): 2.3 Implement GEMM, elementwise and tree-prediction kernels with parity
  tests; verify with `cargo test -p sciencekit_gpu_cuda kernels`.
- [ ] DEFERRED (2026-11-16): 2.4 Implement NVRTC/PTX compilation and caching with build-log errors;
  verify with `cargo test -p sciencekit_gpu_cuda compile`.

## 3. Fallback and acceptance

- [ ] DEFERRED (2026-11-16): 3.1 Implement feature-disabled/no-driver/unsupported-kernel fallback
  and reason reporting; verify with `cargo test -p sciencekit_gpu_cuda fallback`.
- [ ] DEFERRED (2026-11-16): 3.2 Add small/large parity, concurrency and `Send + Sync` tests; verify
  with `cargo test -p sciencekit_gpu_cuda concurrency`.
- [ ] DEFERRED (2026-11-16): 3.3 Add hardware-gated acceptance tests that skip cleanly without a
  CUDA GPU; verify with `cargo test -p sciencekit_gpu_cuda --features gpu-cuda`.
- [ ] DEFERRED (2026-11-16): 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace
  --all-targets -- -D warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] DEFERRED (2026-11-16): 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on CUDA kernels and streams); verify with `mdbook build docs`.
- [ ] DEFERRED (2026-11-16): 3.6 Independent review by a separate agent validating every scenario in
  `cuda-backend`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
