# Tasks

## 1. Spike and crate

- [x] 1.1 Spike `opencl3` vs `ocl` for MSRV 1.85 and ICD-loader behavior; record the choice
  and evidence.
- [x] 1.2 Create `crates/sciencekit_gpu_opencl` with the `gpu-opencl` feature and the
  dispatcher modules; verify `cargo check -p sciencekit_gpu_opencl --features gpu-opencl`.
- [x] 1.3 Implement device enumeration, capability reporting and deterministic selection
  with a no-platform test; verify with
  `cargo test -p sciencekit_gpu_opencl device`.

## 2. Buffers and kernels

- [x] 2.1 Implement device buffers, upload/download and shape/oom errors; verify with
  `cargo test -p sciencekit_gpu_opencl buffer`.
- [x] 2.2 Implement pairwise-distance kernels and CPU-parity tests for every metric; verify
  with `cargo test -p sciencekit_gpu_opencl pairwise`.
- [x] 2.3 Implement GEMM, elementwise and tree-prediction kernels with parity tests; verify
  with `cargo test -p sciencekit_gpu_opencl kernels`.
- [x] 2.4 Implement program compilation/caching and build-log error handling; verify with
  `cargo test -p sciencekit_gpu_opencl compile`.

## 3. Fallback and acceptance

- [x] 3.1 Implement feature-disabled/no-device/unsupported-kernel fallback paths and
  reason reporting; verify with `cargo test -p sciencekit_gpu_opencl fallback`.
- [x] 3.2 Add small/large parity, concurrency and `Send + Sync` tests; verify with
  `cargo test -p sciencekit_gpu_opencl concurrency`.
- [x] 3.3 Add hardware-gated acceptance tests (POCL when available) and skip cleanly when
  no device exists; verify with `cargo test -p sciencekit_gpu_opencl --features gpu-opencl`.
- [x] 3.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [x] 3.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on OpenCL kernels and transfers); verify with
  `mdbook build docs`.
- [x] 3.6 Independent review by a separate agent validating every scenario in
  `opencl-backend`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
