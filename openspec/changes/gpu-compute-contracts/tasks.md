# Tasks

## 1. Backend kinds and device contracts

- [ ] 1.1 Add `OpenCl`, `Cuda`, `Rocm` to `SKBackendKind`, update every workspace match and
  add display/parse tests; verify with `cargo test --workspace backend_kind`.
- [ ] 1.2 Create `crates/sciencekit_gpu` with `SKDeviceInfo`, capabilities and a mock
  backend enumeration; verify with `cargo test -p sciencekit_gpu device`.
- [ ] 1.3 Implement deterministic device selection and CPU fallback with multi-device and
  no-device tests; verify with `cargo test -p sciencekit_gpu selection`.

## 2. Buffers and kernels

- [ ] 2.1 Implement `SKDeviceBuffer`, upload/download and shape/dtype validation with a
  mock backend round-trip; verify with `cargo test -p sciencekit_gpu buffer`.
- [ ] 2.2 Implement the closed `SKGpuKernel` catalogue and dispatch contract with supported
  and unsupported kernel tests; verify with `cargo test -p sciencekit_gpu kernel`.
- [ ] 2.3 Add transfer-failure, unsupported-dtype and capacity tests; verify with
  `cargo test -p sciencekit_gpu buffer_validation`.

## 3. Planning and observability

- [ ] 3.1 Extend `SKExecutionContext` with a device inventory and make planning select GPU
  or CPU deterministically; verify with `cargo test -p sciencekit_common gpu_planning`.
- [ ] 3.2 Record the selected kind and fallback reason in spans and add a test; verify with
  `cargo test -p sciencekit_common gpu_span`.
- [ ] 3.3 Add concurrent planning/discovery tests and `Send + Sync` assertions; verify with
  `cargo test -p sciencekit_gpu concurrency`.

## 4. Documentation and review

- [ ] 4.1 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on device buffers, planning and fallback); verify with
  `mdbook build docs`.
- [ ] 4.2 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace`; fix every failure before pushing.
- [ ] 4.3 Independent review by a separate agent validating every scenario in
  `gpu-compute-contracts`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
