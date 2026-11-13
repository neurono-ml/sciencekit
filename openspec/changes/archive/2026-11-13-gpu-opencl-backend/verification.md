# Independent verification — `gpu-opencl-backend`

Reviewer: separate agent (task 3.6), not the author of the code. Code worktree
`temporary/worktrees/feat/gpu-opencl-backend` (PR #369), planning worktree
`temporary/worktrees/feat/gpu-opencl-backend-openspec` (PR #229), book worktree
`docs/gpu-opencl-backend-book` (PR #370).

## Environment

| Item | Value |
|---|---|
| Host device | `NVIDIA GeForce RTX 3070 Ti Laptop GPU` via ICD loader (`Platform #0: NVIDIA CUDA`) |
| Device memory | 7840 MiB, `double_fp_config` reports f64 support |
| Toolchain | `rustc 1.85.1` / `cargo 1.85.1` (workspace `rust-toolchain.toml` pin) |
| Build env | `CARGO_TARGET_DIR=/home/claudino/.cache/sciencekit-target-gpu-opencl`, `CARGO_PROFILE_DEV_DEBUG=0` |

The device is genuinely present, so every device test below executed the device
path; none passed by skipping.

## Verdict matrix — `openspec/changes/gpu-opencl-backend/specs/opencl-backend`

| Requirement / scenario | Named test | Verdict |
|---|---|---|
| Discovery — device list (stable order, ≥1 compatible device) | `opencl::device::device_tests::enumeration_is_stable_and_describes_opencl_devices`, `...::selection_is_deterministic_and_honours_the_device_index`, `opencl::backend::fallback_tests::a_real_device_serves_the_contract_surface` | PASS (stability asserted; non-empty asserted by the fallback and contract tests) |
| Discovery — no platform (empty list + CPU fallback) | `opencl::device::device_tests::an_empty_inventory_has_no_device_to_select`, `opencl::backend::fallback_tests::fallback_discovery_matches_the_enumeration` (feature-off run) | PASS (simulated through an empty inventory; this host has an ICD loader, so a truly platform-less host is not available) |
| Buffers — round-trip within tolerance | `opencl::buffer::buffer_tests::buffers_round_trip_without_loss`, `...::degenerate_shapes_round_trip` | PASS (exact equality, both precisions and 1×n/n×1) |
| Buffers — out-of-memory structured error | `opencl::buffer::buffer_tests::an_oversized_allocation_reports_a_structured_error` | PASS |
| Kernels — pairwise parity (CPU reference) | `...::pairwise_metrics_match_the_cpu_reference` (4 metrics, 13×11 / 17×11), `...::large_pairwise_matches_the_cpu_reference` (257×33 / 129×33), `...::single_precision_pairwise_matches_within_tolerance` | PASS |
| Kernels — GEMM/elementwise/tree parity on non-square, non-power-of-two shapes | `...::gemm_matches_the_cpu_reference` (17×23×19, 64×96×32, 129×47×65), `...::elementwise_kernels_match_the_cpu_reference` (37×29), `...::tree_prediction_matches_the_cpu_reference` (5×2, hand-built tree incl. threshold tie) | PASS (numbers asserted against CPU references, not buffer creation) |
| Kernels — compile once per device, cached | `...::programs_are_cached_once_per_scalar_type` (new), `...::a_broken_program_reports_the_build_log` | PASS (same program reused via `Arc::ptr_eq`; separate entry per scalar type) |
| Kernels — compile failure reports the build log | `...::a_broken_program_reports_the_build_log` | PASS for the diagnostics; **WEAK** for the “falls back to CPU” clause — the backend returns `Conversion` and leaves the fallback decision to the planner, which no test in this crate exercises |
| Fallback — feature disabled, no OpenCL call | `cargo test -p sciencekit_gpu` feature-off run: `fallback_operations_report_structured_errors`, `fallback_discovery_matches_the_enumeration` | PASS (the module gates every symbol behind `#[cfg(feature = "opencl")]`; the off build enumerates nothing and every operation returns `DeviceUnavailable`) |
| Fallback — unsupported kernel | shared contract test `kernel::kernel_tests::an_unsupported_kernel_names_itself_in_the_error` | PASS (vacuous for OpenCL: `SKGpuKernel` has exactly the four kernels this backend implements, so no unsupported variant can reach it) |
| Fallback — honesty of the recorded kind/reason | `opencl::backend::fallback_tests::a_backend_without_a_device_records_the_fallback_reason`, `...::an_unavailable_operation_records_the_host_fallback_and_reason` (new), `opencl::backend::observability_tests::a_kernel_operation_records_backend_device_and_bytes` | PASS after finding **F2** |
| Concurrency — concurrent dispatch equals sequential | `opencl::backend::concurrency_tests::concurrent_pairwise_dispatches_agree`, `...::concurrent_transfers_agree` (8 threads, one `Arc<SKOpenClBackend>`) | PASS |
| Concurrency — `Send + Sync` | `...::backend_types_are_send_and_sync` | PASS |
| Large input parity | `...::large_pairwise_matches_the_cpu_reference`, `...::gemm_matches_the_cpu_reference` (129×47×65) | PASS |

## Findings

| # | Severity | Finding | Resolution |
|---|---|---|---|
| F1 | Critical (silent wrong numbers) | The program cache was keyed by kernel name only, while the compiled source is precision-specific. One backend dispatching `f32` and then the same kernel in `f64` received the `f32` program and produced garbage: observed deviation `157679605.90625` against the CPU reference. | Fixed in `c1a934d`; cache key now carries kernel name + scalar type, covered by `one_backend_serves_both_precisions_of_the_same_kernel` (failed before the fix, passes after). |
| F2 | High (dishonest observability) | `observability::operation_span` hardcoded `backend = opencl`. Operations on a backend with no device (feature off, no platform, bad index) opened an `opencl` span although no OpenCL work happened and the caller falls back to the CPU; no `fallback_reason` was recorded, so a trace could not tell the two apart. | Fixed in `8db8eea`; the span now records `opencl` only with an open session, otherwise the host fallback kind (`faer`) plus `SKFallbackReason`, mirroring `sk_resolve_operation_backend`. Covered by `an_unavailable_operation_records_the_host_fallback_and_reason`, which runs in both feature configurations. |
| F3 | Low (missing check) | The “compile once per device” requirement had no direct test; the cache hit was only implied by dispatch results. | Test added in `81fa946`; asserts program reuse and per-precision separation. |

No parity or honesty assertion was weakened: both fixes strengthen checks, and
all pre-existing parity tests still compare against CPU references.

## Resource safety

- Buffers, contexts, queues and programs are owned by `opencl3` RAII types; programs
  are cached as `Arc<Program>` in the session and released with it; no buffer or
  program leak path was found.
- No `unwrap`/`expect`/`panic!` on user input in non-test code (`grep` over
  `crates/sciencekit_gpu/src/opencl` returns none outside `*_tests.rs`).
- Failures map to `SKError` (`DeviceUnavailable`, `DeviceTransferFailure`,
  `ShapeMismatch`, `UnsupportedScalarType`, `Conversion`); no raw `cl_*` handle is
  exposed on the public surface (`device_identifier()` is crate-private).

## MSRV and pinning

- `opencl3 = "=0.12.0"` (exact pin, not a floating range); `Cargo.lock` resolves
  `opencl3 0.12.0` with `cl3 0.13.2` (runtime dynamic loader for `libOpenCL.so.1`).
- Every gate below ran on the pinned `rustc 1.85.1`, including the `opencl`
  feature build, so the pin still compiles at the MSRV.
- The CI MSRV job builds the workspace without optional features; the OpenCL
  feature is additionally verified here with the same 1.85 toolchain.

## Gates (all green, `CARGO_TARGET_DIR`/`CARGO_PROFILE_DEV_DEBUG` as above)

| Command | Result |
|---|---|
| `cargo fmt --all --check` | exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `cargo test --workspace` | exit 0 |
| `cargo test -p sciencekit_gpu` (feature off) | 50 passed, 0 failed |
| `cargo test -p sciencekit_gpu --features opencl` | 70 passed, 0 failed |
| `cargo test -p sciencekit_gpu_opencl --features gpu-opencl` | 1 passed, 0 failed |

## Device output (excerpt)

```
[opencl] device 0: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`, 7840 MiB, f64=true
[opencl] kernel parity runs on device 0: `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
[opencl] `squared-euclidean` matched the CPU reference
[opencl] `euclidean` matched the CPU reference
[opencl] `manhattan` matched the CPU reference
[opencl] `chebyshev` matched the CPU reference
[opencl] gemm 17x23x19 matched the CPU reference
[opencl] gemm 64x96x32 matched the CPU reference
[opencl] gemm 129x47x65 matched the CPU reference
[opencl] build log surfaced: OpenCL program build failed with code CL_BUILD_PROGRAM_FAILURE: <kernel>:1:25: error: expected parameter declarator
```

## Residual notes (not defects for this change)

- Parity is asserted in f64 for every kernel and in f32 for pairwise (plus the new
  mixed-precision sequence); f32 GEMM/elementwise/tree parity is covered by the
  same precision-generic source path but has no dedicated f32 test.
- The spec’s “kernel compile failure falls back to CPU” clause is satisfied at the
  planning boundary (structured `Conversion` error), not by an in-backend CPU
  recompute; this matches design decision 4.
- `opencl::device::device_tests::registering_the_inventory_resolves_opencl_to_a_device`
  mutates the process-wide inventory without the common test lock; a parallel
  inventory test could race, though all gates passed repeatedly.

Conclusion: **ACCEPT** after F1/F2 fixes. The device path runs on this host, parity
numbers agree with the CPU references, the fallback reports its real reason, and
the gates are green.
