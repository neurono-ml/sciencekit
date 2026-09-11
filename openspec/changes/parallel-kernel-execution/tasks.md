## 1. Resolution rule (size-aware parallelism + memory guard)

- [x] 1.1 TDD: add failing tests to `execution_tests.rs` proving `sk_resolve_execution_plan` derives `parallelism` from work-unit size and grain — small in-memory dataset → 1; streaming batch hint below one grain → 1; memory-mapped → `cores`; determinism on repeated resolution with identical context (spec `execution-planning`).
- [x] 1.2 Minimal implementation of the size-aware rule in `crates/sciencekit_common/src/execution/resolve.rs` with per-kernel grain inputs documented; keep resolution pure (no machine reads).
- [x] 1.3 TDD: failing tests for the double-buffer guard — `2 × batch > available_memory` (simulated) → structured error naming both quantities, raised before processing; automatic intent success path when the guard holds (spec `execution-planning`).
- [x] 1.4 Minimal implementation of the guard (new `SKError` variant in the central taxonomy); commit + sync `-openspec` worktree (`tasks.md` check, touched spec artifacts).

## 2. Kernels honor the plan

- [x] 2.1 TDD: failing tests in `kernels_tests.rs` for the new `parallelism` parameter — all four kernels (`sk_elementwise_transform`, `sk_binary_combine`, `sk_axis_sum`, `sk_scale_in_place`): sequential path at `parallelism = 1` (no dispatch), parallel path matching sequential reference within float tolerance on large arrays, and one array shape driven through both plans agreeing (spec `higher-order-kernels`).
- [x] 2.2 Minimal implementation: signatures gain trailing `parallelism: usize`; `1 → azip!`/`zip_mut_with` sequential forms, `> 1 → par_azip!`; bounds tightened (`Sync`/`Send` closures) following the `gemm` precedent; update all in-workspace callers.
- [x] 2.3 TDD: failing test for race-free parallel axis-0 — wide/tall matrix, parallel column sums equal sequential reference within tolerance; per-chunk partial accumulation, no shared mutable accumulator (spec `higher-order-kernels`).
- [x] 2.4 Minimal implementation of the axis-0 partial-accumulation path; commit + sync `-openspec` worktree.

## 3. Grain calibration (provenance, no magic constants)

- [x] 3.1 Add `criterion` benches under `crates/sciencekit_math/benches/` covering the calibration matrix: each kernel × cheap/medium/expensive closures × sizes 10¹–10⁷; sequential-vs-parallel crossover extraction; raw artifacts under `temporary/YYYY-MM-DD/parallel-kernel-execution/`.
- [x] 3.2 Record each kernel's grain constants beside the kernel with provenance comment (protocol, machine, date); re-run confirmation on all four kernels (spec `higher-order-kernels` provenance scenario); wire recorded grains into the resolution inputs of 1.2 if they differ from the provisional values; commit + sync `-openspec` worktree.

## 4. Streaming driver (I/O ∥ CPU)

- [ ] 4.1 TDD: failing tests in a new `execution/*_tests.rs` companion for the driver contract over an instrumented `SKLazySource` — prefetch ordering (source asked for batch k+1 before batch k completes, via read log), ordered delivery of owned batches, buffer depth respected (spec `streaming-executor`).
- [ ] 4.2 Minimal implementation of the prefetch pipeline in `crates/sciencekit_common/src/execution/` (`streaming_driver.rs` per design): dedicated I/O thread, bounded double buffer, owned-batch handoff, compute on the rayon pool per plan parallelism.
- [ ] 4.3 TDD: failing tests for control flow — early stop via `Continue | Stop` callback (no reads beyond stop), structured error propagation from an intermediate read failure (earlier effects preserved, taxonomy error returned), exactly-once final batch delivery on a finite source (spec `streaming-executor`).
- [ ] 4.4 Minimal implementation of the control-flow paths; commit + sync `-openspec` worktree.

## 5. Integration, gates and documentation

- [ ] 5.1 Concurrency acceptance (PRD §8.7): tests running the driver and parallel kernels under a shared rayon pool from multiple threads; small-data and large-data acceptance passes; sequential fallback (`parallelism = 1`) verified end-to-end; model-export/metric smoke check for an existing streaming consumer example.
- [ ] 5.2 CI gates locally: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets` (warnings as errors), `cargo test --workspace`; fix until green.
- [ ] 5.3 mdBook documentation: update `docs/src/` chapters (math-kernel and execution/architecture pages) for the new parallelism policy, grain provenance, kernel signature and streaming driver; Mermaid diagrams rendered error-free; build `mdbook build docs`; doc branch + PR per the documentation rule; commit + sync `-openspec` worktree final state.
