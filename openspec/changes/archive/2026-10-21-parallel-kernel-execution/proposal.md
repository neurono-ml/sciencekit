## Why

The execution-planning architecture already separates user intent (`SKExecutionMode::Automatic` default) from the resolved plan (`SKExecutionPlan { mode, parallelism, batch_size }`), but the resolved `parallelism` is never actually used by the compute kernels, and resolution itself ignores data size (it always returns `cpu_cores`). The consequence is incoherent parallelism today: `sk_scale_in_place` parallelizes even 4-element arrays while `sk_elementwise_transform`, `sk_binary_combine` and `sk_axis_sum` never parallelize at all — regardless of execution mode, dataset size, or machine. PRD §5.2/§5.3 prescribe that the automatic decision mechanism governs data parallelism; this change wires that decision through: resolution becomes size- and mode-aware, kernels honor the plan, and out-of-core streaming gains the I/O ∥ CPU overlap the batch contract was designed for.

## What Changes

- **Resolution rule per mode (PRD §5.3):** `sk_resolve_execution_plan` now decides `parallelism` from the context — `min(cores, ceil(unit_elements / grain))` where the work unit is the whole dataset (in-memory) or one batch (streaming); single-core or sub-grain units resolve to `1` (sequential); out-of-core memory-mapped shards across `cores`. The rule stays a pure, injectable, deterministic function (same intent + context → same plan).
- **Grain with provenance, no magic constants:** each kernel documents a `grain` (minimum elements per thread to amortize parallel dispatch overhead) measured via a criterion calibration protocol (cheap/medium/expensive closures × sizes 10¹–10⁷), recorded with machine and date, reused identically for in-memory and streaming regimes.
- **Kernels honor the plan:** the four dense kernels (`sk_elementwise_transform`, `sk_binary_combine`, `sk_axis_sum`, `sk_scale_in_place`) receive `parallelism: usize` (precedent: `SKMathBackend::gemm`) and run `par_azip!` only when `parallelism > 1`, otherwise plain `azip!` with zero parallel overhead. Axis-0 reduction parallelizes rows via per-thread partial accumulation (no shared-accumulator race).
- **Streaming executor (I/O ∥ CPU):** a generic streaming driver in `sciencekit_common/src/execution/` owns the prefetch pipeline over `SKLazySource`: a dedicated I/O thread reads batch k+1 into a double buffer while rayon computes batch k via an algorithm-supplied update closure `update(SKDataBatch, &mut state) -> Continue | Stop`; fallible iteration errors and the final-batch marker follow the `streaming-batches` contract. Algorithms stop reimplementing (or never start) bespoke prefetch loops.
- **Buffer-safety check at resolution:** streaming plans verify `2 × batch_bytes ≤ available_memory_bytes` (double buffering needs two resident batches) and surface a structured error when the hint violates it.
- **BREAKING (crates only, pre-release):** kernel signatures gain the `parallelism` parameter and `SKFloat` bounds tighten to `Send + Sync` on the closure parameters; `sk_resolve_execution_plan` error taxonomy gains the buffer-oversize error.

## Capabilities

### New Capabilities
- `streaming-executor`: generic prefetch pipeline over `SKLazySource` — owned-batch handoff to the compute pool, per-batch update closure with early stop, structured error propagation, final-batch delivery, and plan-driven buffer depth.

### Modified Capabilities
- `execution-planning`: resolution now derives `parallelism` from mode, work-unit size (dataset or batch), cores, and documented per-kernel grain — not unconditionally `cpu_cores`; adds the double-buffer memory check and its structured error.
- `higher-order-kernels`: kernels accept the resolved `parallelism` and switch sequential/parallel accordingly; axis-0 reduction is parallelized without shared-accumulator races; grain values are documented with measurement provenance.

## Impact

- **Code:** `crates/sciencekit_common/src/execution/` (resolve rule, new buffer check, new `streaming_driver.rs`); `crates/sciencekit_math/src/kernels/` (all four kernels + companion tests); benches under `crates/sciencekit_math/benches/` for calibration.
- **APIs:** kernel free functions gain a `parallelism: usize` parameter (**BREAKING** within the workspace, pre-release); no public API removed.
- **Dependencies:** `criterion` (dev-dependency) for calibration; no new runtime dependencies.
- **Acceptance (PRD §8.7):** correctness identical for small and large data (parallel ≡ sequential within float tolerance), behavior stable under concurrency (rayon pool shared), sequential fallback proven for `parallelism = 1`, calibration artifacts under `temporary/` with results recorded beside the grain constants.
