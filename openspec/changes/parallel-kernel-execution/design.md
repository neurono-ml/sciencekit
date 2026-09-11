## Context

The planning architecture already resolves execution intents into plans (`sciencekit_common/src/execution/`), but two wires are loose:

1. `sk_resolve_execution_plan` returns `parallelism = cpu_cores` unconditionally; dataset and batch sizes in `SKExecutionContext` only influence the mode, never the parallelism decision.
2. Of the four dense kernels in `sciencekit_math/src/kernels/`, only `sk_scale_in_place` uses `par_azip!` — unconditionally, even for tiny arrays. The other three are strictly sequential. Nothing in the kernel layer reads `SKExecutionPlan`.

The backend layer is the in-repo precedent: `SKMathBackend::gemm(a, b, alpha, parallelism)` honors the plan via `Par::rayon(parallelism)` vs `Par::Seq` in the faer backend (`backend/faer_backend.rs`).

Streaming batch contracts already exist (`SKLazySource`, `SKDataBatch` — owned, `Send`, sequence-positioned, final-flagged) and the spec `streaming-batches` explicitly names the I/O ∥ CPU overlap as their purpose; no generic driver consumes that contract yet.

Out of the exploration discussions that shaped this change: decide the parallel question at resolution time (never via constants scattered through kernels), prefer full compute parallelism in streaming (I/O and compute use disjoint pools, so prefetch makes cores available rather than contesting them), and reuse one measured grain per kernel across regimes.

## Goals / Non-Goals

**Goals:**

- The `Automatic` execution mode becomes the single place where parallel-vs-sequential is decided — driven by mode, work-unit size (dataset or batch), cores and documented grain — testable purely by context injection.
- All four dense kernels honor `parallelism: usize` (1 = sequential plain `azip!`; > 1 = `par_azip!`), following the `gemm` precedent, with results identical to sequential within float tolerance.
- A generic streaming driver in `sciencekit_common/src/execution/` owns the prefetch pipeline (double buffer, owned-batch handoff, fallible errors, final-batch delivery, early stop), so algorithms only supply `update(batch, state) -> Continue | Stop`.
- Grain constants carry measurement provenance (protocol, machine, date) from a criterion calibration protocol; no unexplained magic numbers anywhere.
- Sequential fallback is provable: `parallelism = 1` exercises the plain sequential path.

**Non-Goals:**

- Adaptive/downscaling parallelism at runtime (re-resolution via injected measured context is a future extension; the pure-resolve contract already permits it).
- Compute-depth pipelining beyond one for one-pass streaming algorithms (double-buffered I/O only; depth > 1 is future work, motivated by benches).
- Tokio orchestration for disk sources (`SKLazySource` is synchronous; the driver starts with a dedicated plain thread — Tokio belongs to `InProcessAsynchronous` network sources, a separate change).
- GPU/SIMD backends, allocator changes, Python bindings (separate change groups per plan rules).
- Changing any existing resolver semantics for mode selection (only the `parallelism` derivation and the new buffer guard are touched).

## Decisions

### Decision 1: Parallelism is resolved, not decided in the kernel

Kernels take `parallelism: usize` and do nothing but branch `1 → azip!` / `> 1 → par_azip!`. All policy lives in `sk_resolve_execution_plan`.

*Why:* keeps resolution pure, deterministic and injectable (existing spec contract); mirrors the `gemm` precedent; avoids duplicating thresholds; single-core machines degrade for free because `parallelism` collapses to 1.

*Alternatives considered:* unconditional `par_azip!` everywhere (regression on small data, ignores plans); per-kernel internal thresholds (duplicates policy, untestable without the real machine); separate `_par` function variants (duplicated API surface, decision pushed to every caller).

### Decision 2: The crossover rule is `min(cores, ceil(unit / grain))`, with grain measured per kernel

The threshold is not a global constant but a per-kernel documented grain (minimum elements per thread to amortize dispatch overhead), applied to the right work unit: whole dataset (in-memory), one batch (streaming), any shard (memory-mapped → always `cores`).

*Why:* the crossover ratio overhead/cost-per-element varies by machine, closure cost and data locality — a fixed `N` cannot be correct. The grain expresses the same physics honestly: a number with semantics (elements per thread), origin (benches) and provenance (machine + date). Streaming reuses the identical grain: dispatch overhead and per-element cost do not change with data provenance; cold-cache effects shift throughput, not the crossover.

*Alternatives considered:* global constant threshold (wrong for at least one of machine, algorithm, data); closure-cost hints plumbed into resolution (possible future extension following the `batch_size_hint` pattern, deferred until benches show the conservative default is insufficient).

### Decision 3: Conservative defaults bias upward in streaming, downward in memory

In-memory with unknown/costly traits: a conservative (large) grain errs on the safe side because expensive closures parallelize "late but well". Streaming with no batch hint: `parallelism = cores` — excess compute parallelism is nearly free while the pipeline is I/O-bound (cores idle between batches), but starvation when compute-bound loses the whole speedup. The asymmetric costs favor these directions; the double-buffer memory guard (`2 × batch ≤ available`) protects the one real streaming risk.

*Alternatives considered:* `parallelism = 1` while streaming (rejected: discards compute-bound speedup and implicitly assumes disk-bound, unknowable at resolution); adaptive measured downscaling now (breaks the pure-resolution contract as first version).

### Decision 4: One generic streaming driver, not per-algorithm pipelines

`sciencekit_common/src/execution/` gains a streaming driver consuming `SKLazySource` through a dedicated I/O thread into a bounded double buffer, handing owned `SKDataBatch`es to an algorithm callback `update(SKDataBatch, &mut state) -> Continue | Stop`, computing on the rayon pool per the plan.

*Why:* the batch contract was designed exactly for this handoff ("block survives the source"); concurrency subtleties (ordering, error propagation, final-batch handling) get written and tested once; algorithms gain streaming by writing only their update step. The callback shape supports early stopping (convergence) and later non-mutating consumers. Algorithms needing exotic control (resampling, multi-pass per batch) may drive the source directly — the driver is the default path, not a prison.

*Alternatives considered:* per-algorithm prefetch loops (N copies of subtle concurrency code); Tokio-based pipeline for disk sources (unnecessary dependency for synchronous iteration; reserved for the asynchronous mode).

### Decision 5: Axis-0 reduction parallelizes rows through partial accumulation

Per-thread/thread-chunk column accumulators computed independently, then combined into the output — never concurrent writes to a shared accumulator. Axis-1 parallelizes trivially per row (independent outputs).

*Alternatives considered:* atomics/shared accumulator (contention and different summation order per run); keeping axis-0 sequential (asymmetric loss: it is the common reduction shape for column statistics).

### Decision 6: Kernel signature change is accepted (pre-release breaking)

All four kernels gain a trailing `parallelism: usize`; closure parameters tighten to `Sync` (+ `Send` where moved). Internal callers update in the same change; the umbrella crate has no public re-exports yet, so blast radius is workspace-internal.

*Alternatives considered:* preserving signatures via mode-free duplicate functions («`_par` variants») — rejected in Decision 1.

## Risks / Trade-offs

- [Bench-calibrated grain can still be wrong for a closure far outside the calibration trio (cheap/medium/expensive)] → Error is bounded: conservative grain only delays parallelization of heavy closures, which still net-gains; provenance makes re-calibration mechanical. Follow-up calibration with domain-representative closures is cheap.
- [Double-buffered streaming doubles resident batch memory] → Resolution-time guard `2 × batch ≤ available_memory_bytes` fails structurally before any read; batch hint is user-tunable.
- [Parallel float summation reorders additions (axis-0 partials, axis-1 row folds)] → Contract is tolerance-based equality (the repo's `assert_close` pattern), not bit-identity; documented per kernel.
- [Driver concurrency bugs (ordering, lost wakeups, error swallowing)] → Single implementation covered by TDD scenarios mirroring the spec (read log concurrency observation, structured error propagation, exactly-once final batch); no algorithm-side bespoke loops to hide regression.
- [Oversubscription under nested parallelism (driver inside an external `par_iter`)] → The plan parameter allows callers to pass reduced parallelism; a future work-estimate hint can automate this. Accepted trade-off for now.
- [Grain constants measured on one machine may not hold across architectures] → Provenance + re-calibration protocol; the contract (docs + tolerance scenarios) is independent of the constant.

## Migration Plan

1. Land the resolution rule + buffer guard (pure functions; tests inject simulated contexts — zero machine dependence).
2. Land kernel signatures + sequential/parallel paths + race-free axis-0 (TDD: red on tolerance scenarios, green on minimal branch logic).
3. Run the calibration protocol (criterion benches under `temporary/YYYY-MM-DD/parallel-kernel-execution/`), record grain constants with provenance beside kernels.
4. Land the streaming driver against the batch contract with its own TDD suite.
5. Rollback: commits are ordered and independently revertible; kernels revert to sequential-only by passing `1` everywhere (no API removal), the driver is additive. No data migration, no consumer migration in-workspace.
6. Docs: mdBook chapter updates (math-kernel / architecture pages) ride along per the documentation rule; changelog entry on merge.

## Open Questions

None blocking. Deferred-with-default-direction items (runtime-adaptive re-resolution; compute depth > 1 for one-pass streams; cost hints in the context) are recorded in Non-Goals with their default paths and motivated by the same benches that will exist after this change's calibration.
