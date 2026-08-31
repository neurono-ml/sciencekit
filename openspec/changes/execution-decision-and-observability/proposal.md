## Why

Phases 0.1–0.3 froze the contract vocabulary (`sciencekit_common`) and the numeric substrate
(`sciencekit_math`), but nothing yet chooses *how* an operation actually runs. Wave 0.4
closes Phase 0 by adding the **automatic execution decision** (route each operation to a
concrete execution mode and a concrete math backend), the **mandatory builder foundation**
every public estimator exposes, **observability** (`tracing` + OpenTelemetry) and the
**custom-allocator feature flags** (`allocator-jemalloc`/`allocator-mimalloc`). Without this
wave, no Phase-1 algorithm can be built: builders, the automatic mode, and backend selection
are prerequisites for every estimator (PRD §5.3, §9).

## What Changes

- **Automatic execution decision mechanism**: a pure resolver that turns an
  `SKExecutionMode::Automatic` intent (the default) plus an `SKExecutionContext` into a
  concrete `SKExecutionPlan`, and routes heavy dense algebra to the active
  `SKMathBackend` via `sk_default_math_backend`. Built on the existing
  `execution-planning` contracts in `sciencekit_common`.
- **Base builders**: the mandatory builder pattern — every public estimator/transformer
  exposes a builder with `execution_mode(...)` defaulting to `Automatic` and a `build()`
  returning `Result`. Provides the shared builder machinery (execution intent accumulation,
  validation) once, so algorithm crates reuse it.
- **Observability**: operations emit `tracing` spans and structured logs; an opt-in
  OpenTelemetry exporter (`tracing-opentelemetry`) is wired behind a feature flag.
- **Custom allocators**: `allocator-jemalloc` and `allocator-mimalloc` feature flags swap
  the global allocator (`tikv-jemallocator` / `mimalloc`); the default build keeps the
  system allocator.
- New workspace dependencies: `tracing`, `tracing-opentelemetry 0.33`, `opentelemetry`,
  `tikv-jemallocator 0.5`, `mimalloc 0.1`.

Out of scope: any algorithm (Wave 1+), GPU backends, Python bindings.

## Capabilities

### New Capabilities

- `execution-decision`: the automatic execution decision mechanism — given an
  `SKExecutionMode` (default `Automatic`) and an `SKExecutionContext`, resolve a concrete
  `SKExecutionPlan` and select the active `SKMathBackend` for heavy dense algebra.
- `base-builders`: the mandatory builder pattern — every public estimator/transformer
  exposes a builder with `execution_mode(...)` defaulting to `Automatic` and a `build()`
  returning `Result`.
- `observability`: `tracing` spans and structured logs for library operations, with an
  opt-in OpenTelemetry export path.
- `allocator-selection`: `allocator-jemalloc`/`allocator-mimalloc` feature flags that select
  the global allocator, defaulting to the system allocator.

### Modified Capabilities

None — Phase 0.4 introduces new behavior; existing capability specs are unchanged.

## Impact

- **Code:** `sciencekit_common` gains the shared builder machinery and the automatic
  execution resolver; `sciencekit_math` exposes the backend-selection hook
  (`sk_default_math_backend`) already present in W0.3. New `sciencekit_common` feature flags
  for allocators.
- **Dependencies:** `tracing`, `tracing-opentelemetry 0.33`, `opentelemetry`,
  `tikv-jemallocator 0.5`, `mimalloc 0.1`.
- **Downstream:** every Wave-1+ estimator consumes the base builders and the automatic
  execution decision, so its PR only adds algorithm-specific hyperparameters and logic.
- **Acceptance (PRD §8.7):** builders resolve automatic mode with little and large data,
  stay correct under concurrency, and the resolver is pure and deterministic; spans are
  emitted for representative operations; allocator features build green.