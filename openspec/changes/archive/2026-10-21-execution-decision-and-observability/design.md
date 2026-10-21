# Design — execution-decision-and-observability

## Context

Phase 0.1–0.3 delivered `sciencekit_common` (contract vocabulary, including the
`execution-planning` types `SKExecutionContext`/`SKExecutionMode`/`SKExecutionPlan` and the
pure `sk_resolve_execution_plan`) and `sciencekit_math` (the `SKMathBackend` abstraction with
`sk_default_math_backend`). Nothing yet composes these into the automatic decision that a
Phase-1 estimator needs, and there is no shared builder, observability or allocator layer.
See `proposal.md` — Why for motivation; the requirements live in the four capability specs.

## Goals / Non-Goals

**Goals:**
- A deterministic automatic execution resolver that composes the `execution-planning`
  contracts with backend selection.
- A reusable builder foundation so every estimator exposes `execution_mode(...)` (default
  `Automatic`) and `build() -> Result`.
- `tracing` observability on public operations with an opt-in OTel export.
- `allocator-jemalloc`/`allocator-mimalloc` feature flags with a clean default.

**Non-Goals:**
- Any algorithm (Wave 1+). No GPU backends, no Python bindings.
- No change to the existing `execution-planning` or `blas-interface` specs — this wave adds
  new behavior on top of them.

## Decisions

### D1. Backend-agnostic resolver in `sciencekit_common`; backend routing stays in `sciencekit_math`
`sciencekit_common` cannot depend on `sciencekit_math` (layering). So the automatic decision
stays split: the deterministic `sk_resolve_execution_plan` (already in common) resolves the
plan; `sk_default_math_backend` (already in math) selects the backend. W0.4 does **not** merge
them into one cross-crate function; instead the shared builder machinery composes both at the
algorithm-crate level.
- *Alternatives considered:* a new `sciencekit_execution` crate pulling both — rejected as
  over-engineering before the first algorithm exists; a common→math dependency — rejected,
  breaks layering.

### D2. A `SKBuilder` trait + `SKBuilderState` in `sciencekit_common`
Define `trait SKBuilder<Model>` with `execution_mode(SKExecutionMode) -> &mut Self` and
`fn build(self) -> Result<Model, SKError>`. Provide a small `SKBuilderState` carrying the
execution intent (default `Automatic`) plus shared validation helpers, so algorithm builders
compose it instead of re-implementing intent storage.
- *Alternatives considered:* a derive macro for builders — rejected for now (mature only once
  several algorithms exist); a hand-rolled builder per algorithm with no shared state —
  rejected, duplicates the execution-intent plumbing W0.4 exists to remove.

### D3. Observability via `tracing`, OTel gated behind `observability-export`
Public operations emit `trace_span!`/`#[instrument]` spans with structured fields (shape,
mode, backend, duration); errors are recorded on the span. The `observability-export` feature
pulls `tracing-opentelemetry`/`opentelemetry`; without it no exporter is initialized and no
OTLP traffic occurs.
- *Alternatives considered:* manual logging only — rejected, spans give context across async
  boundaries; always-on OTel — rejected, violates the default-no-OTLP requirement.

### D4. Allocator features live in `sciencekit_common`
`allocator-jemalloc` enables `tikv-jemallocator`, `allocator-mimalloc` enables `mimalloc`;
each installs `#[global_allocator]` behind its feature. `compile_error!` fires when both are
enabled (a single global allocator can be installed at most once). Default build uses the
system allocator.
- *Alternatives considered:* a dedicated `sciencekit_allocator` crate — rejected, premature;
features on `common` keep the surface small.

## Risks / Trade-offs

- [Global allocator in a library crate] → Feature-gated `#[global_allocator]` is a standard
  pattern; document that consumers needing a specific allocator enable exactly one feature,
  and the `compile_error!` guard catches the both-enabled mistake at build time.
- [Observability overhead on hot paths] → Spans are enabled/disabled by the subscriber; the
  default build emits nothing and the cost is limited to the `tracing` disabled path.
- [Builder trait genericity vs. ergonomics] → `build(self) -> Result<Model, _>` trades a
  little ergonomics for uniform failure handling; revisited once the first estimator lands.

## Migration Plan

Pure additive change to `sciencekit_common`/`sciencekit_math` (new traits, features, spans).
No existing behavior changes; no rollback risk beyond reverting the feature flags and
builder traits if an algorithm PR exposes a design flaw.

## Open Questions

None that would change the specs, approach or task breakdown. (The first Wave-1 estimator may
refine the exact `SKBuilder` signature; that is an additive, backward-compatible evolution.)