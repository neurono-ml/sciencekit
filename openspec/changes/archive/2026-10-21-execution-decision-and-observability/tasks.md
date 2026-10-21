# Tasks — execution-decision-and-observability

Every task follows TDD (the `tdd` skill): test first in a companion `*_tests.rs` module,
confirmed failure, minimal implementation. No file beyond 200 lines — use folder modules.

## 1. Base builders

- [x] 1.1 TDD of the `SKBuilder` trait (`execution_mode(...)`, `build(self) -> Result<Model, SKError>`) in `sciencekit_common`
- [x] 1.2 TDD of `SKBuilderState` (execution intent storage defaulting to `Automatic`) + shared validation helper
- [x] 1.3 TDD of a reference estimator exposing only its builder (private direct constructor) and honouring the default/override mode

## 2. Execution decision

- [x] 2.1 TDD that the automatic intent resolves to a concrete plan (in-memory for small, streaming for oversized) — deterministic, pure
- [x] 2.2 TDD that an explicit incompatible intent fails with the precise `ExecutionModeIncompatible` error; automatic never conflicts
- [x] 2.3 TDD of the backend-routing hook: default build routes to `SKFaerBackend`, `blas-backend` build routes to `SKNdArrayLinalgBackend` (via `sk_default_math_backend`)
- [x] 2.4 TDD that concurrent resolutions are independent (identical inputs → identical plans)

## 3. Observability

- [x] 3.1 TDD that public operations emit `tracing` spans with structured fields (shape, mode, backend, duration)
- [x] 3.2 TDD that a failed operation records the error on its span
- [x] 3.3 Implement the `observability-export` feature (`tracing-opentelemetry` + `opentelemetry`), gated off by default; verify a default build initializes no exporter

## 4. Allocator selection

- [x] 4.1 TDD of the `allocator-jemalloc` feature installing `tikv-jemallocator` (default build uses system allocator)
- [x] 4.2 TDD of the `allocator-mimalloc` feature installing `mimalloc`
- [x] 4.3 Verify `compile_error!` when both allocator features are enabled together

## 5. Acceptance and review

- [x] 5.1 Run all local gates (fmt, strict clippy, tests, doctests) and confirm green
- [x] 5.2 Verify the change's adapted acceptance checklist: builder/resolver correct with large and small data, deterministic under concurrency, pure functions returning owned/borrowed output, companion coverage, complete nomenclature with correct prefixes, no file beyond 200 lines