# Design

## Context

See `proposal.md` — Why. The Rust API is stable; PyO3 0.30 and numpy 0.30 both declare
MSRV 1.85. The binding crate depends on every model crate and exposes a Python package. The
hard parts are zero-copy NumPy input, the large wrapper surface and packaging.

## Goals / Non-Goals

**Goals:**
- Zero-copy input and dtype-preserving output.
- Complete family coverage with a small, macro-driven wrapper layer.
- GIL release and thread-safe models.
- A wheel that installs and a pytest suite that exercises every family.

**Non-Goals:**
- Python-side training loops or GPU.
- Native ONNX conversion.
- Reimplementing the Rust logic in Python.

## Decisions

### Decision 1 — `sciencekit_python` crate and `sciencekit` package

The crate is a workspace member built with maturin. The Python package exposes family
submodules and a top-level namespace. Rust types are wrapped in `#[pyclass]` structs; the
Python names mirror the Rust builder/model names (for example
`StandardScaler`/`StandardScalerModel`).

### Decision 2 — NumPy boundary

Input: `PyReadonlyArray2<F>` → `ArrayView2<F>` without copying. The wrapper rejects
non-contiguous or unsupported layouts with a clear error (or copies explicitly when the
user opts in). Output: `PyArray2<F>`/`PyArray1<F>` allocated from Rust results. f32/f64
are supported through generic macros; other dtypes raise.

### Decision 3 — Wrapper macros

A macro generates the builder wrapper, setter methods and model wrapper for a family,
reducing boilerplate. Family modules add only the operations that exist on the Rust type
(`fit`, `predict`, `predict_proba`, `transform`, `fit_predict`, `score`, accessors).

### Decision 4 — Error mapping

`SKError` maps to Python exceptions: shape/hyperparameter/conversion → `ValueError`;
I/O/runtime → `RuntimeError`; not-fitted → a custom `NotFittedError`; unsupported operation
→ `NotImplementedError`. The original message is preserved.

### Decision 5 — GIL and concurrency

Compute-heavy calls use `py.allow_threads` around the Rust operation. Fitted model objects
are immutable and `Send + Sync`; the wrapper holds them behind the GIL only for conversion.
NumPy buffers are borrowed only for the duration of the call.

### Decision 6 — Pipelines, model selection and metrics

Pipeline/FeatureUnion/grid/halving wrappers compose Python objects that hold Rust
configurations. Metrics are exposed as functions accepting NumPy arrays. Model persistence
delegates to the Rust interop formats.

### Decision 7 — Packaging

`pyproject.toml` uses maturin; type stubs describe every public class/function; pytest
tests cover import, dtype, zero-copy, each family, pipelines, metrics, errors and
concurrency. CI builds wheels for Linux/macOS/Windows (Python versions pinned).

### Decision 8 — File layout

```
crates/sciencekit_python/
  Cargo.toml
  pyproject.toml
  src/
    lib.rs
    numpy_boundary.rs
    errors.rs
    macros.rs
    preprocessing.rs, impute.rs, linear_model.rs, neighbors.rs, svm.rs,
    tree.rs, ensemble.rs, cluster.rs, decomposition.rs, outlier.rs,
    metrics.rs, model_selection.rs, pipeline.rs, interop.rs
  python/sciencekit/*.pyi
  tests/*.py
```

Family files are thin wrappers; shared conversion/error code lives in the support modules.

## Risks / Trade-offs

- **Large wrapper surface** → macros and a coverage checklist per family.
- **Zero-copy is only possible for compatible layouts** → explicit copy fallback or error.
- **GIL release with borrowed NumPy buffers** → borrow for the call duration and release
  around pure Rust compute.
- **Wheel matrix** → maturin CI and pinned Python versions.
- **Type stubs drift** → a stub test and a generated coverage check.
- **PyO3/numpy MSRV** → both declare 1.85; verify in the spike.

## Migration Plan

Additive: new crate and Python package. No existing Rust API changes.

## Open Questions

None blocking. Async and GPU bindings are future changes.
