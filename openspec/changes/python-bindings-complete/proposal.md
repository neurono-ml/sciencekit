# Proposal

## Why

The PRD's final deliverable is a complete Python binding so the library can be used from
the ecosystem it reimplements. The Rust API is stable enough after Waves 0–6; PyO3 0.30 and
numpy 0.30 both declare MSRV 1.85, matching the workspace. This change ships the Python
package, zero-copy NumPy input and bindings for every estimator, pipeline and metric.

## What Changes

- New `sciencekit_python` crate using PyO3 0.30 and numpy 0.30.
- A Python package `sciencekit` with submodules by family: preprocessing, impute,
  linear_model, neighbors, svm, tree, ensemble, cluster, decomposition, outlier, metrics,
  model_selection and pipeline.
- Builder/model wrappers for every estimator: chainable setters, `fit`, `predict`,
  `predict_proba`, `transform`, `score`, `fit_predict` and accessors.
- Zero-copy NumPy input (`PyReadonlyArray2` → `ArrayView2`) and NumPy outputs; dtype
  preservation for f32/f64.
- Error mapping from `SKError` to Python exceptions (`ValueError`, `RuntimeError`,
  `NotFittedError`-style custom exceptions).
- GIL release around compute; thread-safe model objects.
- Pipelines, grid/halving search, metrics and model persistence hooks.
- Packaging: `pyproject.toml`, `maturin`, type stubs (`.pyi`), README and pytest suite.
- Explicit non-goals: Python-side training loops, GPU, async and native ONNX conversion.
- Companion tests: import, dtype, zero-copy, every estimator family, pipelines, metrics,
  errors, concurrency and wheel build.

## Capabilities

### New Capabilities

- `python-bindings`: the Python module layout, numpy zero-copy boundary, estimator/pipeline
  wrappers, error mapping, GIL behavior, packaging and tests.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_python` workspace member and a Python package.
- **APIs:** additive Python API; Rust crates unchanged.
- **Dependencies:** `pyo3 0.30`, `numpy 0.30`, `maturin` for packaging.
- **Downstream:** exposes the full library to Python users and completes the roadmap's
  Python deliverable.
- **Acceptance criteria:** every family imports, fits and predicts on NumPy arrays; f32/f64
  preserved; zero-copy verified; errors mapped; wheel builds.
