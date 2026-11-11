# Tasks

> **External repository note:** In accordance with ADR #285 and ADR #311, the Python
> binding lives in its own standalone repository,
> [`neurono-ml/sciencekit-py`](https://github.com/neurono-ml/sciencekit-py), built with
> PyO3 `0.30` and `maturin` over the shared `sciencekit_ffi` C ABI (`branch = "main"` git
> dependency), with English mdBook documentation published at
> <https://neurono-ml.github.io/sciencekit-py/>. No Python code or packaging metadata is
> added to the main `neurono-ml/sciencekit` Rust workspace.

## 1. Crate, module and boundary (`neurono-ml/sciencekit-py`)

- [x] 1.1 Create standalone `sciencekit_python` crate and `sciencekit` mixed Rust/Python
  package in `neurono-ml/sciencekit-py` (`Cargo.toml`, `pyproject.toml`, `src/lib.rs`,
  `python/sciencekit/`) depending on `sciencekit_ffi` via git dependency; verify
  `cargo check`, `cargo clippy --locked --all-targets -- -D warnings` and `cargo test --locked`.
- [x] 1.2 Implement the NumPy boundary (`DenseDescriptor` zero-copy borrow over C-contiguous
  `f32`/`f64` arrays with explicit `copy=True` fallback, plus `CsrDescriptor` sparse
  validation and explicit `.to_dense()`) and owned output conversion in
  `neurono-ml/sciencekit-py`.
- [x] 1.3 Implement structured error mapping (`ScienceKitError` and `InvalidHandleError`,
  `ShapeError`, `InvalidArgumentError`, `UnsupportedError`, `NotFittedError`, `IoError`,
  `InternalError` carrying `.status_code`, `.status_name`, and `.message`) in
  `neurono-ml/sciencekit-py`; verify with `pytest`.

## 2. Family wrappers (`sciencekit_ffi` ABI 1.0 surface in `neurono-ml/sciencekit-py`)

- [x] 2.1 Add `sciencekit.preprocessing.StandardScaler` / `StandardScalerModel` wrappers
  (chainable `with_mean`, `with_std`, `backend`, `fit`, `transform`, `fit_transform`,
  `f64`/`f32`) and `pytest` coverage in `neurono-ml/sciencekit-py`.
- [x] 2.2 Add `sciencekit.linear_model.LinearRegression` / `LinearRegressionModel` and
  model persistence (`save` / `load`) wrappers (`include_intercept`, `backend`, `fit`,
  `predict`, `score`, `f64`/`f32`) and `pytest` coverage in `neurono-ml/sciencekit-py`.
- [ ] 2.3 Add neighbours/SVM wrappers and `pytest` coverage in `neurono-ml/sciencekit-py`
  once `sciencekit_ffi` exposes those estimator families.
- [ ] 2.4 Add tree/ensemble wrappers and `pytest` coverage in `neurono-ml/sciencekit-py`
  once `sciencekit_ffi` exposes those estimator families.
- [ ] 2.5 Add cluster/decomposition/outlier wrappers and `pytest` coverage in
  `neurono-ml/sciencekit-py` once `sciencekit_ffi` exposes those estimator families.
- [ ] 2.6 Add pipeline, FeatureUnion, grid/halving search and standalone metrics wrappers
  and `pytest` coverage in `neurono-ml/sciencekit-py` once `sciencekit_ffi` exposes them.

## 3. GIL and zero-copy (`neurono-ml/sciencekit-py`)

- [x] 3.1 Wrap native calls in `py.detach` with `Mutex`/`RwLock` handle synchronization,
  idempotent `close()`, `Drop` finalizer release, leak-free failure loops and multi-threaded
  concurrency tests in `neurono-ml/sciencekit-py`.
- [x] 3.2 Add explicit zero-copy tests (`DenseDescriptor.data_pointer` identity and
  `copied is False`), `copy=True` fallback tests for Fortran/strided views, and CSR
  validation tests in `neurono-ml/sciencekit-py`.

## 4. Packaging, stubs and acceptance (`neurono-ml/sciencekit-py`)

- [x] 4.1 Add `pyproject.toml`, `maturin` configuration, `.github/workflows/ci.yml` and
  `maturin build --release` wheel install smoke verification in `neurono-ml/sciencekit-py`.
- [x] 4.2 Write PEP 561 type stubs (`__init__.pyi`, `linear_model.pyi`, `preprocessing.pyi`,
  `py.typed`) and `mypy --strict` + `__all__` parity tests in `neurono-ml/sciencekit-py`.
- [x] 4.3 Add `pytest` suite covering import, ABI metadata, `LinearRegression`,
  `StandardScaler`, `f32`/`f64`, persistence round-trip, dense & CSR descriptors,
  structured status-code errors, close idempotency, repeated-failure leak checks, large
  arrays and multi-threaded concurrency in `neurono-ml/sciencekit-py`.
- [x] 4.4 Run `cargo fmt --all --check`, `cargo clippy --locked --all-targets -- -D warnings`,
  `cargo test --locked`, `maturin build --release` and `pytest -q` in
  `neurono-ml/sciencekit-py`; fix every failure.
- [x] 4.5 Publish the English mdBook documentation site (`docs/documentations` branch,
  `sitemap.xml` postbuild and GitHub Pages workflow) at
  `https://neurono-ml.github.io/sciencekit-py/`; verify with `mdbook build docs`.
- [x] 4.6 Independent review by a separate agent validating the `neurono-ml/sciencekit-py`
  implementation and documentation; address findings and re-run gates.

## Workflow follow-up

- Created ADR issue #311 in `neurono-ml/sciencekit` and referenced it from PR #251 with
  `Closes #311`.
- Implementation and documentation live in `neurono-ml/sciencekit-py` (`main` and
  `docs/documentations`).
