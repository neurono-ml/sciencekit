# Proposal

## Why

R is the lingua franca of statistical modelling: the ecosystem around `ggplot2`, `dplyr`,
`tidymodels` and `caret` assumes an R-native surface — matrices, data frames, S3 objects
and conditions — not raw FFI calls. The library already exposes a stable C ABI
(`sciencekit_ffi`, spec `ffi-abi`), so R can consume the very same implementation the Go
and Lua bindings consume, without a second Rust binding layer.

The gap is the glue. R owns the lifetime of its objects, but the C ABI owns the lifetime of
its opaque builder and model handles. The binding must bridge the two without leaking,
without use-after-free, and without hiding the ABI's diagnostics. R also has no native
compiled-code registration story for a Rust library: the package must ship a small C shim,
register its routines with `R_registerRoutines`, and be callable through `.Call`.

## What Changes

- New R package `sciencekitr` in its own repository (`neurono-ml/sciencekit-r`), consuming
  the stable C ABI of `neurono-ml/sciencekit`.
- A C shim in `src/` that calls `sciencekit_ffi` and registers its routines with
  `R_registerRoutines`; R calls it through `useDynLib(sciencekitr, .registration = TRUE)`
  and `.Call`, never through `dyn.load` symbol lookup.
- Builders and fitted models as R external pointers with a finalizer plus an explicit,
  idempotent `close()`.
- Dense matrix conversion at the boundary with rectangularity and type validation;
  `f64`-only semantics.
- `fit`, `predict`, `transform`, `score`, `save` and `load` exposed on the R objects.
- Structured errors: R conditions of class `sciencekit_error` carrying the ABI status name,
  the numeric status code and the native message.
- A `testthat` suite and a CI workflow running `R CMD check .`.
- Explicit non-goals: R-native estimators, sparse input, async, GPU-specific APIs,
  `f32` semantics, zero-copy matrices.

## Capabilities

### New Capabilities

- `r-bindings`: module loading and native routine registration, external-pointer lifecycle,
  dense matrix conversion, operations, structured errors and packaging.

### Modified Capabilities

None.

## Impact

- **Code:** new R package in its own repository (`neurono-ml/sciencekit-r`): `DESCRIPTION`,
  `NAMESPACE`, `R/`, `src/`, `tests/testthat/`, plus a build script for the native library
  and a header-sync script.
- **APIs:** additive R package.
- **Dependencies:** the `sciencekit_ffi` shared library (vendored header + documented build
  step; the ABI crate is not on crates.io yet, so the native library is built from a
  checkout of the main repository).
- **Downstream:** exposes the library to R users.
- **Acceptance criteria:** the package loads, every supported family fits/predicts/scores
  on a known fixture, the scaler transforms, persistence round-trips, errors carry the
  status name and code, `close()` is idempotent, repeated failures leak nothing, and
  `R CMD check .` is green.

## References

- ADR issue: <https://github.com/neurono-ml/sciencekit/issues/289>.
