# Design

## Context

See `proposal.md` — Why. `ffi-core` provides the stable C ABI
(`crates/sciencekit_ffi/include/sciencekit.h`); `interop-foundation` provides the
model-state contract behind `save`/`load`. R has no native FFI story for Rust: an R
package that exposes native routines must compile them itself (a `.c`/`.cpp` file in
`src/`), register them with `R_registerRoutines`, and call them with `.Call`.

The R package lives in its own repository (`neurono-ml/sciencekit-r`), outside this Cargo
workspace, following the same pattern as `neurono-ml/sciencekit-go` and
`neurono-ml/sciencekit-lua`.

## Goals / Non-Goals

**Goals:**
- A small, idiomatic R surface: builders and models behave like ordinary R objects.
- Safe handle lifecycle through a finalizer plus an explicit `close()`.
- Structured, inspectable errors carrying the ABI status name and code.
- Reproducible CI: `R CMD check .` on a clean machine.

**Non-Goals:**
- R-native estimators, sparse/mixed input, async, GPU-specific APIs.
- `f32` semantics; the R surface is `f64`-only.
- Zero-copy matrices; the boundary copies.

## Decisions

### Decision 1 — Repository and dependency shape

The package lives in `neurono-ml/sciencekit-r` as a standalone R package (no Cargo
workspace). Because `sciencekit_ffi` is not published on crates.io yet, the package
**vendors the ABI header** and links the **shared library built from a checkout of the main
repository**:

- `scripts/sync-header.sh` copies `crates/sciencekit_ffi/include/sciencekit.h` from a
  checkout of `neurono-ml/sciencekit` into `src/`, so the compiled-against ABI never drifts
  from the targeted ABI.
- `scripts/build-native-library.sh` builds `cargo build -p sciencekit_ffi --release` in
  that checkout and installs the shared object where the package's linker flags find it.

This mirrors the Go binding's `lib/` + vendored-header pattern and keeps the R package
free of a Cargo dependency it cannot express. A crates.io release of `sciencekit_ffi`
would later let the build script fetch a prebuilt artifact instead; the seam is the same.

### Decision 2 — Native routine registration

The package ships a C shim (`src/sciencekitr.c`) that declares the `sciencekit_ffi` entry
points it needs and exposes a small, stable set of `.Call` routines to R. Registration
happens in `src/init.c`:

```c
static const R_CallMethodDef call_methods[] = {
    {"C_sciencekit_version", (DL_FUNC) &c_sciencekit_version, 0},
    ...
    {NULL, NULL, 0}
};

void R_init_sciencekitr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
```

`R_useDynamicSymbols(dll, FALSE)` closes the door on `.Call("symbol_name", ...)` lookups,
so the only entry points into the native code are the registered ones. The NAMESPACE
declares `useDynLib(sciencekitr, .registration = TRUE)`.

### Decision 3 — External pointers, finalizers and `close()`

Builders and fitted models are R external pointers (`R_MakeExternalPtr`) whose protected
payload points at a small C struct holding the native handle and the kind of handle it is.
Two release paths share one idempotent routine:

- `close()` — R-level, explicit, eager; sets the payload's handle to `NULL`.
- The finalizer — installed with `R_SetExternalPtrProtected`, called by the GC when the
  pointer becomes unreachable.

Both call the same C routine, which checks the handle for `NULL` before freeing and
clears it afterwards. Calling `close()` twice is a no-op, not an error; calling an
operation on a closed handle raises a `sciencekit_error` with status
`invalid_handle` instead of dereferencing freed memory. The finalizer therefore cannot
double-free, and `close()` cannot leak.

### Decision 4 — Dense matrix conversion and `f64`-only semantics

R matrices are converted to the ABI's `SKFfiArray` descriptor at the boundary:

- The object must be a numeric matrix (or a data frame / vector coerced to one) with
  `double` storage. Integer and logical matrices are rejected with a clear message; they
  are **not** silently coerced, because the binding is `f64`-only.
- The matrix must be rectangular: R's matrix type guarantees this, but a data frame with
  ragged columns must be rejected before the call.
- The conversion copies the values into a C buffer (`double`), which is passed to the ABI
  for the duration of the call only. The R object stays alive for the whole call because
  the `.Call` argument list holds a reference to it.

Outputs (`predict`, `transform`) come back as R matrices built from the ABI-owned buffer;
the ABI's output descriptor is released with `sk_ffi_array_free` exactly once.

### Decision 5 — Operations

The R surface mirrors the ABI's families:

- `sk_builder(family, ...)` — a builder for `linear_regression` or `standard_scaler`,
  with setters for the intercept, the scaler's centering/scaling and the math backend.
- `fit(builder, x, y)` — returns a fitted model object.
- `predict(model, x)`, `transform(model, x)`, `score(model, x, y)`.
- `save_model(model, path)` and `load_model(path)`.

The fitted model is an R object of class `sciencekit_model` holding the external pointer,
so it prints informatively and can carry attributes.

### Decision 6 — Structured errors

The C shim never raises R errors itself for ABI failures; it returns the status code and
the thread-local message. The R layer converts a failure into a condition:

```r
structure(
  class = c("sciencekit_error", "error", "condition"),
  list(message = ..., status = "shape", status_code = 2L, operation = "fit")
)
```

The status name comes from `sk_ffi_status_name` and the code is the numeric ABI status.
R users can therefore branch on the status (`conditionCall`/`status`) rather than parsing
a message string.

### Decision 7 — Packaging and CI

`DESCRIPTION` declares the package with `Suggests: testthat`; `NAMESPACE` exports the
documented surface and the `useDynLib` directive. The test suite lives in
`tests/testthat/` and runs under `testthat`. CI (`.github/workflows/R-CMD-check.yaml`)
installs R, builds the native library and runs `R CMD build .` followed by `R CMD check .`.

## Risks / Trade-offs

- **Matrix copies** → documented; the ABI's descriptor model permits a zero-copy path later.
- **`f64`-only** → R's `double` is the natural numeric type; `f32` is rejected explicitly.
- **Native library not on crates.io** → the build script builds it from a checkout; CI
  clones the main repository first.
- **GC finalizer ordering** → `close()` plus an idempotent release makes ordering
  irrelevant.

## Migration Plan

Additive: a new package in a new repository. No existing API changes.

## Open Questions

None blocking. A data-frame/Arrow zero-copy path and `f32` support are future work.
