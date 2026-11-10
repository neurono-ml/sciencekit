# sciencekit-go

Go bindings for the [sciencekit](https://github.com/neurono-ml/sciencekit)
machine-learning library. The module is a thin, typed layer over the stable C
ABI (`sciencekit_ffi`, spec `ffi-abi`): opaque handles, dense array conversion,
structured errors and an explicit lifecycle, without hand-written `unsafe` in
application code.

```go
import sciencekit "github.com/neurono-ml/sciencekit-go"
```

The module lives in `go/` of the sciencekit repository and is not a member of
the Cargo workspace.

## Requirements

- Go 1.22 or newer with cgo enabled (the default on a host with a C compiler).
- A C toolchain (`cc`, `clang` or `gcc`).
- Rust 1.85 to build the native library from source.

## Build and link the native library

The binding loads `sciencekit_ffi` as a shared library. Build it from the
repository root:

```sh
cargo build -p sciencekit_ffi --release
```

That produces `target/release/libsciencekit_ffi.so` on Linux
(`libsciencekit_ffi.dylib` on macOS, `sciencekit_ffi.dll` on Windows). The cgo
directives in the package already include the header at
`crates/sciencekit_ffi/include/sciencekit.h` and search
`target/release` relative to the module, so a default in-tree build needs no
extra flags:

```sh
go test ./go/...    # from the repository root
```

When the library is built elsewhere — for example with `CARGO_TARGET_DIR` —
point cgo and the loader at it:

```sh
export CGO_CFLAGS="-I$PWD/crates/sciencekit_ffi/include"
export CGO_LDFLAGS="-L$CARGO_TARGET_DIR/release -lsciencekit_ffi"
export LD_LIBRARY_PATH="$CARGO_TARGET_DIR/release:$LD_LIBRARY_PATH"   # DYLD_LIBRARY_PATH on macOS
go test ./go/...
```

`scripts/build-go-library.sh` performs the build and prints the three
variables for the current shell. Build tags are not required; cgo is mandatory.

## Usage

```go
features, err := sciencekit.NewMatrix(4, 2, []float64{
    0, 0,
    1, 0,
    0, 1,
    2, 3,
})
if err != nil {
    log.Fatal(err)
}

builder, err := sciencekit.NewLinearRegressionBuilder[float64]()
if err != nil {
    log.Fatal(err)
}
defer builder.Close()

model, err := builder.Fit(features, []float64{1, 3, 4, 14})
if err != nil {
    log.Fatal(err)
}
defer model.Close()

predictions, err := model.Predict(features)
```

`Matrix[T]` is a row-major `[]T` plus `Rows` and `Columns`; `NewMatrix`
validates that `len(Data) == Rows*Columns`. `T` is `float32` or `float64`, and
the binding selects the matching ABI dtype. Predictions and transforms are
copied into fresh Go slices, and the native buffers are released immediately.

### Estimator families

| Family | Builder | Model operations |
| --- | --- | --- |
| Linear regression | `NewLinearRegressionBuilder[T]`, `IncludeIntercept`, `Backend` | `Fit`, `Predict`, `Score`, `Save` |
| Standard scaler | `NewStandardScalerBuilder[T]`, `WithMean`, `WithStd`, `Backend` | `Fit`, `Transform`, `Save` |

`Backend` accepts `BackendFaer` (default), `BackendMatrixMultiply` and
`BackendNdArrayLinalg`.

## Errors

Every failure is a `*sciencekit.ScienceKitError` carrying the ABI `Code` (a
`Status`), the `Operation` and the native `Message`:

```go
var failure *sciencekit.ScienceKitError
if errors.As(err, &failure) {
    fmt.Println(failure.Code.Name(), failure.Message)
}
if sciencekit.IsStatus(err, sciencekit.StatusShape) {
    // recover from a shape mismatch
}
```

Calls on a closed handle return `StatusInvalidHandle`. `StatusUnknown` marks
the pointer-returning entry points (`fit`, `model_load`) that report their
reason only through the thread-local message.

## Lifecycle

Builders and models own an opaque native handle. Call `Close` when done; it is
idempotent. Every wrapper also installs a `runtime.SetFinalizer` that releases
the handle if the caller forgets, so handles are freed exactly once. Models are
immutable and safe for concurrent use; builders serialize their configuration
and fit calls internally.

## Persistence

`Save` writes the interop model state and `LoadLinearRegression` reads it back.
The native contract currently covers fitted double-precision linear-regression
models: saving a scaler or an `float32` model returns `StatusUnsupported`.

## Scope

Estimator families are exactly those the C ABI carries. Go-native estimators,
asynchronous execution, GPU backends and zero-copy beyond the cgo pointer are
explicit non-goals.

## Testing

```sh
cargo build -p sciencekit_ffi --release
go test ./go/...          # unit, example, concurrency and large-array tests
go test -race ./go/...
```
