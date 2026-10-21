## Why

`SKPredictor::predict` returns `Array1<f64>` and `SKTargetView::Continuous` holds `ArrayView1<f64>` regardless of the feature scalar `F: SKFloat`. Users working in `f32` pay a forced upcast on every prediction and must convert back to chain pipeline stages, defeating the zero-copy and `f32` memory/throughput promises.

## What Changes

- **BREAKING**: `SKPredictor` becomes generic over the feature scalar: `SKPredictor<F: SKFloat>` with `fn predict -> Result<Array1<F>, Error>`. A model fitted on `f32` predicts `f32`; fitted on `f64` predicts `f64`. No implicit cast.
- **BREAKING**: Split prediction semantics: introduce `SKRegressorPredictor<F>` (continuous `Array1<F>`) and `SKClassifierPredictor` (canonical `Array1<i64>` labels, dtype-independent) plus `predict_proba<F> -> Array2<F>` where applicable. The old float-encoded class indices (`0.0`/`1.0`) are removed.
- **BREAKING**: `SKTargetView` becomes generic: `SKTargetView<'a, F: SKFloat>` with `Continuous(ArrayView1<'a, F>)`; `Integer` and `Nominal` variants unchanged. `as_continuous()` elevates to `F`, not `f64`.
- **BREAKING**: `SKSupervisedScorer` / scoring contracts become generic over `F` with separate continuous (`ArrayView1<F>`) and label (`ArrayView1<i64>`) paths.
- `SKPipeline` carries the scalar `F` so stage compatibility (`transform: F -> F`, `predict: F -> F|i64`) is checked at compile time.

## Capabilities

### New Capabilities

- `common-core/scalar-prediction-contract`: generic predictor/target/scorer contracts over `SKFloat` (regressor vs classifier split, pipeline scalar threading, acceptance: f32/f64 parity, concurrency, export+metrics).
- `common-core/pipeline-scalar-threading`: compile-time scalar propagation through `SKPipeline` stages.

### Modified Capabilities

(none — `openspec/specs/` holds no published capabilities yet; the contracts above are introduced here as new deltas.)

## Impact

- Affected code: `sciencekit_common` (SKFloat-adjacent traits, `SKDataView`/`SKTargetView`, fit/predict/transform/scorer traits, execution-plan scalar size accounting); every current and future estimator, pipeline, model-selection scorer, and interop export (Safetensors/ONNX dtype mapping).
- Downstream (separate associated changes, not this one): Python dtype dispatch (`f32`/`f64` NumPy), GPU/BLAS/SIMD kernels per dtype, per-algorithm migrations.
- Acceptance follows PRD §8.7/§10.3: large and small data, concurrency, model export plus metrics, companion `*_tests.rs`.
