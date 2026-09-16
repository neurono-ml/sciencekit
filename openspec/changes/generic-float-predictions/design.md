## Context

See proposal.md (Why). Current contracts fix predictions/targets at `f64` while features are generic over sealed `SKFloat` (`f32`/`f64`). This design introduces `SKPredictor<F>` semantics plus the regressor/classifier split in `sciencekit_common`, threading `F` through pipeline and scorers. Specs: `specs/common-core/scalar-prediction-contract/spec.md`, `specs/common-core/pipeline-scalar-threading/spec.md`.

## Goals / Non-Goals

**Goals:**

- End-to-end `f32`/`f64` parity with zero implicit casts in the observable API.
- Compile-time scalar mismatch rejection in pipelines.
- Keep sealed `SKFloat` (exactly `f32`/`f64`), zero-copy views, builder conventions, and PRD §8.7/§10.3 acceptance.

**Non-Goals:**

- Python dtype dispatch, GPU/BLAS/SIMD per-dtype kernels, per-algorithm migration — separate associated changes.
- Supporting `f16`/`bf16`/third-party floats; `f128` consideration stays out.

## Decisions

- **Generic `SKPredictor<F: SKFloat>` + split traits over one untyped `predict`.** `SKRegressorPredictor<F> -> Array1<F>` preserves precision/throughput; `SKClassifierPredictor -> Array1<i64>` plus `predict_proba<F> -> Array2<F>` fixes the float-encoded-label lie. Alternative (single generic `predict -> Array1<F>` for both) rejected: it forces classifiers to encode labels as floats.
- **Generic `SKTargetView<'a, F>` with `Continuous(ArrayView1<'a, F>)`.** Keeps features and targets in one scalar; `Integer`/`Nominal` untouched, nominal canonicalization to `i64` unchanged. Alternative (keep `Continuous(f64)` + internal cast) rejected: it reintroduces the exact copy this change removes.
- **Pipeline parameterized over `F`.** Mixed-scalar chains fail at compile time; bridging requires an explicit cast stage. Alternative (runtime `SKError::Conversion`) rejected: defers a statically known error to runtime.
- **Scorers split continuous/label.** Continuous metrics over `F`, label metrics over `i64`. Alternative (all-`f64` scorers + cast) rejected: same conversion tax.
- **Sealed `SKFloat` unchanged.** No new scalars; `Send + Sync + 'static` bounds retained for rayon. Execution-plan accounting already carries `scalar_size_bytes`; it now uses `size_of::<F>()` per operation.

## Risks / Trade-offs

- [Risk] Monomorphization doubles generated code per algorithm → Mitigation: only two scalars; hot paths already generic via `azip!`/`par_azip!`.
- [Risk] Breaking change ripples to every estimator/pipeline/scorer/interop signature → Mitigation: single coordinated `sciencekit_common` change; per-algorithm migrations follow as associated changes; migration note documents `f64`-call-site updates.
- [Risk] `f32` precision regressions in tests with tight tolerances → Mitigation: per-dtype tolerances in companion `*_tests.rs`; large+small data acceptance per PRD.
- [Risk] ONNX/Safetensors dtype mapping complexity → Mitigation: map `F` to tensor dtype explicitly; export tests per scalar.

## Migration Plan

1. Land `sciencekit_common` trait changes with deprecation-free breaking bump (`0.x` minor).
2. Migrate in-tree estimators/pipelines/scorers and interop exports in dependency order (common → math → pipeline/model-selection → interop).
3. Document call-site migration (replace `Array1<f64>` predictions with `Array1<F>`; classifiers switch to `i64` labels).
4. Rollback: revert the single change; no data migration involved.
