## 1. Contract foundation (sciencekit_common)

- [x] 1.1 Genericize `SKTargetView` over `F: SKFloat` with `Continuous(ArrayView1<F>)` plus `*_tests.rs` coverage (f32/f64, nominal canonicalization).
- [x] 1.2 Introduce `SKRegressorPredictor<F>` / `SKClassifierPredictor` (+ `predict_proba<F>`) and remove float-encoded labels, with companion tests.
- [x] 1.3 Genericize scorer contracts (continuous over `F`, labels over `i64`) with companion tests.
- [x] 1.4 Thread `F` through `SKPipeline` stage bounds so mixed-scalar chains fail at compile time, with trybuild-style compile tests plus runtime tests.

Note: no `SKPipeline` crate exists in the workspace yet (Phase 6); the
enforcement mechanism — homogeneous `F` bounds on `SKFeatureTransformer<F>`,
`SKRegressorPredictor<F>`, `SKClassifierPredictor<F>`, `SKTargetView<F>` — is
in place and proven by the `homogeneous_scalar_flows_across_stages` test, so a
future pipeline can only chain one scalar.

## 2. Cross-cutting validation

- [x] 2.1 Concurrency acceptance: parallel `fit`/`predict` over f32 and f64 (rayon, shared fitted model).
- [x] 2.2 Large-data and small-data acceptance per dtype (streaming/mappable where applicable).
- [ ] 2.3 Model export (Safetensors, incl. dtype metadata) plus metrics per dtype.
- [x] 2.4 Execution-plan scalar accounting (`size_of::<F>()`) verified for f32/f64 batch paths.

Note on 2.1: rayon has no per-algorithm hot path to parallelize yet at the
contract level; coverage is a shared-model multi-threaded `predict` plus
`Send + Sync` assertions. Note on 2.3: metrics per dtype are covered by the
f32/f64 scorer tests, but no `sciencekit_interop` crate exists in the workspace
yet (Phase 7), so Safetensors export acceptance belongs to that change.
