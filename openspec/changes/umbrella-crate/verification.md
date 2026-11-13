# Independent Review — Umbrella Crate

Task 3.3 verification of the `umbrella-crate` change.

- Code branch: `feat/umbrella-crate` (PR #402)
- OpenSpec branch: `feat/umbrella-crate-openspec` (PR #260)
- Documentation branch: `docs/umbrella-crate-book` (PR #403)
- Reviewer: separate agent, did not write the implementation.
- Reference state: code branch merged with `origin/main` at `6a38953` (includes
  `lora-adapters`, commit `e33626c`).
- Gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace`, `cargo doc -p sciencekit --no-deps` with
  `RUSTDOCFLAGS=-D warnings`, `mdbook build docs`,
  `openspec validate umbrella-crate --strict` — all green (see §6).

## 1. Completeness — every sub-crate has a door

The plan said `crates/sciencekit`; Cargo cannot host two packages named
`sciencekit`, so the umbrella lives in the existing root package. The deviation
is documented in `design.md` Decision 0, the crate docs and the book chapter, and
the published paths are identical.

Automated sweep: every public item re-exported by a sub-crate root resolves
through the umbrella (702 distinct names collected from `src/*.rs`), and every
public module is addressable. There are **21** sub-crates under `crates/`.

| Umbrella module | Sub-crate(s) | Surface | Verdict |
|---|---|---|---|
| root | `sciencekit_common` | 91 root items + `allocator` (fixed) | PASS |
| `math` | `sciencekit_math` | 65 root items, 7 modules, `prelude` (fixed) | PASS |
| `preprocessing` | `sciencekit_preprocessing` | 5 modules | PASS |
| `impute` | `sciencekit_impute` | 4 modules | PASS |
| `linear_model` | `sciencekit_linear_model` | 11 modules + `adapters`/LoRA (fixed) | PASS |
| `neighbors` | `sciencekit_neighbors` | 24 root items, 7 modules | PASS |
| `svm` | `sciencekit_svm` | 15 root items, 6 modules | PASS |
| `tree` | `sciencekit_tree` | 16 root items, 6 modules | PASS |
| `ensemble` | `sciencekit_ensemble` | 33 root items, 5 modules | PASS |
| `cluster` | `sciencekit_cluster` | 14 root items, 3 modules | PASS |
| `decomposition` | `sciencekit_decomposition` | 12 root items, 3 modules | PASS |
| `outlier` | `sciencekit_outlier` | 9 root items, 1 module | PASS |
| `model_selection` | `sciencekit_model_selection` | 40 root items, 8 modules | PASS |
| `pipeline` | `sciencekit_pipeline` | 35 root items, 5 modules | PASS |
| `metrics` | `sciencekit_metrics` | 6 modules | PASS |
| `interop` | `sciencekit_interop` | 17 modules, feature gates mirrored | PASS |
| `interop` (onnx) | `sciencekit_onnx` | 23 root items under `onnx` | PASS |
| `accelerators` | `sciencekit_gpu` | 21 root items, 5 modules | PASS |
| `accelerators` (opencl) | `sciencekit_gpu_opencl` | 8 items under `opencl` | PASS |
| `telemetry` | `sciencekit_telemetry` | 22 root items, 3 modules | PASS |
| `ffi` | `sciencekit_ffi` | 32 root items, 10 modules | PASS |

Gaps found and fixed:

1. `sciencekit_common::allocator` (`SKAllocatorKind`, `sk_allocator_kind`) had no
   door at all — not at the root and not as a module (commit `918111e`).
2. `sciencekit_math::prelude` was the only public module of the crate not
   re-exported (same commit).
3. `lora-adapters` landed after the branch was written; the explicit
   `linear_model` item list hid the whole adapter API (`SKLoraAdapter`, builder,
   registry, attached/merged models, target module, state helpers). Re-exported
   with a compile test (commit `138afd9`).

No other omission: the sweep reports zero missing root items and zero missing
public modules.

## 2. Collision honesty

The three `SKClassWeight` policies (tree/ensemble, linear-model logistic, SVM)
and the two `sk_weighted_average` functions (neighbors, impute) live only in
their family modules. Confirmed by inspection: none of those names appears in
`src/prelude.rs`, and the prelude is a curated list of explicit `pub use` paths,
never a glob (no `pub use *` in the umbrella).

| Name | Modules | Resolution | Prelude |
|---|---|---|---|
| `SKClassWeight` | `tree`, `ensemble` (one type) / `linear_model` / `svm` | distinct types, family paths | absent |
| `sk_weighted_average` | `neighbors`, `impute` | distinct functions | absent |
| `SKSplitter` | `tree` (enum), `model_selection` (trait) | distinct item kinds, family paths | absent |
| `SKFeatureBlock`, `SKIntoFeatureBlock` | root, `pipeline` | one identity | root |
| device vocabulary | root, `accelerators` | one identity | root |
| `SKOutlierLabel`, `SKOutlierPredictor` | root, `outlier` | one identity | root |
| `SKKernel`, `SKKernelGamma`, `SKKernelParameters` | `math`, `svm` | one identity | absent |
| `SKTreeCriterion`, `SKMaximumFeatures` | `tree`, `ensemble` | one identity | `tree` import |

`SKSplitter` was not recorded by the design; the review added it to Decision 3b,
the prelude docs and the book (commit `353c418`).

Identity is now executable, not just asserted: `shared_identities_are_one_type`
in `src/reexports_tests.rs` assigns between the two spellings for every shared
name (a second definition would fail the build), and `TypeId`-free compile
helpers cover the trait pairs. A user of the prelude cannot obtain a colliding
name: it is not exported there.

## 3. Feature matrix

`cargo check -p sciencekit --no-default-features` plus one build per feature,
each run independently:

| Build | Result |
|---|---|
| `--no-default-features` | PASS |
| `opencl` | PASS |
| `telemetry` | PASS |
| `onnx` | PASS |
| `memmap` | PASS |
| `parquet` | PASS |
| `polars` | PASS |
| `datafusion` | PASS |
| `otlp-grpc` / `otlp-http` | PASS |
| `observability-export` | PASS |
| `allocator-jemalloc` / `allocator-mimalloc` | PASS (tests assert the active allocator) |
| `compression-gzip` / `-brotli` / `-snappy` | PASS |
| `--all-features` | expected FAIL — the two allocator switches are mutually exclusive in `sciencekit_common`; docs.rs metadata lists the integration features instead, and that exact list was doc-built with `-D warnings` |

Two mirroring defects found and fixed:

- `memmap` enables the sub-crate's `arrow` adapter, but the umbrella re-exported
  `interop::arrow` only under `parquet`, so the adapter was compiled and hidden
  (commit `219fe14`).
- `telemetry` forwarded only the dependency, not `sciencekit_telemetry/telemetry`,
  so every OTLP transport stayed off and `sk_build_otlp_pipeline` always returned
  `UnsupportedOperation`. The test builds a real pipeline; it fails with the old
  forwarding and passes now (commit `47da5be`).

`cargo tree -p sciencekit --features <f> -e features` shows no cross-feature
bleed: `polars` pulls no DataFusion, `datafusion` pulls no Polars, `onnx` pulls
no engine adapter, `opencl` pulls no telemetry stack, and the default build
pulls neither `opencl3` nor `opentelemetry-otlp` nor Arrow. The remaining
sub-crate switches were forwarded too (commit `cfc6ed4`): allocator selection,
compression codecs, `observability-export` (and its gated
`sk_opentelemetry_layer`). Three are deliberate, documented omissions:
`blas-backend` (needs a system BLAS at link time) and `gpu-cuda`/`gpu-rocm` (no
backend implemented yet).

## 4. Documentation

- `RUSTDOCFLAGS=-D warnings cargo doc -p sciencekit --no-deps`: PASS, no broken
  intra-doc links. The docs.rs build (`features = [...]`, since `all-features`
  cannot be used with mutually exclusive allocators) also passes.
- One pre-existing flake was observed once during the workspace run:
  `sciencekit_svm`'s `configured_backend_is_recorded_in_fit_and_prediction_spans`
  rebuilds the global tracing interest cache and can miss spans under parallel
  load. The file is untouched by this change; the suite passes alone and the
  full workspace rerun is green.
- Every module listed at the crate root exists; every feature in the table exists
  in `Cargo.toml` and was compiled.
- The crate docs promise nothing the crate does not export: the only difference
  found was the `telemetry` transport gap above, fixed rather than documented
  away.
- The book chapter `ch67-umbrella-crate.md` was amended (docs branch): the
  feature snippet showed the broken `telemetry` forwarding, and the "roads not
  taken" bullet still described `lora-adapters` as a future change. Both are
  corrected. `mdbook build docs`: PASS.

## 5. Convention audit

| Rule | Verdict |
|---|---|
| Umbrella adds no logic | PASS — `src/*.rs` contain only `//!` docs, `pub use`, `pub mod` and test modules |
| No `pub use *` hiding collisions | PASS — zero glob re-exports |
| `SK`/`sk_` naming intact | PASS — names are re-exported verbatim, no local aliases |
| Prelude is curated, not a glob | PASS — explicit paths only |
| Feature forwarding uses `dep:`/sub-crate features | PASS — no version re-declared |
| One identity per shared type | PASS — compile-tested |

## 6. Gates

```
cargo fmt --all --check                                PASS (no output)
matrix: 16 feature builds + --no-default-features        PASS
--all-features                                          expected FAIL (both allocator
                                                        features are mutually exclusive
                                                        by design; docs.rs metadata lists
                                                        the integration features instead
                                                        and that build was verified)
cargo clippy --workspace --all-targets -- -D warnings    PASS (finished, no warnings)
cargo test --workspace                                  PASS (45 suites, 0 failed)
cargo doc -p sciencekit --no-deps (default + docs.rs features)  PASS (-D warnings)
mdbook build docs                                       PASS (book written)
openspec validate umbrella-crate --strict               PASS (Change is valid)
```

## Verdict

**Approved with fixes.** Six defects were found and fixed with red-then-green
tests, one commit per finding: the missing allocator/prelude doors, the
undocumented `SKSplitter` collision, the hidden low-rank adapter API, the
`memmap`/Arrow gate, the transport-less `telemetry` feature, and the rest of the
sub-crate feature surface. The re-export surface is complete for all 21
sub-crates, the three name collisions are contained by family modules and absent
from the prelude, shared identities are compile-proven, and every gate is green.
No merge performed by the reviewer.
