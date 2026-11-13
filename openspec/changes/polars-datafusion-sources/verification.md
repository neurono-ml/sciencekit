# Independent verification — `polars-datafusion-sources`

Reviewer: separate agent, not an author of this change. Reviewed the code branch
`feat/polars-datafusion-sources` (PR #358) against every scenario in
`specs/tabular-sources/spec.md`, the `AGENTS.md` conventions and the honest-fallback claims.

## Scenario verdicts

| # | Scenario | Verdict | Evidence |
|---|---|---|---|
| 1 | Numeric columns become a feature block | PASS | `tabular::materialization_tests::tabular_source_numeric_columns_form_feature_block_in_order` asserts the exact dense matrix in config order. |
| 2 | Categorical columns become a categorical view | PASS | `tabular_source_categorical_columns_form_owned_block` (codes + label table) and `missing_tests::tabular_missing_categorical_null_becomes_sentinel` (documented sentinel). |
| 3 | Ignored columns are excluded | PASS | `tabular_source_ignored_columns_are_excluded`, `polars_dataframe_applies_column_selection`; the ordered matrix also proves the feature width excludes the ignored column. |
| 4 | DataFrame conversion | PASS | `polars_dataframe_maps_numeric_columns_in_order`, `..._maps_categorical_columns`, `..._projects_the_configured_schema` (projected schema order + `TryFrom`). |
| 5 | LazyFrame batches | WEAK (documented deviation) | `polars_lazy_source_streams_ordered_batches`, acceptance `polars_acceptance_lazy_batches_keep_order`. Polars 0.46 exposes no incremental Rust stream over a `LazyFrame`; the source executes the plan once and slices it into bounded batches. It never claims otherwise (`lazy_source.rs` type docs, mdBook ch. 57 §4). The scenario wording in the spec delta and design Decision 3 were amended to state this instead of claiming bounded plan streaming. |
| 6 | RecordBatch conversion | PASS | `datafusion_record_batch_matches_the_dataframe_outputs` compares schema, dense features, codes and label tables against the DataFrame path. |
| 7 | Parquet through DataFusion | PASS | `datafusion_parquet_converts_with_the_same_rules` now asserts row order, the exact feature matrix, categorical codes and schema width. |
| 8 | Numeric null | PASS | `tabular_missing_numeric_null_becomes_nan`, `polars_dataframe_maps_numeric_nulls_to_nan`, `datafusion_missing_values_follow_the_policy` (asserts NaN, not the height). |
| 9 | Categorical null | PASS | `tabular_missing_categorical_null_becomes_sentinel`, `polars_dataframe_maps_categorical_nulls_to_sentinel`, `datafusion_missing_values_follow_the_policy` (asserts `-1`). |
| 10 | Unsupported dtype | PASS | `tabular_source_kind_mismatch_names_the_column` (names `label`), `polars_dataframe_rejects_an_unsupported_dtype`, `datafusion_record_batch_rejects_an_unsupported_dtype`. |
| 11 | Concurrent reads agree | PASS | `polars_acceptance_large_table_and_concurrent_reads`, `datafusion_acceptance_large_table_and_concurrent_reads`, `datafusion_dataset_reads_agree_across_threads` (4 threads × tiny + 2 000 rows). |
| 12 | Validation taxonomy | PASS | `tabular_source_rejects_empty_input` (`EmptyInput`), `..._unknown_column_is_rejected` (both engines), `dataset_rejects_row_count_mismatch` (`UnevenColumnLengths`), `dataset_rejects_schema_width_mismatch` (`SchemaMismatch`), `..._rejects_infinite_values` (`NonFiniteValue`), `..._reject_policy_names_the_column` (`NullValueInColumn`). |

## Convention audit (`AGENTS.md`)

- **Prefixes** (`SK` types, `sk_` free functions): compliant across every new public item.
- **No abbreviations**: compliant (`datafusion` is a proper noun; no `max`/`idx`/`tmp`).
- **Builder-only construction**: `SKTabularSourceConfig` has private fields and a validating
  builder. `SKTabularDataset`, `SKNominalMatrixOwned`, `SKPolarsLazySource` and
  `SKDataFusionStreamSource` are value/source types with fallible `new`, not estimators; no
  configurable surface was left bypassable.
- **`backend(SKBackendKind)` / observability span**: not applicable — this change adds data
  adapters, no estimator and no `fit`/`predict`/`transform` operation.
- **Dense/sparse parity, no densification**: both layouts are produced directly (`to_dense`
  stacks columns, `to_sparse_csr` builds triplets; ignored/categorical paths never build a
  dense matrix). `tabular_source_honors_sparse_layout` now compares every sparse entry against
  the dense block and checks that real zeros stay unstored.
- **Folder modules with pure-dispatcher `mod.rs`**: `tabular/`, `polars/` and `datafusion/`
  `mod.rs` only declare/re-export; tests are declared last.
- **`*_tests.rs` companions at the end**: compliant (`nominal_matrix_owned_tests.rs` added and
  declared last in `tabular/mod.rs`).
- **No manual index loops**: the hot-path DataFusion column converter was rewritten to
  `iter().map(..)`; the remaining index expressions are guarded slice picks in
  `sk_dataset_from_batches`/`sk_concatenate_column` and a small canonicalized-code scatter in
  `SKNominalMatrixOwned::from_extraction`, not compute loops.
- **File size**: implementation files are ≤ 200 lines; the test companions
  (`datafusion_tests.rs`, `acceptance/tabular_source_tests.rs`) exceed it, matching the rest of
  the repository's test layout.

## Defects found and fixed

1. **Missing `parquet` feature** — the proposal, design and task 3.3 promise
   `--features datafusion,parquet`, but the manifest had no such feature, so the documented
   command failed. `parquet = ["arrow"]` now exposes it without duplicating the dependency.
2. **Nested-runtime panic** — `SKDataFusionStreamSource::new/open/next_batch` blocked on their
   own Tokio runtime and panicked when called inside another runtime. They now return
   `UnsupportedOperation`; proven by `datafusion_stream_source_refuses_to_nest_runtimes`.
3. **Undocumented-invalid nominal blocks** — `SKNominalMatrixOwned::new` documented a validation
   it did not perform (a code could point outside its label table). It now rejects undecodable
   codes while accepting the missing sentinel; proven by `nominal_matrix_owned_tests.rs`.
4. **Weak scenario tests** — RecordBatch/Schema, Parquet/row order and DataFusion missing-value
   scenarios were asserted only through row counts; they now pin the actual outputs. Polars now
   asserts the projected schema and `TryFrom`; sparse layout asserts parity.
5. **Manual index loops** in the DataFusion array converter, against the `AGENTS.md` rule;
   rewritten with `iter()`/`map`.

All fixes are separate commits (`fix(interop): ...`, plus two test/refactor commits). No
`unwrap`, `expect`, `panic!` or indexing on user input exists in the new library code.

## Gates (green, run in the code worktree)

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p sciencekit_interop --features polars,datafusion,arrow
cargo test -p sciencekit_interop --features datafusion,parquet   # was failing before fix 1
cargo clippy -p sciencekit_interop --all-targets --features polars,datafusion,arrow -- -D warnings
```

Default build (`cargo test --workspace`) passes without either engine, so the feature matrix
is correct both on and off; `--features arrow` and `--features polars` were also verified
individually.

## Residual notes

- The Polars `LazyFrame` limitation is a property of Polars 0.46, not a code defect; it is
  documented in the type, the design and the mdBook chapter, and the spec wording now matches.
- No other scenario is unmet.
