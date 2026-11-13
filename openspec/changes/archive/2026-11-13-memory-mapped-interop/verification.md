# Independent Safety Review — `memory-mapped-interop`

Reviewer: independent agentsubagent (did not write the change).
Scope: code worktree `temporary/worktrees/feat/memory-mapped-interop` (PR #355) and docs
worktree `temporary/worktrees/docs/memory-mapped-interop-book` (PR #354), against
`specs/memory-mapped-sources/spec.md`.

Note: both PRs were already merged when the review started; the fixes below land on the
existing branches as follow-up commits and do not merge anything.

## Soundness verdict

| Item | Verdict | Evidence / reasoning |
|---|---|---|
| (a) mapped region cannot be mutated/truncated while a `&[F]` view exists | **CONTRACT — documented, not enforceable** | A read-only `Mmap` cannot stop another process, or safe code holding the same path, from truncating/rewriting the file. Sciencekit never writes the mapping, but the invariant behind `Mmap::map` cannot be enforced portably. It is now stated on the `memmap` module, `sk_map_read_only` and every `open`/`open_with_cast`. Failure mode documented: `SIGBUS` on a truncated page, torn/stale values on rewrite; neither is a catchable `SKError`. |
| (b) alignment of the mapped pointer | **PASS** | `sk_check_alignment` verifies `align_of::<F>()` before every zero-copy view; `.npy` payload offset is validated once at open (`misaligned_header_payload_is_rejected`); raw and page-zero mappings are page-aligned. `SKFloat` is sealed to `f32`/`f64`, so every bit pattern is valid. |
| (c) length arithmetic on crafted headers | **FIXED** | `.npy` computed `element_count * element_size` unchecked → overflow panic on `absurd_npy_dimension_is_a_typed_error` (was `attempt to multiply with overflow`). Now `checked_mul`, returns `MalformedMappedHeader`. Raw descriptor, `element_count`, row offsets and sequential bounds are checked or provably in range. Arrow row counts now use `checked_add`. |
| (d) `.npy` parsing | **PASS** | Magic/version framing guarded (`bytes.len() < 10/12`), header length bounded by file length, shape is `u64 → usize` with rejection, non-2-D shapes, negative values (impossible after `npyz`), Fortran order and big-endian all rejected with structured errors. |
| (e) Arrow IPC offsets/buffer lengths | **PASS + hardened** | Footer length `checked_sub`, start/end `ARROW1` magic, each block's `offset + metadata_length + body_length <= file` verified, negative block offset rejected, metadata parse failure → `UnreadableMappedBatch`. Row count accumulation was unchecked; now checked. `row()` no longer trusts metadata: it `checked_mul`s the offset, bounds it against the decoded batch, and verifies decoded rows == indexed rows. |
| (f) `Send`/`Sync` | **PASS** | Auto-derived: `Mmap` is `Send + Sync`, `AtomicUsize` is shared-safe, Arrow cache is `Vec<OnceLock<Box<[F]>>>` (Sync because `F: Send + Sync`). Compile assertion `mapped_sources_are_send_and_sync` plus three 8-thread reader tests. |
| (g) use-after-close | **PASS** | `Mmap` is owned by the source (`Arc<Mmap>`) and outlives every borrowed row; the `File` handle may drop after mapping (POSIX mapping stays valid). Arrow row views are now derived from `OnceLock::get(&self)` — no raw pointer, no lifetime escape. |

## Hostile-input tests (new `memmap/hostile_input_tests.rs`)

| Input | Expected | Result |
|---|---|---|
| truncated raw file (23 of 48 bytes) | `MappedLengthMismatch` | PASS |
| wrong shape descriptor (2×2 file, 1×8 declared) | `MappedLengthMismatch` naming file and both byte counts | PASS |
| `.npy` with absurd shape (`2^61 × 2` `f8`) | typed `MalformedMappedHeader` (previously overflow panic) | PASS |

Plus pre-existing truncation tests for `.npy` headers, raw files, and Arrow IPC files.

## Spec scenario matrix

| Scenario | Test | Verdict |
|---|---|---|
| Row access does not scan | `raw_binary_tests::row_access_does_not_scan_or_copy`, `random_access_returns_the_requested_row` (pointer identity) | PASS |
| Shape mismatch is rejected | `shorter_file_is_rejected_naming_the_file`, `longer_file_is_rejected`, `hostile_input_tests::wrong_shape_descriptor_is_a_typed_error` | PASS |
| float64 npy maps zero-copy | `numpy_tests::float64_rows_alias_the_mapping` | PASS |
| Fortran-order npy is rejected | `numpy_tests::fortran_order_is_rejected_with_a_c_order_suggestion` | PASS |
| Non-float npy opted in explicitly | `numpy_tests::int64_without_cast_is_rejected`, `int64_with_cast_materializes_values` | PASS |
| Indexed batch lookup is direct | `arrow_ipc_tests::opening_indexes_batches_without_decoding_them`, `row_lookup_decodes_only_the_containing_batch`, `first_middle_and_last_rows_are_correct` | PASS |
| Truncated IPC fails on open or read | `arrow_ipc_tests::truncated_file_is_rejected_on_open` | PASS |
| Sequential and random access agree | `numpy_tests::sequential_blocks_match_random_access`, `arrow_ipc_tests::sequential_streaming_agrees_with_random_access`; raw covered transitively through `row` in `sequential_blocks_respect_...` | PASS (raw equivalence implicit) |
| Block size is respected | raw/npy/arrow sequential tests, `a_large_file_streams_in_bounded_batches` | PASS |
| Concurrent readers | `eight_threads_read_disjoint_rows_from_{a_raw_file,a_numpy_file,an_ipc_file}` | PASS |
| Wrong endianness is rejected | `descriptor_tests::big_endian_descriptor_is_rejected_at_open`, `numpy_tests::big_endian_npy_is_rejected` | PASS |
| Malformed npy header | `numpy_tests::malformed_header_is_rejected_without_panicking`, `truncated_header_is_rejected` | PASS |

Feature matrix: `cargo tree -p sciencekit_interop -e features` shows no `memmap2`/`npyz` with the
feature off; the module is `#[cfg(feature = "memmap")]`, so the neutral crate is unaffected.

## Conventions

- `SK`/`sk_` prefixes respected; methods unprefixed; companion `*_tests.rs` modules beside
  implementations; pure-dispatcher `mod.rs`; no manual index loops outside the copy primitives.
- Construction is `open`/`open_with_cast` + `with_block_size`, matching the other interop sources;
  the estimator builder rule does not apply to data sources.
- Observability: N/A — these are sources, not estimators; there is no backend to record in a span.

## Defects fixed (one commit per finding)

1. `5f83881 fix(interop): reject .npy byte lengths that overflow the address space` — checked
   byte-length arithmetic + hostile-input suite.
2. `46fb312 fix(interop): make Arrow IPC row access safe against crafted batch metadata` — checked
   row-count accumulation, decoded-length verification, bounded row offset, `OnceLock` cache
   replacing the `Mutex` + `from_raw_parts` unsafe block.
3. `46cebc1 docs(interop): state the memory-mapping file-stability contract` — module and per-`open`
   contract with `SIGBUS`/torn-value failure modes.

Docs branch follow-up: `2e82a4f docs(book): keep the memory-mapped chapter in lockstep with the
safety hardening`.

## Gates (all with `CARGO_TARGET_DIR=/home/claudino/.cache/sciencekit-target-memory-mapped CARGO_PROFILE_DEV_DEBUG=0`)

- `cargo fmt --all --check` — green.
- `cargo clippy --workspace --all-targets -- -D warnings` — green (`Finished dev profile`).
- `cargo test --workspace` — green on re-run (`EXIT 0`). The first run failed only
  `sciencekit_svm::observability_and_execution_tests::configured_backend_is_recorded_in_fit_and_prediction_spans`,
  which passes in isolation and on the workspace re-run: a pre-existing global-subscriber test race,
  unrelated to this change.
- `cargo test -p sciencekit_interop` (feature off) — green, 0 failures.
- `cargo test -p sciencekit_interop --features memmap` — green, 0 failures (44 memmap tests).
