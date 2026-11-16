# Tasks

Every task follows red-green-refactor: companion test first, confirmed failure, minimal
implementation, refactor with the gates green.

## 1. Vocabulary and helpers

- [ ] 1.1 Define the span vocabulary constants (names, fields, levels, the step enum and
  `SK_TRACE_BATCH_SPANS_PER_EPOCH`) and the conformance enumeration; verify with
  `cargo test -p sciencekit_common tracing_vocabulary`.
- [ ] 1.2 Add `sk_run_child_operation` (parent-guard nesting) with nesting and error
  propagation tests; verify with `cargo test -p sciencekit_common tracing_hierarchy`.
- [ ] 1.3 Add the `access_pattern` field to operation spans; verify with
  `cargo test -p sciencekit_common operation_attributes`.

## 2. Pipeline and streaming hierarchy

- [ ] 2.1 Emit `sk.pipeline` and ordered `sk.pipeline.stage` spans in fit and predict, with
  hierarchy tests; verify with `cargo test -p sciencekit_pipeline tracing_hierarchy`.
- [ ] 2.2 Emit `sciencekit.operation.epoch` (DEBUG) and `sciencekit.operation.batch`
  (TRACE) spans in the streaming driver with ordered indices, the producer-thread span
  propagation, the per-epoch cap and the `batches_aggregated` closing event; verify with
  `cargo test -p sciencekit_common streaming_driver`.
- [ ] 2.3 Emit `sk.operation.step` spans in one representative iterative algorithm with a
  phase test; verify with `cargo test -p sciencekit_cluster tracing_step`.

## 3. Inference and conformance

- [ ] 3.1 Prove predict/transform/score emit the same vocabulary as fit; verify with
  `cargo test -p sciencekit_common tracing_hierarchy`.
- [ ] 3.2 Run the vocabulary conformance test against the pipeline, a streamed fit and an
  inference; verify with `cargo test --workspace tracing_hierarchy`.

## 4. Errors and FFI

- [ ] 4.1 Record `error` and `error.kind` at the origin span and status ERROR with
  `child failed` on every ancestor; verify with
  `cargo test -p sciencekit_pipeline tracing_error_chain`.
- [ ] 4.2 Add the FFI `traceparent` getter and the optional `traceparent` parameter on the
  FFI entry points with the remote-parent nesting; verify with
  `cargo test -p sciencekit_ffi traceparent`.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
