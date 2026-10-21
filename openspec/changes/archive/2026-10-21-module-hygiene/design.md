## Context

See proposal.md — Why. Fourteen `mod.rs` files carry implementation inline instead of
being pure dispatchers. The repo already establishes the target pattern in
`allocator/`, `builders/` and `observability/`: `mod.rs` declares submodules and
re-exports; each implementation lives in its own file under the folder, with companion
`*_tests.rs` beside it. This change is a mechanical, behavior-preserving refactor.

## Goals / Non-Goals

**Goals:**
- Make every one of the 14 `mod.rs` a pure dispatcher (declarations + re-exports only).
- Place inline tests at the end of any file that contains implementation.
- Codify the "tests at the end" rule in `AGENTS.md` and `openspec/config.yaml`.
- Keep all existing tests green; zero behavior change.

**Non-Goals:**
- No public-API or behavior changes; no dependency changes.
- `backend/` is intentionally excluded — owned by the companion `backend-kernel-expansion`
  change (the 14th module).
- No logic reordering or refactor beyond moving code between files.

## Decisions

**Decision 1 — Apply the pure-dispatcher rule to all 14 modules, grouping by concern.**
To honour "all 14" while avoiding over-fragmentation of tiny leaf modules, each split
groups the implementation into the minimum descriptive files that read cleanly, rather
than one file per function. `mod.rs` becomes declarations + re-exports only.

**Decision 2 — Target layouts (proposed, refined during implementation).**
```
sciencekit_math:
  kernels/     mod.rs + elementwise.rs + reductions.rs + scaling.rs
  pairwise/    mod.rs + squared_euclidean.rs + manhattan.rs + cosine.rs + simd_dot.rs
  layout/      mod.rs + memory_layout.rs
  sparse_ops/  mod.rs + csr_dense.rs + sparse_sparse.rs

sciencekit_common:
  errors/      mod.rs + error_kind.rs            (+ constructors beside it)
  execution/   mod.rs + modes.rs + context.rs + resolve.rs
  fit_traits/  mod.rs + supervised.rs + unsupervised.rs + transformer.rs + predictor.rs
  scorer_traits/ mod.rs + supervised.rs + unsupervised.rs
  data_view/   mod.rs + data_view.rs + conversions.rs
  batching/    mod.rs + data_batch.rs + lazy_source.rs + mappable_source.rs
  label_table/ mod.rs + label_table.rs + canonicalize.rs
  target_view/ mod.rs + target_view.rs + conversions.rs
  sk_float/    mod.rs + definition.rs            (+ private seal beside it)
```
A single small leaf module (e.g. `sk_float`, `errors`) may use one descriptive
implementation file plus its companion `*_tests.rs`, keeping the split meaningful rather
than cosmetic.

**Decision 3 — Tests at the end.**
Any inline `#[cfg(test)] mod …` sits at the very end of its file; companion `*_tests.rs`
declarations go at the end of `mod.rs`. No inline tests exist today (verified), so this
is codification plus a guard for future work.

**Decision 4 — Documentation rule.**
Add to `AGENTS.md` (and mirror in `openspec/config.yaml` context): when tests are inline
in the same file as implementation, the `#[cfg(test)]` block MUST be at the end of the
file, so implementation reads top-down; tests otherwise live in companion `*_tests.rs`.

## Risks / Trade-offs

- [Mechanical move introduces a regression] → Move code verbatim, no logic reordering;
  the existing suite must stay green after each module's split (TDD-style: run tests
  after each move).
- [Over-fragmentation of leaf modules] → Decision 1/2: group into minimal descriptive
  files; a 30-line module becomes one implementation file + mod.rs, not ten tiny files.
- [Conflicts with `backend-kernel-expansion`] → `backend/` is excluded here; apply after
  that change (or in parallel without touching `backend/`).

## Migration Plan

Refactor-only; each module is split and re-tested in place. No deployment or rollback
surface beyond git history.

## Open Questions

None — deferrable naming of individual implementation files is resolved during
implementation without changing the approach.