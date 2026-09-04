## Why

Fourteen `mod.rs` files across `sciencekit_common` and `sciencekit_math` violate the
established convention that a `mod.rs` is a pure dispatcher (declarations and re-exports
only), carrying instead their module's implementation inline. There is no explicit rule
that inline tests must live at the end of their file. This change restores the
convention, codifies the "tests at the end" rule, and is a **pure refactor** — no
observable behavior changes, so it opts out of specs (`skip_specs: true`).

## What Changes

- **BREAKING** (structural, not behavioral): split implementation out of each
  non-conforming `mod.rs` into submodules, leaving `mod.rs` as a pure dispatcher that
  only declares submodules and re-exports their public items.
- **BREAKING** (structural): move inline tests so any `#[cfg(test)]` block in a file with
  implementation sits at the very end, and declare companion `*_tests.rs` modules at the
  end of `mod.rs`.
- Add the "tests at the end" rule to `AGENTS.md` (and mirror it in the OpenSpec
  `config.yaml` context).
- Files over 200 lines introduced by the split follow the standardized folder-module
  layout (`mod.rs`, `builder.rs`, `core_implementation.rs`, `*_tests.rs`).
- **Scope note / dependency**: `backend/mod.rs` (the 14th module) is **excluded** here —
  it is fully restructured by the companion change `backend-kernel-expansion`, which
  owns the backend rewrite. This change covers the other 13 modules.

## Capabilities

### New Capabilities
<!-- None: pure refactor, no new behavior. -->

### Modified Capabilities
<!-- None: no requirement-level behavior changes. This change is a pure refactor and
     sets skip_specs: true (see .openspec.yaml). -->

## Impact

- `crates/sciencekit_common/src/`: `errors`, `execution`, `fit_traits`, `scorer_traits`,
  `data_view`, `batching`, `label_table`, `target_view`, `sk_float` — split into
  implementation submodules with pure-dispatcher `mod.rs`.
- `crates/sciencekit_math/src/`: `kernels`, `pairwise`, `layout`, `sparse_ops` — split
  into implementation submodules with pure-dispatcher `mod.rs`. (`backend/` handled by
  `backend-kernel-expansion`.)
- `AGENTS.md` and `openspec/config.yaml` — new "tests at the end" rule.
- No dependency changes, no public-API signature changes, no behavior changes. All
  existing tests must stay green after the mechanical move.
- **Dependency / ordering**: applies after `backend-kernel-expansion` (so the 14th module
  is covered exactly once). If applied in parallel, it MUST NOT touch `backend/`.