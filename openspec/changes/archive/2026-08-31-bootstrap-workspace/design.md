# Design — bootstrap-workspace

## Context

Repository without a Cargo workspace, unpinned toolchain and no CI (see proposal — Why). PRD constraints: MSRV Rust 1.85, edition 2024 (§3.6), Apache-2.0 license (§11.3), small PRs with continuous quality gates (§3.2). The workspace is born with no members; the 19 sub-crates arrive one per change.

## Goals / Non-Goals

**Goals:**
- Identical build environment on any machine: toolchain pinned by a versioned file, not by convention.
- A single CI gate serving every future crate without per-crate edits (`--workspace` covers everything).
- Clean base for mandatory TDD: green `cargo test` on a fresh clone.

**Non-Goals:**
- Creating empty sub-crates in bulk (churn without value; each crate is born in its own change).
- Configuring branch protection on GitHub (administrative action outside the repository).
- Release profiles, `cargo-deny`, coverage, benchmarks — they arrive when there is code justifying them.

## Decisions

1. **Toolchain via `rust-toolchain.toml` with exact channel `1.85`**, not `stable`.
   - *Why:* absolute reproducibility and an effective MSRV gate — if anyone uses newer features, the pinned build breaks in their own environment, before CI.
   - *Rejected alternative:* the `stable` channel — floats with releases and masks MSRV violations.

2. **Root workspace with empty members.**
   - *Why:* a lone `[workspace]` already validates the whole command chain (`fmt`, `clippy --workspace`, `test --workspace`) and avoids creating 19 empty stubs that would turn into merge conflicts.
   - *Rejected alternative:* full sub-crate scaffolding now — big churn, zero behavior.

3. **CI in a single workflow with separate jobs per gate.**
   - Jobs: formatting; static analysis with `-D warnings`; workspace tests (includes doctests); examples (build + test); dedicated MSRV at exactly 1.85.
   - *Why:* failures get attributed to the specific gate; independent jobs run in parallel.
   - *Rejected alternative:* a monolithic job — worse diagnostics with no real gain.

4. **Strict clippy from day zero (`-D warnings`) instead of progressive adoption.**
   - *Why:* straightening out later is more expensive; today the cost is zero (no code).

5. **Examples as tested citizens from the start.**
   - `cargo build/test --examples` in CI even with no existing examples — the gate is already active when the first example appears, and it will feed the mdBook on the `docs/documentations` branch.

## Risks / Trade-offs

- [Exact `1.85` channel may lag behind `stable` fixes] → consciously accepted: predictability outweighs automatic patches; a toolchain upgrade is an explicit, reviewable change.
- [CI without dependency caching gets slow as crates grow] → mitigable later with `~/.cargo` caching on Actions; does not block this change.
- [Empty workspace does not exercise `--all-targets` on real code] → accepted: the gates prove the chain; real coverage begins with `sciencekit_common`.

## Migration Plan

Additive change over the repository root; rollback = revert the merge. No consumers yet, no migration.

## Open Questions

None.
