# Proposal

## Why

Users should not have to add a dozen `sciencekit_*` crates to start a project. The PRD asks
for a `sciencekit` umbrella crate with unified re-exports and documentation. It also gives
docs.rs and crates.io one obvious entry point.

## What Changes

- New `sciencekit` crate re-exporting the stable public surface of every sub-crate.
- A `prelude` module with the traits and types needed by typical code.
- Feature flags for optional accelerators/integrations (interop formats, Polars/DataFusion,
  ONNX, Python is a separate crate and is not re-exported).
- Unified crate-level documentation and examples.
- No new logic; the umbrella only re-exports.
- Compile tests proving the re-exports resolve and the prelude compiles.
- Explicit non-goals: changing sub-crate APIs, bundling Python bindings, publishing a
  different version line.

## Capabilities

### New Capabilities

- `umbrella-crate`: the re-export surface, prelude, feature flags and documentation entry
  point.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit` workspace member; root `Cargo.toml` membership.
- **APIs:** additive re-exports.
- **Dependencies:** every sub-crate, optional features for heavy integrations.
- **Downstream:** simplifies onboarding and gives docs.rs a single front page.
- **Acceptance criteria:** re-export compile tests, prelude example, docs build and
  feature-flag matrix build.
