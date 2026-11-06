# Proposal

## Why

Dense and sparse support is currently uneven across the library: the pairwise-distance
kernels are dense-only (with a sparse change in flight), some estimators fit CSR but
others silently require dense input, and mixed dense/sparse operations have no explicit
contract. The PRD promises zero-copy dense *and* sparse data from day one, and Wave 3/4
algorithms (neighbours, SVM, trees, ensembles) will otherwise each invent a different
subset. The rule needs to be written once, at the boundary capability, so every later
change inherits it.

## What Changes

- `data-view-boundary` gains an explicit **representation parity** rule: every algorithm
  that mathematically tolerates more than one feature representation SHALL accept dense,
  sparse and mixed dense/sparse inputs, dispatch once per operation and never densify a
  sparse input.
- The rule defines the **dense-only exception**: an algorithm whose mathematics genuinely
  requires dense geometry (for example, axis-aligned tree bounds) MAY declare dense-only
  support only when its public estimator surface still covers dense, sparse and mixed
  inputs through a documented alternative path, and it rejects the unsupported
  representation with `UnsupportedRepresentation` plus a suggestion.
- `docs/PRD.md` (§4 representations, §6.4 sparse support, §12 Phase 3/4) is amended to
  state the rule and to mark the algorithms that rely on the dense-only exception.
- `AGENTS.md` code conventions gain the same rule so every future change plan carries it.
- No code changes and no breaking changes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `data-view-boundary`: adds the representation-parity requirement, the mixed
  two-matrix requirement and the dense-only exception contract.

## Impact

- **Docs/specs:** `openspec/specs/data-view-boundary/spec.md`, `docs/PRD.md`,
  `AGENTS.md`.
- **Downstream:** every algorithm change from Wave 3 onward must show how it satisfies
  the rule or which documented exception it invokes. The neighbours, SVM and tree changes
  already in flight reference this capability.
- **Acceptance criteria:** the rule is validated by spec scenarios; no runtime behavior is
  introduced by this change.
