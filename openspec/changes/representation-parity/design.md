# Design

## Context

See `proposal.md` — Why. `data-view-boundary` already defines the dense/sparse view, the
zero-copy conversion seam, one-time dispatch and precise rejection of unsupported
representations. `backend-selection`, `estimator-contracts` and `pairwise-distances`
define the surrounding contract. The missing piece is the cross-cutting rule that tells
algorithm changes which representations they owe and when an exception is legitimate.

## Goals / Non-Goals

**Goals:**
- One normative rule for dense/sparse/mixed parity at the data boundary.
- A narrow, auditable dense-only exception for algorithms whose geometry cannot be sparse.
- PRD and AGENTS language that makes the rule visible during planning.

**Non-Goals:**
- Implementing sparse support in any algorithm.
- Changing the `SKDataView` enum or its conversions.
- Requiring GPU, out-of-core or streaming parity from this change.

## Decisions

### Decision 1 — Put the rule in `data-view-boundary`, not in each algorithm spec

The rule is about the data boundary, not about neighbours or SVM specifically. Adding it
to the existing capability gives every future algorithm one place to cite and lets
`openspec validate` check that scenarios exist. Repeating it in each estimator spec would
drift.

Alternative considered: a new `representation-parity` capability (rejected — it would
overlap `data-view-boundary` and split one behavior across two specs).

### Decision 2 — Define "tolerates" as mathematical, not effort-based

An algorithm tolerates a representation when its formula can be evaluated without
densifying the input and without changing the result. The rule therefore covers distance
kernels, linear models, SVM kernels, brute-force neighbours and imputation on dense data.
It does not force a sparse KD-tree: axis-aligned bounding boxes and ball centroids are
dense geometry, so the tree may declare the exception while the estimator-level surface
covers sparse input through the brute-force searcher.

Alternative considered: requiring every algorithm to support every representation
unconditionally (rejected — it would force dense geometry to become sparse at unbounded
memory cost and contradict the zero-copy promise).

### Decision 3 — The exception must name its alternative path

A dense-only declaration is only acceptable when the public estimator still serves the
unsupported representation through a documented path. The proposal of the invoking change
must state why the mathematics requires dense geometry and name the alternative. This
keeps the exception auditable without blocking legitimate dense-only kernels.

### Decision 4 — Two edits keep the rule alive

`docs/PRD.md` states the product expectation and the exceptions per wave; `AGENTS.md`
states the implementation rule so every future change plan carries it. The spec delta is
the normative source, PRD is the product narrative and AGENTS is the agent instruction.

## Risks / Trade-offs

- **Rule inflation** → "tolerates" is defined mathematically and the exception is narrow,
  so not every algorithm is forced into a sparse rewrite.
- **PRD/AGENTS drift from the spec** → the change updates all three in one task group and
  the independent review checks them against the spec.
- **Future change authors miss the rule** → the AGENTS clause is mandatory reading and the
  change template already asks for affected capabilities.

## Migration Plan

Documentation-only: no code, no runtime behavior. Reverting restores the previous wording
and removes the delta.

## Open Questions

None. Individual algorithms decide in their own proposals whether they tolerate a
representation or invoke the dense-only exception.
