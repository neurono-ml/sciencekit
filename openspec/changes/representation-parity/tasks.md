# Tasks

## 1. Spec delta

- [x] 1.1 Add the representation-parity, dense-only-exception and mixed-representation
  requirements to `openspec/changes/representation-parity/specs/data-view-boundary/spec.md`;
  verify with `openspec validate representation-parity`.

## 2. PRD amendment

- [x] 2.1 Update `docs/PRD.md` §4 (representations) to state that algorithms tolerating
  dense and sparse representations must accept dense, sparse and mixed inputs.
- [x] 2.2 Update `docs/PRD.md` §4.5 (sparse support and representation parity) and §12 Phase 3/4 to name the
  algorithms that rely on the dense-only exception and the alternative path that keeps
  estimator-level parity.
- [x] 2.3 Verify the amended PRD paragraphs are internally consistent with
  `openspec/specs/data-view-boundary/spec.md` and with the in-flight neighbours, SVM and
  sparse-pairwise proposals.

## 3. Agent rule

- [x] 3.1 Add the representation-parity rule and the dense-only exception to the
  `Code conventions` section of `AGENTS.md`, naming the spec as the normative source.
- [x] 3.2 Verify every future algorithm change template still points at the affected
  capabilities and that the new rule is visible in the planning instructions: the
  `openspec/config.yaml` project context now restates it.

## 4. Documentation and changelog

- [x] 4.1 Update the `Development Explained` data-boundary chapter (or add a subsection to
  the existing chapter) explaining dense, sparse and mixed parity with a vivid Mermaid
  diagram, and verify with `mdbook build docs`.
- [x] 4.2 Update `CHANGELOG.md` (English, keep-a-changelog 1.1.0) under `Unreleased →
  Changed` for the new representation-parity rule and the dense-only exception.

## 5. Review

- [x] 5.1 Independent review by a separate agent validating that the spec, PRD and AGENTS
  wording agree and that the rule does not contradict `estimator-contracts`,
  `backend-selection` or `pairwise-distances`; address every finding.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
