# Proposal

## Why

There are 55 active OpenSpec changes across Waves 3–7 and cross-cutting work. Agents need
one authoritative dependency graph, with completion checks, so they can start independent
changes in parallel, block on real dependencies, and avoid merge conflicts in shared
crates. Today the ordering is spread across wave-plan prose and per-change designs.

## What Changes

- Add `openspec/changes/implementation-dag/dag.yaml`: a machine-readable graph of every
  active change with `depends_on`, optional `soft_depends_on`, lane, join semantics and
  completion checks.
- Add a human-readable design with the layered view, parallel groups, merge lanes and the
  automated execution loop.
- Define completion checks: OpenSpec status, package/language tests, workspace gates,
  docs build, independent review and PRs merged.
- Define lane capacities so two agents never edit the same crate concurrently.
- Document join nodes (`model-export-retrofit`, `umbrella-crate`,
  `python-bindings-complete`, `workflows-cli`) and how they can be split into family-level
  work.
- No application code and no runtime behavior change; `.openspec.yaml` sets
  `skip_specs: true`.

## Capabilities

### New Capabilities

None. This is a planning/coordination artifact; no spec-level behavior changes.

## Impact

- **Docs/specs:** new `dag.yaml`, `design.md` and tasks under
  `openspec/changes/implementation-dag/`.
- **Downstream:** every future agent reads `dag.yaml` to compute ready/blocked changes.
- **Acceptance criteria:** the YAML parses, every dependency resolves to a known change,
  every change has a lane and completion checks, and the design matches the YAML.
