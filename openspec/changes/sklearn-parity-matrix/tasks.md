# Tasks

## 1. Reference surface

- [x] 1.1 Add `data/sklearn-1.9.1-api.txt` with the pinned public estimator, transformer
  and function-family listing; verify it is stable, sorted and reproducible from the
  recorded scikit-learn version.
- [x] 1.2 Add the schema header to `parity.yaml` (`baseline`, `reference`, `crates`,
  `entries`) and document the `kind`, `tier` and `alternative` vocabularies; verify the
  file parses as YAML.

## 2. Matrix authoring

- [x] 2.1 Fill the 246 estimator and transformer rows with `status`, `crate`, `change`,
  `tier` and, where applicable, `reason`; verify no baseline class is missing or
  duplicated.
- [x] 2.2 Fill the function-family rows (metrics, dataset generators and loaders,
  preprocessing functions, inspection utilities, multiclass and multioutput utilities);
  verify every entry names a crate or an explicit absence reason.
- [x] 2.3 Cross-check every `planned` row against `implementation-dag/dag.yaml`; verify no
  planned row names a change absent from the graph.
- [x] 2.4 Audit the implemented estimators against the parity doctrines
  (`conformance-audit.md`): scan streaming/partial_fit/sample_weight/export/backend per
  estimator, review the in-flight changes, and record the corrective actions.
- [x] 2.5 Record `reuse-map.md`: existing core primitives per family, the sixteen shared
  primitives to promote or create, the family-to-primitive map and the six reuse rules;
  verify every planned algorithm family names the primitives it consumes and every
  primitive lists at least three consumers.

## 3. Consistency tooling

- [ ] 3.1 Add the parity checker (single-status, exhaustive coverage, tier presence,
  owner resolution, absence reasons) and a failing fixture per violation; verify each
  fixture fails with the offending row named.
- [ ] 3.2 Add the implemented-entry coverage check (every `implemented` row maps to its
  change's completion test) and a fixture for a stale row; verify it reports both the row
  and the missing test.
- [ ] 3.3 Wire the checker into CI before the workspace gates; verify a deliberately
  broken matrix fails the job.

## 4. Documentation and roadmap

- [ ] 4.1 Add a documentation page with the coverage counts per family and per status,
  generated from `parity.yaml`; verify `mdbook build docs` and that the page is registered
  in `SUMMARY.md`.
- [ ] 4.2 Revise PRD §12 to reference the matrix as the authority and reconcile the §1.1
  "all algorithms" promise with the enumerated roadmap; verify no remaining contradiction.

## 5. Graph

- [ ] 5.1 Extend `openspec/changes/implementation-dag/dag.yaml` with the parity program
  (structural changes first, then one change per algorithm family) and its lanes;
  verify the graph parses, every dependency resolves and the lane merge orders match.
- [ ] 5.2 Update the `implementation-dag` design tables (lane orders and parity waves) to
  match the YAML; verify with the consistency check.
- [x] 5.3 Add the sixteen shared primitives and the `iterative-imputer` gap to the graph,
  rewire 104 algorithm changes onto them, and add the `stats`, `graph` and `impute`
  lanes; verify the graph parses, every dependency resolves and same-lane ordering holds.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English (Y-statement for the pinning
  decision, Nygard if the scope decision is recorded too) and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
