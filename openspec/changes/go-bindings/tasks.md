# Tasks

## 1. Module and cgo

- [ ] 1.1 Create the Go module with a cgo preamble, version/status declarations and a
  build smoke test; verify with `go test ./...`.
- [ ] 1.2 Implement builder/model wrappers, `Close` and finalizers; verify with
  `go test ./...`.

## 2. Arrays and operations

- [ ] 2.1 Implement matrix conversion with shape validation, contiguous copies and
  `runtime.KeepAlive`; verify with `go test ./...`.
- [ ] 2.2 Implement fit/predict/transform/score/save/load; verify with `go test ./...`.
- [ ] 2.3 Implement error mapping and closed-handle tests; verify with `go test ./...`.

## 3. Packaging and acceptance

- [ ] 3.1 Add examples, every-family fixtures, concurrent-use and large-array tests; verify
  with `go test ./...`.
- [ ] 3.2 Document the native library build/link flags and add CI; verify in a clean
  environment.
- [ ] 3.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on cgo arrays and finalizers); verify with
  `mdbook build docs`.
- [ ] 3.4 Independent review by a separate agent validating every scenario in
  `go-bindings`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
