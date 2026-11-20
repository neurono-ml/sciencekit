# Tasks

## 1. Package and discovery

- [ ] 1.1 Create the Julia package skeleton (`Project.toml`, module, library discovery);
  verify `julia --project -e 'using ScienceKit'`.
- [ ] 1.2 Add ccall declarations for version/status/handles and a smoke test; verify with
  `Pkg.test()`.

## 2. Arrays and operations

- [ ] 2.1 Implement array conversion with `GC.@preserve`, contiguous checks and output
  copying; verify with `Pkg.test()`.
- [ ] 2.2 Implement builder/model wrappers and fit/predict/transform/score; verify with
  `Pkg.test()`.
- [ ] 2.3 Implement save/load and error mapping/finalizers; verify with `Pkg.test()`.

## 3. Packaging and acceptance

- [ ] 3.1 Add tests for every family, non-contiguous arrays, errors, finalizers and
  concurrent use; verify with `Pkg.test()`.
- [ ] 3.2 Add CI instructions for Julia 1.10+ and the native library build; verify with a
  clean-environment run.
- [ ] 3.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on ccall and GC preservation); verify with
  `mdbook build docs`.
- [ ] 3.4 Independent review by a separate agent validating every scenario in
  `julia-bindings`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
