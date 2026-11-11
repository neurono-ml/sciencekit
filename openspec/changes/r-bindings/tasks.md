# Tasks

## 1. Package skeleton and native registration

- [ ] 1.1 Create the R package skeleton (`DESCRIPTION`, `NAMESPACE`, `R/`, `src/`,
  `tests/testthat/`, `README.md`, `LICENSE`); verify the package loads with
  `R CMD build .` and `R -e 'library(sciencekitr)'`.
- [ ] 1.2 Vendor the ABI header with `scripts/sync-header.sh` and add
  `scripts/build-native-library.sh`; verify the native library builds from a checkout of
  the main repository.
- [ ] 1.3 Write the C shim and register its routines with `R_registerRoutines` and
  `R_useDynamicSymbols(dll, FALSE)`; verify the package loads and reports the ABI version.

## 2. Handles, conversion and operations

- [ ] 2.1 Implement the external-pointer handles with a finalizer and an idempotent
  `close()`; verify with a test that closes twice and that an operation on a closed handle
  raises the `invalid_handle` error.
- [ ] 2.2 Implement the dense matrix conversion with rectangularity and type validation
  and the `f64`-only contract; verify with tests for ragged, integer and shape-mismatch
  inputs.
- [ ] 2.3 Implement `fit`, `predict`, `transform` and `score`; verify with the known
  linear fixture and the scaler fixture.
- [ ] 2.4 Implement `save` and `load`; verify the persistence round-trip.
- [ ] 2.5 Implement the structured error mapping to `sciencekit_error` conditions; verify
  the status name, the status code and the native message are all present.

## 3. Packaging, CI and acceptance

- [ ] 3.1 Add the `testthat` suite covering module load, fit/predict/score, transform,
  persistence round-trip, error mapping, close idempotency and no leaks across repeated
  failures; verify with a local run.
- [ ] 3.2 Add the CI workflow that installs R, builds the native library and runs
  `R CMD build .` and `R CMD check .`; verify with a clean-environment run.
- [ ] 3.3 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` subsection on `.Call` registration and external pointers);
  verify with `mdbook build docs`.
- [ ] 3.4 Independent review by a separate agent validating every scenario in
  `r-bindings`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
