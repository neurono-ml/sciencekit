# Tasks

## 1. Crate, module and boundary

- [ ] 1.1 Create `crates/sciencekit_python` with `Cargo.toml`, `lib.rs` and the pymodule;
  add it to `[workspace.members]`; verify `cargo check -p sciencekit_python`.
- [ ] 1.2 Implement the NumPy boundary (`PyReadonlyArray2` → `ArrayView2`, output arrays)
  with f32/f64 and dtype/layout tests.
- [ ] 1.3 Implement error mapping and the not-fitted exception; verify with pytest.

## 2. Family wrappers

- [ ] 2.1 Add preprocessing/impute wrappers (builders, fit, transform, accessors) and
  pytest coverage.
- [ ] 2.2 Add linear-model wrappers (fit, predict, predict_proba, coefficients) and pytest
  coverage.
- [ ] 2.3 Add neighbours/SVM wrappers and pytest coverage.
- [ ] 2.4 Add tree/ensemble wrappers and pytest coverage.
- [ ] 2.5 Add cluster/decomposition/outlier wrappers and pytest coverage.
- [ ] 2.6 Add pipeline, FeatureUnion, grid/halving search and metrics wrappers and pytest
  coverage.

## 3. GIL and zero-copy

- [ ] 3.1 Wrap compute calls in `allow_threads` and add a concurrency test.
- [ ] 3.2 Add an explicit zero-copy test (buffer identity) and a copy-fallback test for
  incompatible layouts.

## 4. Packaging, stubs and acceptance

- [ ] 4.1 Add `pyproject.toml`, maturin configuration and a wheel build/install smoke test.
- [ ] 4.2 Write type stubs for every public class/function and a stub-resolution test.
- [ ] 4.3 Add a pytest suite covering import, every family, dtype, pipelines, metrics,
  errors and concurrency.
- [ ] 4.4 Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace` and the pytest suite; fix every failure.
- [ ] 4.5 Update `CHANGELOG.md` and the companion mdBook chapter (including a
  `Development Explained` chapter on the NumPy boundary and GIL release); verify with
  `mdbook build docs`.
- [ ] 4.6 Independent review by a separate agent validating every scenario in
  `python-bindings`; address every finding and re-run the gates.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the code branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
