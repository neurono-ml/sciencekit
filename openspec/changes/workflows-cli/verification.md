# Independent review — workflows CLI

Task 4.4: an independent agent reviewed the change, ran the compiled binary
against adversarial inputs and fixed every defect found. Reviewed artifacts:
the code worktree (PR #405), the `-openspec` worktree (PR #261) and the docs
worktree (PR #406).

## Method

- Built the real `sciencekit` binary from the code worktree and invoked it for
  every failure mode below; exit codes and stderr were captured from the process,
  not asserted in-process.
- Proved precedence with one value (`alpha`) set by a flag, the environment, a
  workflow file and the default simultaneously.
- Ran one workflow twice into two output directories and compared every artifact
  byte for byte.
- Reviewed the public naming, module layout and test placement against
  `AGENTS.md`.
- Fixed each defect test-first: a failing process-level test first, then the
  minimal implementation, then a `fix(cli): ...` commit.

## Verdict matrix

### Error handling

| Case | Exit | Message quality |
|---|---|---|
| Workflow `version: 2` | 2 | Names `workflow.version`, the found version and the supported one — actionable |
| Unknown key `estimator.typo` | 2 | Names `workflow.estimator.typo` and lists the accepted keys — actionable |
| Workflow names `dataset.path` that does not exist | 3 | Names the file and the OS error — actionable |
| Workflow file that does not exist | 3 | Names the file and the OS error — actionable |
| CSV row wider/narrower than its header | 3 | Names the file, line and expected/found record counts — actionable |
| Model file with the wrong format | 3 | Names the file and the container byte-count mismatch — actionable |
| Parquet dataset without the feature | 4 | Names the `parquet` Cargo feature — actionable |
| ONNX export without the feature | 4 | Names the `onnx` Cargo feature — actionable |
| YAML workflow without the feature | 4 | Names the `yaml` Cargo feature — actionable |
| `--dry-run` with an invalid format, estimator, metric or backend | 2 / 4 | Names the value and the accepted set (fixed; previously exited 0) |
| Workflow output directory missing | 2 | Names `configuration.predictions` and the directory (fixed; previously left a partial model) |
| Unknown flag / missing subcommand | 2 | Clap usage text with a suggestion — actionable |

Every failure map matches the documented contract: `0` success, `2`
usage/validation, `3` data/model I/O, `4` missing build capability, `1`
unexpected runtime.

### Precedence

One value set by all four layers resolved `4.0` (flag), `3.0` (environment),
`2.0` (workflow file) and `1.0` (default). The artifact format followed the same
rule after the fix: the workflow file's `output.format` beat the `.csv`
extension for `predict` and `transform`, and an empty `SCIENCEKIT_OUTPUT_FORMAT`
no longer counted as a choice.

### Reproducibility

Before the fix the model artifact differed between two runs at byte 38: the
single-file Safetensors encoder converted its ordered metadata into a `HashMap`,
so key order followed a per-process random seed. Predictions already matched.
After the fix the model and predictions files are byte-identical across runs and
across output directories; no timestamp is embedded.

### Schema honesty

An estimator name the schema does not implement (`gradient_boosting`) exits `2`
with the accepted list — it is a vocabulary error, not a build capability. A
format the build was not compiled with (Parquet, ONNX, YAML, `backend:
opencl`) exits `4` naming the feature; nothing falls back silently.

### Conventions

- `SK`/`sk_` prefixes, no abbreviations, builder-only construction and
  English-only output hold on every reviewed file.
- Every folder `mod.rs` is a pure dispatcher; companion `*_tests.rs` modules sit
  at the end of their `mod.rs`.
- One justified deviation: the dataset adapter materializes a row-major matrix
  from columnar Arrow batches with indexed row loops. The Arrow column API takes
  a row index, so iterator combinators cannot remove the index; the loop is kept
  and noted rather than obscured.
- Process-level CLI suites live under `crates/sciencekit_cli/tests/`, the only
  place Cargo compiles a second binary per file; task 4.1/4.4 require
  process-level tests, so this is the documented exception to the
  companion-module rule.

## Defects fixed

1. `1a99275` — model artifacts were not byte-reproducible (metadata `HashMap`
   ordering). Fixed in the Safetensors single-file encoder; pinned by an interop
   unit test and a CLI process-level reproducibility test.
2. `648694b` — `--dry-run` accepted values execution rejects (dataset, output
   and model formats, estimator, metric, backend). Validation now runs before
   the dry-run early return, and command-specific formats/extensions are checked
   before it too.
3. `16d9846` — the workflow file's `output.format` was ignored by `predict` and
   `transform`, and an empty environment variable overrode extension inference.
   The resolved configuration now records whether a layer chose the format.
4. `180c908` — a `run` whose later output failed left a partial model artifact.
   Outputs are preflighted before the first write.

## Gates

- `cargo fmt --all --check` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean (finished with no
  warnings).
- `cargo test --workspace` — exit 0, 53 suites passed. One earlier run tripped a
  pre-existing flake in `sciencekit_decomposition`
  (`truncated_representation_wide_sparse_matrix_fits_without_densification`):
  its tracing-capture guard is global, so a parallel test can inject a warning
  event. The test passes in isolation and the full suite passes on rerun; the
  crate is untouched by this change and the flake is reported here rather than
  fixed.
- `mdbook build docs` (docs worktree) — clean.
- `openspec validate workflows-cli --strict` — valid.

Verdict: the change meets the `workflows-cli` spec; task 4.4 is complete.
