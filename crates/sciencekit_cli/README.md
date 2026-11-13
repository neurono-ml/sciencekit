# sciencekit_cli

The command-line entry point over the [`sciencekit`](../..) umbrella crate.

The `sciencekit` binary exposes `fit`, `predict`, `transform`, `evaluate`,
`export`, `inspect` and `run`. `run` executes a versioned workflow file that
records the dataset, feature pipeline, estimator, optional grid search, metrics
and output artifacts, so an experiment can be reproduced from one document.

Configuration resolves as **flags > environment (`SCIENCEKIT_*`) > workflow
file > defaults**, and `--dry-run` prints the effective configuration without
writing artifacts. Failures map to stable exit codes: `0` success, `2` usage or
validation, `3` data or model I/O, `4` a capability this build lacks, `1`
runtime.

The full guide, including the workflow schema and the exit-code contract, lives
in the project book under *Development Explained*.
