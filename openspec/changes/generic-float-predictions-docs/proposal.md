## Why

The `development-explained` book (notably ch02 common-core-contracts) documents `predict -> Array1<f64>` and `SKTargetView::Continuous(f64)` as the contract. Once the code contract becomes generic over `SKFloat`, the book contradicts the implementation and teaches the lossy `f64`-only pattern.

## What Changes

- Rewrite ch02 sections `SKTargetView`, `SKPredictor`, `SKFeatureTransformer` family diagram, and `SKSupervisedScorer` examples to the generic `F: SKFloat` contract with the regressor/classifier split.
- Update every `Array1<f64>`-as-prediction snippet, Mermaid contract diagram (`predict features -> Array1 f64`), and KaTeX-free prose that claims dtype-independence of predictions.
- Add a short "Which float should I use?" guide (`f32` vs `f64`: memory, SIMD/GPU throughput, precision) plus a migration note for the breaking change.
- Verify with `mdbook build docs` and a rendered-page check; keep Mermaid `accTitle`/`accDescr`, KaTeX math, and English prose per the documentation guide.

## Capabilities

(none — documentation-only change; behavior is specified by `generic-float-predictions`.)

## Impact

- Affected docs: `docs/src/development-explained/ch02-common-core-contracts.md`, related tutorial/chapter snippets referencing `predict`/`SKTargetView`, Mermaid diagrams, `docs/book.toml` untouched.
- No Rust code, no API, no dependency changes. Merged into `docs/documentations` per repo workflow, not `main`.
