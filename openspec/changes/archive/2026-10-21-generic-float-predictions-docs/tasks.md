## 1. Book rewrite

- [x] 1.1 Rewrite ch02 `SKTargetView` / `SKPredictor` / scorer sections to the generic `F: SKFloat` contract with the regressor/classifier split.
- [x] 1.2 Update the contract family Mermaid diagram (`predict -> Array1 f64`) plus `accTitle`/`accDescr`; replace float-label examples with `i64` labels.
- [x] 1.3 Add "Which float should I use?" guide (f32 vs f64) and breaking-change migration note, in English with KaTeX only for real formulas.

Note: the `Array1<f64>` Welford accumulators, normal-equation state, and SGD
`f64` state in the transformer/linear/streaming tutorials are internal
accumulation precision, not prediction outputs, and were deliberately left
untouched.

## 2. Verification

- [x] 2.1 Run `mdbook build docs`, open the rebuilt ch02 page, and confirm diagrams render and prose reads correctly.
- [ ] 2.2 Confirm the change merges into `docs/documentations`, not `main`.

Note on 2.1: the rebuilt ch02 page was verified in the built HTML
(`SKRegressorPredictor` present, `accTitle` present, no `Array1 f64`
remainder, new section present). Note on 2.2: merge happens via the doc PR.
