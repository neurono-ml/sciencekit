# sciencekit

[![Documentation](https://img.shields.io/badge/docs-mdBook-183e91?logo=readthedocs&logoColor=white)](https://neurono-ml.github.io/sciencekit/)
[![Stars](https://img.shields.io/github/stars/neurono-ml/sciencekit?style=flat&color=f9c440)](https://github.com/neurono-ml/sciencekit/stargazers)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Rust](https://img.shields.io/badge/rust-1.85-dea584?logo=rust)
![Status](https://img.shields.io/badge/status-phase%200%20—%20foundations-8b5cf6)

**All of scikit-learn. Natively in Rust.**

`sciencekit` reimplements every algorithm and utility of scikit-learn from scratch,
with extreme performance (SIMD, rayon, custom allocators), memory safety without a
garbage collector, zero-copy public APIs and native out-of-core support for datasets
larger than RAM — through streaming or memory-mapping.

## 📖 Documentation

The full project book — vision, algorithm catalog, architecture and roadmap — lives at:

**→ [neurono-ml.github.io/sciencekit](https://neurono-ml.github.io/sciencekit/)**

## Example of the API we are building

```rust
let model = SKKMeansClassifierBuilder::new()
    .number_of_clusters(8)
    .maximum_iterations(300)
    .execution_mode(SKExecutionMode::Automatic)
    .build()?;

model.fit(&training_data_view)?;
let predictions = model.predict(&test_data_view)?;
```

Builder pattern everywhere, zero-copy inputs (`ArrayView`/`CowArray`/sparse views),
compile-time-safe pipelines and an automatic execution engine that picks between
in-memory, async, streaming or memory-mapped modes for you.

## Status & plan

The repository is in **Phase 0 — foundations**. See the complete 8-phase plan on the
[Roadmap page](https://neurono-ml.github.io/sciencekit/roadmap.html) and the product truth in
[`docs/PRD.md`](docs/PRD.md).

## ⭐ Contribute

This is a large, well-scoped open source effort — every algorithm is an isolated opportunity.
Specs are written before code (OpenSpec), TDD is mandatory and PRs stay small.

- ⭐ [Star the repo](https://github.com/neurono-ml/sciencekit/stargazers) — it genuinely helps visibility
- 🐛 [Open an issue](https://github.com/neurono-ml/sciencekit/issues) or grab a `good first issue`
- 📚 Improve this site — see the [Documentation Guide](https://neurono-ml.github.io/sciencekit/documentation-guide.html)

## License

Apache-2.0. See [LICENSE](LICENSE).
