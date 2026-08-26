# Contribute

<div class="sk-cta reveal">
  <div class="sk-cta__title">⭐ Star the project</div>
  <p>Stars are the simplest contribution: they boost visibility, attract contributors and tell us
  the community wants this. If sciencekit would help your work, a star takes two seconds.</p>
  <div class="sk-btn-row" style="justify-content: center;">
    <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit/stargazers">⭐ Star on GitHub</a>
    <a class="sk-btn sk-btn--secondary" href="https://github.com/neurono-ml/sciencekit/watchers">👀 Watch releases</a>
  </div>
</div>

## Where help is needed

The roadmap is large by design — every algorithm is an isolated, well-scoped opportunity:

| Area | What you would do | Good for |
|---|---|---|
| 🔢 Algorithms | Implement one estimator following naive → tests → performance → out-of-core | Rust + ML enthusiasts |
| 🧮 Math kernel | Pairwise distances, sparse ops, SIMD hot paths in `sciencekit_math` | Performance engineers |
| 🐍 Python bindings | PyO3 interfaces with maximum zero-copy, per completed algorithm | PyO3 experience |
| 🎮 GPU backends | OpenCL → CUDA → ROCm behind `SKComputeBackend` | GPU/HPC developers |
| 📚 Documentation | This book, API docs, examples from real tests | Writers & newcomers |
| 🐛 Testing & bugs | Reproduce issues, improve coverage of companion test modules | Everyone |

Open an issue or grab one labeled **`good first issue`** — specs are written before code, so you always start with clear requirements.

## How we work

1. **Spec first (OpenSpec):** each change gets a proposal, spec deltas and tasks on a dedicated `-openspec` branch.
2. **TDD:** tests are written first and must fail; implementation follows minimally.
3. **Small PRs:** every change alters the minimum necessary; work happens in git worktrees, never directly on `main`.
4. **Independent review:** an independent reviewer validates the spec was met before merge.
5. **Acceptance:** every implementation runs with lots and little data, under concurrency, exports its model and produces metrics.

Full engineering rules live in [AGENTS.md](https://github.com/neurono-ml/sciencekit/blob/main/AGENTS.md) and the product truth in the [PRD](https://github.com/neurono-ml/sciencekit/blob/main/docs/PRD.md).

## Getting started

```bash
git clone https://github.com/neurono-ml/sciencekit.git
cd sciencekit
cargo build && cargo test    # Rust 1.85 pinned via rust-toolchain.toml
```

<div class="sk-box sk-box--tip">
<strong>First contribution idea:</strong> documentation is a great entry point — this book itself
is built from markdown anyone can improve. See the <a href="./documentation-guide.html">Documentation Guide</a>.
</div>

## License

Apache-2.0 — including patent protection. By contributing you agree your contributions are licensed under it.

<div class="sk-btn-row">
  <a class="sk-btn sk-btn--primary" href="https://github.com/neurono-ml/sciencekit/issues/new">Open an issue →</a>
  <a class="sk-btn sk-btn--secondary" href="./roadmap.html">See the plan</a>
</div>
