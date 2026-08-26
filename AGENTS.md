# AGENTS.md

Instructions for agents working in this repository. Product source of truth: `docs/PRD.md`.

## Overview

- `sciencekit`: an ML library in Rust reimplementing the whole of scikit-learn, with extreme performance, zero-copy and native out-of-core.
- Freshly initialized repository: there is no Cargo workspace yet. Phase 0 of the roadmap (PRD §12) creates the workspace with `sciencekit_*` sub-crates under `crates/`.
- Target toolchain: Rust 1.85, edition 2024. The PRD prevails over `docs/handoff.md`, which contains obsolete decisions (e.g.: MSRV 1.64).
- `graphify-out/` holds the codebase knowledge graph — questions about architecture/relationships between files must go through the graphify skill (`/graphify`) before manual exploration.

## Workflow (mandatory)

- **No direct commits to `main`.** Every change is implemented in git worktrees:
  - Code: worktree `temporary/worktrees/<type>/<change-name>`, branch `<type>/<change-name>` (`type` ∈ `feat|bugfix|chore|docs`).
  - OpenSpec definitions: worktree `temporary/worktrees/<type>/<change-name>-openspec`, branch `<type>/<change-name>-openspec`. The change's `openspec/` files are committed on that associated branch.
  - `temporary/` is in `.gitignore` — worktrees are not versioned.
- **Mandatory TDD:** every task uses the `tdd` skill — test first, confirm failure, minimal implementation, refactoring.
- **Independent review:** at the end of each change, an independent agent reviews the result validating that the spec was actually met, before the PR.
- **Post-merge:** when a PR is merged, the opencode agent on GitHub merges the corresponding `-openspec` branch into `main` and runs change sync + archive.

## Planning (OpenSpec)

- Use the `/opsx-propose`, `/opsx-apply`, `/opsx-archive` commands (`.opencode/commands/`); `openspec` CLI installed.
- Create changes in separate groups:
  1. algorithm implementation;
  2. changes for the different accelerators (GPU OpenCL/CUDA/ROCm, BLAS/SIMD, allocators, Python bindings);
  3. documentation changes.
- Python bindings and GPU backend enter as associated/separate changes, only after the algorithm is ready and validated on CPU.

## Code conventions (from the PRD)

- Mandatory builder pattern; direct constructors private. Every builder exposes `execution_mode(SKExecutionMode::...)` with default `Automatic`.
- **No abbreviations** in any Rust name, with a single exception: the project prefix `sk`/`SK`. Examples: `maximum_number_of_iterations`, not `max_iter`; `nearest_neighbors_count`, not `k`.
- **Mandatory prefix on public items** (full rule in PRD §3.4): structs and traits use `SK` + PascalCase (`SKEstimator`, `SKStandardScaler`); public free-standing functions (outside `impl`), variables and public modules use `sk_` + snake_case (`sk_train_test_split`). Methods — functions inside `impl` blocks of structs or traits — receive no prefix. Crates always keep the full name (`sciencekit`, `sciencekit_*`).
- Zero-copy on public APIs: `ArrayView`/`CowArray`/sparse views (`sprs`), never `Array` by value.
- A `.rs` file over 200 lines becomes a standardized folder module (`mod.rs`, `builder.rs`, `core_implementation.rs`, `fitting_logic.rs`, `*_tests.rs`).
- Tests in companion `*_tests.rs` modules next to the implementation; mock data built with `ndarray`/`sprs`. Never inline nor in a global `tests/` directory.
- Iterative evolution per algorithm, never skipping stages: naive → tests → performance (SIMD/rayon/layout) → streaming/out-of-core.
- Acceptance of every implementation (PRD §8.7): runs with lots and little data, under concurrency, exports the model and produces metrics.
- CPU never blocks async threads (rayon for computation, Tokio for I/O); iteration via `.map()`/`azip!()`/`par_azip!()`, never manual index loops.

## Documentation

- Main documentation branch: `docs/documentations`. Docs worktrees are merged into it (not into `main`).
- That branch hosts an mdBook compatible with GitHub Pages containing: API docs, usage examples and a description of each function.
- Unit tests of API functions and e2e tests are used as examples in the book.
- Book location: configuration in `docs/book.toml`; chapters in `docs/src/`; custom skin (CSS/JS) in `docs/skin/`.
- **Book language: English.** Write new chapters and edits in English; translate PRD concepts faithfully.
- Instructions for documentation agents: `docs/src/documentation-guide.md`.
- Publishing: the GitHub Actions workflow (`.github/workflows/deploy-documentation.yml`) builds and deploys the book to GitHub Pages on every push to `docs/documentations`. Run `mdbook build docs` locally before pushing.

## Agent skill

- Branch `chore/skill`: skill following https://agentskills.io/specification, compatible with the library version on `main`, exposed as new capabilities are developed.
- Structured in files split by capability, with supporting scripts and assets, so simple or advanced AI agents can use the library.

## Releases

- Each library version gets a Git tag, is published to crates.io and associated with a GitHub Release.
