# AGENTS.md

Instructions for agents working in this repository. Product source of truth: `docs/PRD.md`.

## Language

- **Everything durable is written in English.** Code, comments, documentation, OpenSpec specs, commits, PRs, issues, `README.md`, `CHANGELOG.md` — this is an international open-source project.
- Only the live conversation may follow the operator's language; anything persisted in the repository or on GitHub is always in English.

## Overview

- `sciencekit`: Rust ML library reimplementing all of scikit-learn, with extreme performance, zero-copy and native out-of-core.
- Freshly initialized repo: no Cargo workspace yet. Roadmap Phase 0 (PRD §12) creates the workspace with `sciencekit_*` sub-crates under `crates/`.
- Target toolchain: Rust 1.85, edition 2024. The PRD prevails over `docs/handoff.md`, which contains obsolete decisions (e.g., MSRV 1.64).
- `graphify-out/` holds the codebase knowledge graph — architecture/file-relationship questions must go through the graphify skill (`/graphify`) before manual exploration.

## Workflow (mandatory)

- **No direct commits on `main`.** Every change is implemented in **two coexisting git worktrees** that evolve in **parallel** and are merged into `main` via **separate PRs**:
  - Code: worktree `temporary/worktrees/<type>/<change-name>`, branch `<type>/<change-name>` (`type` ∈ `feat|bugfix|chore|docs`).
  - OpenSpec definitions: worktree `temporary/worktrees/<type>/<change-name>-openspec`, branch `<type>/<change-name>-openspec`. The change's `openspec/` files are committed on that associated branch.
  - `temporary/` is in `.gitignore` — worktrees are not versioned.
- **Agile, parallel development:** the code branch and the openspec branch are developed side by side, each merged via its **own PR**, in **any order**. The code is **not** required to wait for the openspec planning PR to merge, and vice versa — merging the plan first is not a precondition for implementing. The openspec PR reviews/specifies the change; the code PR implements it; both reference the same ADR issue (below). This keeps planning and implementation moving in parallel rather than blocking on a sequential merge.
- **Scratch artifacts:** all temporary files — experiments, validation screenshots, throwaway scripts, probes, scratch notes — are written under `temporary/YYYY-MM-DD` when generic, or `temporary/YYYY-MM-DD/<change-name>` when related to a change (`YYYY-MM-DD` = creation date). Never in tracked directories; nothing under `temporary/` is ever committed.
- **Issue per change (ADR):** when a change is opened, create a matching GitHub issue **in English**, written as an Architecture Decision Record:
  - Use the Y-statement template for simple decisions and the Nygard format for more complete ones (templates: https://adr.github.io/adr-templates/). Draft the issue text with the `architecture-decision-records` skill.
  - Reference the corresponding issue(s) in the PR descriptions (both the code PR and the openspec PR) using GitHub closing keywords (`Closes #<n>`, `Fixes #<n>`, `Resolves #<n>`) so issues are closed deterministically when a PR merges into `main`.
- **TDD mandatory:** every task uses the `tdd` skill — test first, confirm failure, minimal implementation, refactor.
- **Independent review:** at the end of each change, an independent agent reviews the result validating that the spec was effectively met, before the code PR.
- **Post-merge:** when the change's PRs are merged, the opencode agent on GitHub runs sync + archive of the change (both branches already landed via their own PRs).
- **Post-merge cleanup:** after a change's PRs are merged, remove its branches and worktree — delete the remote branch (merge with `--delete-branch` or `gh pr merge --delete-branch`), delete the local branch (`git branch -d <branch>`), and remove its worktree (`git worktree remove <path>`, then `git remote prune origin`). Branches whose PRs are still open (e.g. a draft) are kept. This keeps `temporary/worktrees/` and the branch list free of merged work.
- **Changelog:** every finished and merged change must update `CHANGELOG.md`, in English, following https://keepachangelog.com/en/1.1.0/. The snippet between releases in the changelog is used to describe the released version.

## Planning (OpenSpec)

- Use the `/opsx-propose`, `/opsx-apply`, `/opsx-archive` commands (`.opencode/commands/`); the `openspec` CLI is installed.
- Create changes in separate groups:
  1. algorithm implementation;
  2. changes for the different accelerators (GPU OpenCL/CUDA/ROCm, BLAS/SIMD, allocators, Python bindings);
  3. documentation changes.
- Python bindings and GPU backend come as associated/separate changes, only after the algorithm is ready and validated on CPU.

## Code conventions (from the PRD)

- Builder pattern mandatory; direct constructors private. Every builder exposes `execution_mode(SKExecutionMode::...)` defaulting to `Automatic`.
- **No abbreviations** in any Rust name, with a single exception: the project prefix `sk`/`SK`. Examples: `maximum_number_of_iterations`, not `max_iter`; `nearest_neighbors_count`, not `k`.
- **Mandatory prefix on public items** (full rule in PRD §3.4): structs and traits use `SK` + PascalCase (`SKEstimator`, `SKStandardScaler`); free-scope public functions (outside `impl`), variables and public modules use `sk_` + snake_case (`sk_train_test_split`). Methods — functions inside `impl` blocks of structs or traits — get no prefix. Crates always keep the full name (`sciencekit`, `sciencekit_*`).
- Zero-copy on public APIs: `ArrayView`/`CowArray`/sparse views (`sprs`), never `Array` by value.
- A `.rs` file over 200 lines becomes a standardized folder module (`mod.rs`, `builder.rs`, `core_implementation.rs`, `fitting_logic.rs`, `*_tests.rs`).
- Folder modules are pure dispatchers: a `mod.rs` under a folder module carries **no implementation logic** — it only declares submodules and re-exports their public items. Put every implementation in its own file under the folder (e.g. `builders/builder_state.rs`, `observability/run_operation.rs`); companion `*_tests.rs` modules stay beside the implementation.
- Tests live in companion `*_tests.rs` modules beside the implementation; mock data in `ndarray`/`sprs`. Never inline nor a global `tests/` directory.
- Iterative evolution per algorithm, no skipped steps: naive → tests → performance (SIMD/rayon/layout) → streaming/out-of-core.
- Acceptance of every implementation (PRD §8.7): runs with lots and little data, under concurrency, exports the model and produces metrics.
- CPU never blocks async threads (rayon for compute, Tokio for I/O); iteration via `.map()`/`azip!()`/`par_azip!()`, never manual index loops.

## Documentation

- Main documentation branch: `docs/documentations`. Docs worktrees are merged into it (not into `main`).
- That branch hosts an mdBook compatible with GitHub Pages containing: API, usage examples and a description of every function.
- Unit tests of API functions and e2e tests are used as examples in the book.
- **Diagrams:** always Mermaid or SVG — colorful, explanatory, with vivid clear colors harmonized with the mdBook theme. Render them with the drawing/rendering tools and make sure they display without errors before committing. ASCII-art diagrams are forbidden in any documentation (ASCII sketches are fine in conversation).
- **Book location:** configuration in `docs/book.toml`; chapters in `docs/src/`; custom skin (CSS/JS) in `docs/skin/`.
- **Book language: English.** Write new chapters and edits in English; translate PRD concepts faithfully.
- **Documentation agents:** instructions and component catalog in `docs/src/documentation-guide.md`.
- **Publishing:** the GitHub Actions workflow (`.github/workflows/deploy-documentation.yml`) builds and deploys the book to GitHub Pages on every push to `docs/documentations`. Run `mdbook build docs` locally before pushing.
- **README:** keep `README.md` up to date, small, pointing to the documentation, with the expected badges for a Rust project hosted on GitHub (CI status, crates.io, docs.rs, Rust version).

## Agent skill

- Branch `chore/skill`: skill following https://agentskills.io/specification, compatible with the library version on `main`, exposed as new capabilities are developed.
- Structured in files separated by capability, with supporting scripts and assets, so that simple or advanced AI agents can use the library.

## Releases

- Each library version gets a Git tag, is published to crates.io and associated with a GitHub Release.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
