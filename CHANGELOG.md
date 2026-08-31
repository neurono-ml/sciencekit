# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `README.md` with project overview, status badges and a validated architecture diagram.
- Repository workflow rules: keep-a-changelog policy, English-first durable artifacts, ADR issue per branch, Mermaid/SVG documentation standards.
- `CHANGELOG.md` seeded following Keep a Changelog 1.1.0.
- GitHub Actions workflow running the opencode agent on `/oc`/`/opencode` comments (issues and PR review comments).
- Versioned opencode project config registering the graphify plugin.
- `CI` workflow building and testing the Cargo workspace once it exists (green while Phase 0 is pending).
- Scratch-artifacts convention: temporary files go to `temporary/YYYY-MM-DD` (generic) or `temporary/YYYY-MM-DD/<change-name>` (change-related).
- Cargo workspace bootstrap (Phase 0.1): pinned Rust 1.85 toolchain, edition 2024, Apache-2.0 license, and CI gates (fmt, strict clippy, workspace tests + doctests, MSRV, examples).
- `sciencekit_common` sub-crate (Phase 0.2): the shared contract vocabulary — sealed `SKFloat` scalar bound, central `SKError` taxonomy, data/target views, label canonicalization, fit/scorer/execution/streaming traits.
- `sciencekit_math` sub-crate (Phase 0.3): higher-order `azip!`/`par_azip!` kernels, memory-layout helpers, zero-copy pairwise distances (Euclidean/Manhattan/Cosine) with a `wide` SIMD hot path, `sprs` CSR×dense and sparse×sparse products, and the `SKMathBackend` abstraction with `SKFaerBackend` (pure-Rust default) plus the opt-in `blas-backend` feature.
- `sciencekit_common` capability specs published (archive of `common-core-foundation`): scalar typing, error model, estimator contracts, data/target boundary, execution planning, scoring contracts, and streaming batches.
- `wave-plan-foundation` (planning): wave decomposition of PRD phases 0–7 into downstream changes (W0–W7), six foundational technical decisions locked (pure-Rust BLAS default → `faer 0.24.4`; OpenCL 3.0 ICD-agnostic GPU arrival order; `tdigest` online quantile sketch for `SKRobustScaler`; `faer-sparse` + `rsvd-faer` sparse SVD; nested-rayon thread management; `SKKNNImputer` moved to Wave 3), an academic-anchor catalogue, and reconciled dependency pins.
- `workspace-bootstrap` (planning): capability spec published and the change archived, capturing the Phase 0 workspace foundation, CI gates, and acceptance criteria.

### Changed

- Translated `AGENTS.md` and `openspec/config.yaml` to English (international open-source project).
- `README.md` now links the official GitHub Pages site (`https://neurono-ml.github.io/sciencekit/`) at the top; it is also set as the repository homepage.
- GitHub Actions opencode agent now uses the `opencode/big-pickle` model.
