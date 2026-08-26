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

### Changed

- Translated `AGENTS.md` and `openspec/config.yaml` to English (international open-source project).
- `README.md` now links the official GitHub Pages site (`https://neurono-ml.github.io/sciencekit/`) at the top; it is also set as the repository homepage.
