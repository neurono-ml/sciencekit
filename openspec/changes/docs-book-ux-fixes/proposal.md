## Why

The mdBook documentation ships a custom theme (`docs/skin/custom.css`, `custom.js`, `mermaid-init.js`) with six user-reported defects: oversized body typography with no resize control, visually heavy top-bar buttons, a sidebar that never collapses sections, a resize handle that only shrinks content, illegible Mermaid text in the dark theme, and clipped node text in flowcharts.

## What Changes

- Reduce base body typography toward a 12pt-equivalent scale and add a persistent, accessible font-size control.
- Restyle top-bar icon buttons (ghost style) and normalize content `sk-btn` sizing.
- Add sidebar accordion behavior (single-open, persists state, exposes active chapter).
- Fix sidebar/content resize so content reflows instead of only shrinking.
- Standardize Mermaid diagrams on fixed `--sk-*` palette pairs (fill + text) with a neutral fixed edge color and edge-label background; migrate pastel-only `style` directives.
- Fix Mermaid node-text clipping (container overflow, SVG max-width, authoring limits on label length).

## Capabilities

### New Capabilities

- `docs-book-theme`: user-visible documentation theme behavior — typography scale + font-size control, buttons, sidebar accordion + resize, Mermaid contrast and overflow handling.

### Modified Capabilities

- None.

## Impact

- Affected: `docs/skin/custom.css`, `docs/skin/custom.js`, `docs/skin/mermaid-init.js`, Mermaid fences in `docs/src/**/*.md`, `docs/src/documentation-guide.md` (authoring rules).
- No Rust crates, no public API, no runtime behavior outside the built book.
- Verification via `mdbook build docs` plus visual check of affected pages in both themes.
