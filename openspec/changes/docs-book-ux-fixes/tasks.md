# Tasks

- [x] 1. Set up `docs/book-ux-fixes` code worktree (branch `docs/book-ux-fixes`) and `-openspec` worktree; open ADR issue in English.
- [x] 2. Typography: scale prose toward 12pt-equivalent and add persistent A-/reset/A+ control in the top bar (attribute + localStorage, bounded range, both themes).
- [x] 3. Buttons: convert top-bar icon buttons to ghost style; compact `sk-btn` sizing; verify hover/focus states in both themes.
- [x] 4. Sidebar accordion: single-open toggle with `aria-expanded`, active-chapter auto-expand, persisted state, no-JS fallback.
- [x] 5. Resize reflow: fluid `.content main` rule + sidebar min/max clamp; verify dragging reflows instead of only shrinking.
- [x] 6. Mermaid contrast: fixed `themeVariables` + `.mermaid svg` safety net; migrate pastel-only `style` directives to `--sk-*` fill+stroke+color triples; fixed neutral edges and opaque edge-label background.
- [x] 7. Mermaid overflow: container scroll + SVG max-width + flowchart wrap config; apply `<br/>` breaks to "The journey at a glance" and any other clipping diagram; document authoring limits in `documentation-guide.md`.
- [x] 8. Validate: `mdbook build docs`, visual check of affected pages in both themes at 3 font steps; independent review; update `CHANGELOG.md`; open PR with `Closes #<adr-issue>`.
