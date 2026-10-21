# Design: docs-book-ux-fixes

## Context

The book theme lives in three files loaded via `docs/book.toml` `additional-css`/`additional-js`:
`docs/skin/custom.css` (~1150 lines), `docs/skin/custom.js` (theme-list cleanup, brand, reveal),
`docs/skin/mermaid-init.js` (renders `language-mermaid` fences with hardcoded `theme: 'dark'`).
No `.mermaid` CSS exists. Sidebar and resize behavior is inherited from mdBook 0.5.2 defaults.

## Decisions

1. **Typography**: keep `:root 16px` (= 12pt); scale body prose down via `.content p/li` size and proportional heading steps; implement font-size control as three small ghost buttons (A-, A, A+) in the menu bar driven by a `data-sk-font-scale` attribute on `<html>` with `localStorage` persistence. Bounded range (e.g. 0.875–1.125) so KaTeX/Mermaid do not break.
2. **Buttons**: top-bar `.icon-button` becomes borderless (`border: none`, transparent background, muted color, soft hover wash). Content `a.sk-btn` shrinks to `36px` height, `10px` radius, `0.85rem` font. Both variants keyed off existing `--sk-*` tokens; no new palette.
3. **Sidebar accordion**: delegate to `custom.js`; wrap each `ol.section` under a toggle button with `aria-expanded`; single-open policy (opening one closes siblings at the same level); on load, expand the section containing `.active` and restore persisted state. Pure progressive enhancement — no-JS still shows all sections.
4. **Resize reflow**: keep mdBook's `#sidebar-resize-handle`; change `.content main` from fixed centered `max-width: 62rem` to a fluid rule (`min(62rem, 100%)` + flex growth) so dragging the sidebar reflows text; add min/max clamp on sidebar width.
5. **Mermaid fixed palette**: stop relying on Mermaid themes. Pass fixed `themeVariables` (node border/text, edge color, `edgeLabelBackground` opaque) plus a `.mermaid svg` CSS safety net per `.node` class. Map the six `--sk-*` pairs:
   sky `#e0f2fe/#0c4a6e`, emerald `#dcfce7/#14532d`, violet `#ede9fe/#3b0764`,
   amber `#fef3c7/#78350f`, rose `#ffe4e6/#7f1d1d`, neutral `#f4f4f5/#18181b`.
   Migrate all `style X fill:...`-only directives in `docs/src/**` to `fill+stroke+color` triples.
6. **Overflow**: `.mermaid { overflow-x: auto }`, `.mermaid svg { max-width: 100%; height: auto }`, Mermaid `flowchart: { htmlLabels: true, useMaxWidth: true }`, `wrap: true`; authoring rule (<= ~24 chars/line, `<br/>` breaks) added to `documentation-guide.md`; split only if a diagram still clips after CSS (ch05 nested subgraphs are candidates, not mandatory).

## Alternatives considered

- Per-theme Mermaid palettes (light pastel vs dark vibrant): rejected per operator decision for fixed colors (simpler, single test pass).
- Third-party mdBook preprocessors for accordion/resize: rejected; mdBook 0.5 plugin surface is thin and upstream JS already owns the sidebar.

## Verification

- `mdbook build docs` succeeds; open `tutorials/how-to-add-an-algorithm.html` ("The journey at a glance"), `architecture.html`, one `ch*` page in both themes; check font control, accordion, resize, and Mermaid contrast/overflow at 3 font steps.
