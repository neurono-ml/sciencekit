# docs-book-theme

## ADDED Requirements

### Typography scale and font-size control

- SHALL render body prose at a 12pt-equivalent scale (root `16px`, body text at or below the current size, headings scaled proportionally).
- SHALL provide a visible font-size control (decrease / reset / increase) in the top bar that adjusts prose size across a bounded range, persists the choice across page loads, and respects `prefers-reduced-motion` where animation is involved.
- SHALL keep KaTeX and Mermaid text legible at every size step (no clipping introduced by scaling).

### Top-bar and content buttons

- SHALL render top-bar icon buttons as borderless ghost controls (no boxed border, no heavy background) with a subtle hover state, in both light and dark themes.
- SHALL render content `sk-btn` elements at a compact size (shorter than the current `44px` height) with consistent radius and readable labels.

### Sidebar accordion and resize

- SHALL collapse sidebar sections as an accordion: expanding one section collapses its siblings, the active chapter's section is exposed on page load, state is keyboard-accessible (`aria-expanded`) and persists across page loads.
- SHALL make sidebar resize reflow content (content width adapts; no fixed `max-width` trap that only shrinks the reading area) with sensible min/max bounds.

### Mermaid contrast (fixed palette)

- SHALL render every Mermaid node with a fixed fill+text pair drawn from the `--sk-*` documentation palette (sky, emerald, violet, amber, rose, neutral), identical in light and dark themes, with text-to-fill contrast ratio >= 4.5:1.
- SHALL render edges and edge labels in a fixed neutral color legible on both page backgrounds, with an opaque edge-label background that does not inherit the page theme.
- SHALL NOT rely on Mermaid's per-theme defaults for node text, edge color, or edge-label background.

### Mermaid overflow

- SHALL NOT clip text inside flowchart nodes for the documented diagrams (including `tutorials/how-to-add-an-algorithm.md` "The journey at a glance"): long tokens wrap or break, the SVG scales to its container (`max-width: 100%`), and horizontal overflow scrolls within the diagram container rather than clipping.
- SHALL document authoring limits (maximum characters per line, mandatory `<br/>` breaks, short-label guidance) in `docs/src/documentation-guide.md`.
