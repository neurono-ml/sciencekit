# docs-book-theme Specification

## Purpose
Defines the readability and interaction contract of the mdBook documentation theme
(typography, controls, sidebar, Mermaid rendering) shared by the light and dark
palettes.

## Requirements

### Requirement: Typography scale and font-size control

The book SHALL render body prose at a 12pt-equivalent scale (root `16px`, body text at
or below that size, headings scaled proportionally) and SHALL provide a visible
font-size control (decrease / reset / increase) in the top bar that adjusts prose size
across a bounded range, persists the choice across page loads and respects
`prefers-reduced-motion` where animation is involved. KaTeX and Mermaid text SHALL stay
legible at every size step, with no clipping introduced by scaling.

#### Scenario: First visit uses the default size

- **WHEN** a reader opens any page without a previously stored preference
- **THEN** prose renders at the 12pt-equivalent default and the control shows the reset state

#### Scenario: Reader increases the size and reloads

- **WHEN** a reader presses the increase control and then reloads another page
- **THEN** the enlarged size applies again from the stored preference and no KaTeX or Mermaid text is clipped

### Requirement: Top-bar and content buttons

Top-bar icon buttons SHALL render as borderless ghost controls (no boxed border, no
heavy background) with a subtle hover state in both light and dark themes, and content
`sk-btn` elements SHALL render at a compact size (shorter than the original `44px`
height) with consistent radius and readable labels.

#### Scenario: Ghost hover state in both themes

- **WHEN** a reader hovers or focuses a top-bar icon button in the light theme and then in the dark theme
- **THEN** the button shows the same subtle hover treatment with no boxed border or heavy background

#### Scenario: Compact content buttons

- **WHEN** a page renders content `sk-btn` elements
- **THEN** each button is shorter than `44px`, keeps a consistent radius and a readable label

### Requirement: Sidebar accordion and resize

The sidebar SHALL behave as an accordion: expanding one section collapses its siblings,
the active chapter's section is exposed on page load, state is keyboard-accessible via
`aria-expanded` and persists across page loads. Sidebar resize SHALL reflow the content
width (no fixed `max-width` that only shrinks the reading area) within sensible
minimum and maximum bounds.

#### Scenario: Single-open sections

- **WHEN** a reader expands a sidebar section while another one is open
- **THEN** the previously open section collapses and both toggles expose the correct `aria-expanded` state

#### Scenario: Resizing reflows the content

- **WHEN** a reader drags the sidebar resize handle wider or narrower
- **THEN** the reading column reflows to the new available width instead of merely clipping, within the configured bounds

### Requirement: Mermaid contrast with a fixed palette

Every Mermaid node SHALL render with a fixed fill+text pair drawn from the `--sk-*`
documentation palette (sky, emerald, violet, amber, rose, neutral), identical in the
light and dark themes, with a text-to-fill contrast ratio of at least 4.5:1. Edges and
edge labels SHALL use a fixed neutral color legible on both page backgrounds, with an
opaque edge-label background that does not inherit the page theme.

#### Scenario: Diagrams render identically in both themes

- **WHEN** a reader switches between the light and dark themes on a page containing a Mermaid diagram
- **THEN** node fills, node text, edges and edge-label backgrounds keep the same fixed colors with contrast of at least 4.5:1

### Requirement: Mermaid colors independent of the page theme

The theme SHALL NOT rely on Mermaid's per-theme defaults for node text, edge color or
edge-label background; those three elements SHALL be pinned explicitly so a theme switch
never changes diagram legibility.

#### Scenario: Theme switch leaves diagram colors untouched

- **WHEN** a reader toggles the light/dark theme while a Mermaid diagram is on screen
- **THEN** node text, edge color and edge-label background do not change with the page theme

### Requirement: Mermaid overflow containment

Text inside flowchart nodes SHALL NOT be clipped for the documented diagrams (including
`tutorials/how-to-add-an-algorithm.md` "The journey at a glance"): long tokens wrap or
break, the SVG scales to its container (`max-width: 100%`) and horizontal overflow
scrolls within the diagram container rather than clipping. Authoring limits (maximum
characters per line, mandatory `<br/>` breaks, short-label guidance) SHALL be documented
in `docs/src/documentation-guide.md`.

#### Scenario: Long labels stay readable

- **WHEN** a diagram contains a node label longer than the node width
- **THEN** the label wraps or breaks inside the node and the diagram scrolls horizontally within its container instead of clipping

#### Scenario: Authoring limits are documented

- **WHEN** a documentation author follows `docs/src/documentation-guide.md`
- **THEN** the guide states the maximum characters per line, the mandatory `<br/>` breaks and the short-label guidance for Mermaid nodes
