# Documentation Guide

Instructions for agents and contributors writing or updating this book. Follow them so every page keeps the same visual language and quality bar.

## Where things live

| Path | Purpose |
|---|---|
| `docs/book.toml` | mdBook configuration (site URL, theme, search) |
| `docs/src/` | Chapters in markdown (`SUMMARY.md` defines the table of contents) |
| `docs/skin/custom.css` | Design system — all visual components |
| `docs/skin/custom.js` | Sidebar brand, scroll-reveal animations, counters |
| `.github/workflows/deploy-documentation.yml` | Builds and publishes the book to GitHub Pages |

## Branch workflow (mandatory)

- The documentation lives on branch **`docs/documentations`** — never commit directly to it from `main`.
- Every docs change happens in a worktree: `temporary/worktrees/docs/<change-name>` on branch `docs/<change-name>`, merged into `docs/documentations` when done.
- Pushing to `docs/documentations` automatically rebuilds and deploys the site.

## Local preview

```bash
mdbook serve docs --open     # live reload at http://localhost:3000
mdbook build docs            # output in docs/book/ (never committed)
```

## Language rules

- **The book is written in English.** Everything durable in this repository is English.
- The PRD (`docs/PRD.md`) is the source of truth; translate its concepts faithfully, never invent scope.
- Code identifiers stay exactly as they are (`SKStandardScaler`, `sk_train_test_split`) — do not translate names.

## Component catalog

Use the ready-made components instead of raw styling. All accept standard HTML attributes.

### Hero + announcement pill

```html
<div class="sk-hero reveal">
  <div class="sk-announce">
    <span class="sk-announce__tag">New</span>
    Supporting text
  </div>
  <h1 class="sk-hero__title">Big statement title</h1>
  <p class="sk-hero__subtitle">Supporting sentence.</p>
</div>
```

### Cards grid

```html
<div class="sk-cards reveal">
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--sky">⚡</span>
    <div class="sk-card__title">Title</div>
    <p>Description.</p>
  </div>
</div>
```

Icon tints: `sk-icon--sky | violet | emerald | amber | rose | neutral`.
Add `sk-card--flow` for gradient-washed feature cards.

### Callout boxes

```html
<div class="sk-box sk-box--info">…</div>
```

Variants: `sk-box--info` (blue), `sk-box--tip` (violet), `sk-box--success` (green),
`sk-box--warning` (amber), `sk-box--danger` (red).

### Stats band

```html
<div class="sk-stats reveal">
  <div class="sk-stats__item"><div class="sk-stats__value js-count" data-target="39" data-suffix="+">39+</div><span class="sk-stats__label">algorithms mapped</span></div>
</div>
```

`js-count` animates the number when it scrolls into view.

### Pills (status badges)

```html
<span class="sk-pill sk-pill--emerald">Done</span>
```

Variants: `sk-pill--sky | violet | emerald | amber | rose | neutral`.

### Progress bar

```html
<div class="sk-progress"><div class="sk-progress__bar sk-progress__bar--amber" style="width:18%"></div></div>
<div class="sk-progress__meta"><span>label</span><span>18%</span></div>
```

Bar colors mirror pills: `--sky`, `--violet`, `--emerald`, `--amber`.

### Timeline (roadmap)

```html
<div class="sk-timeline">
  <div class="sk-timeline__item is-active reveal">
    <div class="sk-timeline__card">
      <div class="sk-timeline__head">
        <span class="sk-timeline__title">Phase N — Name</span>
        <span class="sk-pill sk-pill--amber">In progress</span>
      </div>
      <p>Summary.</p>
    </div>
  </div>
</div>
```

Item states: `is-done` (green dot), `is-active` (amber dot), default (violet dot).

### Donut chart & CSS bars

Copy the SVG structure from [Algorithms](./algorithms.md) and recompute segments:
circumference = `2π × 80 ≈ 502.65`; segment = `count / total × 502.65`; each circle's
`stroke-dashoffset` is the negative sum of all previous segments. Keep legend counts in sync.

For horizontal bars reuse `.sk-bars` rows as seen in [Plan & Roadmap](./roadmap.md).

### Terminal window

````html
<div class="sk-window">
  <div class="sk-window__bar">
    <span class="sk-window__dot sk-window__dot--r"></span>
    <span class="sk-window__dot sk-window__dot--y"></span>
    <span class="sk-window__dot sk-window__dot--g"></span>
    <span class="sk-window__title">bash</span>
  </div>

  ```bash
  cargo test --workspace
  ```

</div>
````

### Statement panel & closing CTA

```html
<div class="sk-statement reveal"><p class="sk-statement__text">One strong sentence.</p></div>

<div class="sk-cta reveal">
  <div class="sk-cta__title">Call to action</div>
  <p>Why act.</p>
  <a class="sk-btn sk-btn--star" href="#">⭐ Label</a>
</div>
```

### Buttons

```html
<a class="sk-btn sk-btn--primary" href="./page.html">Primary action</a>       <!-- dark -->
<a class="sk-btn sk-btn--secondary" href="#">Secondary</a>                    <!-- outline -->
<a class="sk-btn sk-btn--star" href="...">⭐ Star-flavored</a>                <!-- amber -->
```

## Content rules

1. Add new chapters to `SUMMARY.md` under the right part (Project / Engineering / Community).
2. Keep numbers honest: algorithm counts, phase progress and status pills must match the roadmap state.
3. Prefer components over custom inline CSS; extend `skin/custom.css` only for genuinely reusable pieces.
4. Add the `reveal` class to major blocks for scroll animation — sparingly.
5. Internal links use relative paths ending in `.html` (`./algorithms.html`).
6. Run `mdbook build docs` and fix every warning before pushing; `create-missing = false` means broken links fail the build.

## Checklist for a docs change

- [ ] Worktree created on `temporary/worktrees/docs/<name>`
- [ ] Content in English, consistent with the PRD
- [ ] New pages registered in `SUMMARY.md`
- [ ] Components used per catalog above
- [ ] `mdbook build docs` passes with zero warnings
- [ ] Merged into `docs/documentations` (Pages deploys automatically)
