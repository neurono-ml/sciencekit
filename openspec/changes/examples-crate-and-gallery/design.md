# Design

## Context

sciencekit is a Rust reimplementation of scikit-learn with crates for the foundations
(`sciencekit_common`, `sciencekit_math`), preprocessing, imputation, linear models and
metrics. It has **no** example corpus, **no** dataset surface, **no** gallery
documentation and **no** didactic book. The scikit-learn gallery it must mirror has 34
sections and 281 example pages.

Four facts shape this design:

1. **The corpus is downstream of the algorithms.** Only the algorithm families of
   PRD phases 1–2 exist today; the catalog therefore has to carry a *blocked* status and
   name the missing sciencekit capability, so the corpus grows with the roadmap instead
   of being frozen at a partial snapshot.
2. **The bindings do not exist.** Python, Go, Julia, JVM and Lua bindings are planned
   changes with no implementation. Multi-language code cannot be written against a
   frozen API yet.
3. **`scienceview` is not ready.** The repository exists but publishes no crate version,
   exposes no data API, and ships two trivial examples. Any Rust plotting code written
   today would be guessing.
4. **The repository conventions are strict.** Folder modules with pure dispatchers,
   200-line files, companion `*_tests.rs`, no abbreviations, `SK`/`sk_` prefixes, and the
   `Development Explained` book in lockstep with the code.

See `proposal.md` for the motivation and `specs/` for the behavioral contract.

## Goals / Non-Goals

**Goals:**

- A crate whose structure makes porting one gallery example a mechanical, reviewable
  task with a written recipe.
- A catalog that is the single source of truth for parity, blocking and language
  coverage.
- A documentation topology where the gallery (browse by algorithm) and the book (learn
  the concept) reinforce each other.
- Zero mandatory dependency on anything unready.

**Non-Goals:**

- Porting all 281 gallery entries in this change. This change delivers the
  *infrastructure*, the *catalog*, the *dataset surface* and the first wave of examples
  covering the algorithms that exist today.
- Implementing the multi-language code tabs. Recorded as a future change (Decision 8).
- Adding `scienceview` as a real dependency. The integration point is designed and
  isolated, but the feature stays off.
- Building a `sciencekit_datasets` crate. The dataset surface is owned by the examples
  crate (Decision 4).

## Decisions

### Decision 1 — The example lives in a library module, the cargo target is a shim

Each example's logic is a folder module inside the crate's library; each cargo example
target is a five-line shim. The crate is unpublished and entirely feature-gated.

```
crates/sciencekit_examples/                 publish = false   ← never on crates.io
├── Cargo.toml
│   ├── [features] default = []  examples = []     ← non-default gate
│   └── [[example]] × N   (required-features = ["examples"])
├── src/
│   ├── lib.rs                    # feature-gated module declarations
│   ├── examples/                 # the per-example logic (library, testable)
│   │   ├── mod.rs                # pure dispatcher
│   │   └── linear_model/
│   │       ├── mod.rs            # pure dispatcher
│   │       └── ordinary_least_squares_and_ridge_regression/
│   │           ├── mod.rs        # pure dispatcher + public entry point
│   │           ├── example_header.rs    # the gallery header (title, dataset, knobs)
│   │           ├── dataset_generation.rs
│   │           ├── model_configuration.rs
│   │           ├── metrics_report.rs
│   │           ├── plotting.rs         # the only file that calls the plotting API
│   │           └── ordinary_least_squares_and_ridge_regression_tests.rs
│   ├── datasets/                 # the dataset surface (Decision 4)
│   ├── catalog/                  # the catalog records + the loader (Decision 2)
│   └── plotting/                 # the plotting abstraction (Decision 5)
└── examples/
    └── ordinary_least_squares_and_ridge_regression/
        └── main.rs               # the cargo example target: a shim
```

The run command is uniform and lives in `.cargo/config.toml`:

```toml
[alias]
example = "run -p sciencekit_examples --features examples --example"
```

```
$ cargo example ordinary_least_squares_and_ridge_regression
```

*Rationale.* The library half is where the testable behavior lives; the target half
exists only so the example runs, which is what a reader will type. Three Cargo
behaviours, verified empirically against a probe crate, shape this decision:

1. **`cargo test` compiles example targets** — a broken example fails the workspace test.
   Hence the `required-features` gate: with the feature off, `cargo test --workspace`
   compiles nothing of the corpus, and the repository's CI gate stays fast.
2. **Example targets never run their own `#[test]`s** — hence the logic lives in the
   library, where the companion `*_tests.rs` modules actually run.
3. **`exclude` in the manifest makes Cargo drop the target** ("ignoring example … as it
   is not included in the published package"), so a published manifest carries no
   example target. In this shape that is belt and braces: the crate is `publish = false`
   in the first place.

*Alternatives considered.* (a) Binaries in `src/bin/` — rejected: `cargo run --bin` is
not the idiomatic way to ship a learning corpus, and the binaries would be compiled by
`cargo test` unless gated the same way. (b) The corpus inside the published umbrella
package — rejected: a published crate cannot depend on an unpublished one, so the
support code (datasets, catalog, plotting) would have to ship in the `sciencekit`
tarball, and the tests would need one `[[test]]` entry per example pointing into
`examples/`. (c) Everything in `main.rs` per target — rejected: it violates the
repository's module conventions and makes the examples untestable and unimportable.
(d) One crate per family — rejected: 34 crates for a corpus of examples is operational
noise.

### Decision 2 — The catalog is the source of truth, and it is machine-readable

The catalog is a `YAML`/`JSON` document loaded at compile time (`include_str!`) and
validated by the crate's tests. It is *not* generated from the code, because the code is
the thing that lags behind: the catalog must be able to describe examples that do not
exist yet.

Record schema (one per gallery entry, 281 records):

| Field | Meaning |
|---|---|
| `id` | The example id; equals the module name and the example target name |
| `family` | The gallery family, e.g. `linear_model`, `cluster`, `ensemble` |
| `title` | The gallery title, kept close to scikit-learn's wording |
| `sklearn_source` | Path in the scikit-learn repository, e.g. `examples/linear_model/plot_ols_ridge.py` |
| `sklearn_url` | The rendered gallery URL of that example |
| `dataset_source` | The `sklearn.datasets` symbol used, e.g. `datasets.load_diabetes`, `datasets.make_blobs` |
| `sciencekit_crates` | The sciencekit crates the port depends on |
| `estimators` | The sciencekit estimators involved, with their scikit-learn counterparts |
| `knobs` | The configuration parameters exercised and the number of settings compared |
| `languages` | Languages the example is published in, in tab order |
| `status` | `ported`, `partial`, `blocked`, `omitted` |
| `blocked_by` | The missing sciencekit capability, when `status` is `blocked` |
| `omission_reason` | Why the entry is not ported, when `status` is `omitted` |

The statuses encode the roadmap: `blocked_by` names the capability (e.g.
`sciencekit_cluster`), so a reader can compute exactly how much of the gallery is
reachable today.

**The catalog is bootstrapped, not typed.** A probe against the scikit-learn repository
tree (282 example source files under `examples/`, of which 280 are gallery entries,
plus `developing_estimators/sklearn_is_fitted.py` and
`neighbors/approximate_nearest_neighbors.py`, which the gallery links without listing)
shows that the mechanical part of a record is extractable: the gallery title is the RST
underline heading of the module docstring, the dataset source is the
`from sklearn.datasets import …` statement (38 files use a loader, 74 use a `make_*`
generator, 3 use neither), and the estimators are the `from sklearn.<module> import
<Name>` statements (277 of 282 files). The bootstrap is therefore a script that walks
the scikit-learn tree and emits the catalog skeleton; a human then adds the fields that
cannot be derived mechanically — the sciencekit counterpart of each estimator, the knob
spectrum, the status and the languages. The catalog's `sklearn_source` field keeps the
provenance of the bootstrap so a stale record can be re-derived.

The estimator frequency across the gallery also sizes the effort: `model_selection`
appears in 186 files, `preprocessing` in 148, `metrics` in 142, `linear_model` in 135,
`pipeline` in 82, `ensemble` in 75, `svm` in 52, `decomposition` in 45, `cluster` in 45,
`gaussian_process` in 40, `neighbors` in 29, `compose` in 27. The corpus is not 281
independent algorithms but roughly 30 estimators recombined — which is why the wave plan
follows the *algorithm* roadmap and not the gallery order.

*Alternatives considered.* (a) Derive the catalog from the filesystem — rejected: it
cannot represent blocked or omitted entries, which is precisely the information a
parity report needs. (b) Write the catalog by hand in Markdown — rejected: not
machine-checkable, and it would drift from the code.

### Decision 3 — Every ported example traces to its scikit-learn source, and the porting recipe is written down

This is the part that makes the corpus reproducible by a *different* agent than the one
that wrote it.

**Where the scikit-learn source lives.** The gallery is generated by `sphinx-gallery`
from the `examples/` tree of the [scikit-learn repository](https://github.com/scikit-learn/scikit-learn):

```
scikit-learn/examples/
├── linear_model/plot_ols_ridge.py          ← one file per gallery entry
├── cluster/plot_kmeans_digits.py
├── ensemble/plot_forest_iris.py
└── … 280 gallery files across 34 family directories
            (+ developing_estimators/sklearn_is_fitted.py,
               neighbors/approximate_nearest_neighbors.py)
```

The datasets come from `sklearn/datasets/` (`_base.py` for the loaders,
`_samples_generator.py` for the `make_*` generators). The gallery page URL is
`https://scikit-learn.org/stable/auto_examples/<family>/plot_<name>.html`.

**The porting recipe** — the instructions an implementing agent follows for one entry:

1. **Read the source.** Open the file named in `sklearn_source`. Extract the docstring
   header (title, description, dataset), the estimators used, the configuration
   parameters varied, and the figures produced.
2. **Identify the dataset source.** From `dataset_source`, decide whether it is a
   `make_*` generator or a bundled dataset. Generators are reimplemented in the crate's
   `datasets` module with the scikit-learn semantics; bundled datasets are embedded
   with their real shapes and names. If neither exists yet, add it to the dataset module
   in the same change.
3. **Map the algorithms.** Each scikit-learn estimator maps to its sciencekit
   counterpart — `sklearn.linear_model.Ridge` → `SKRidge` + `SKRidgeBuilder`,
   `sklearn.preprocessing.StandardScaler` → `SKStandardScaler` + `SKStandardScalerBuilder`.
   The mapping is recorded in the catalog's `estimators` field. If the counterpart does
   not exist yet, the record's status is `blocked` and names the capability — the example
   is *not* faked.
4. **Choose the configuration spectrum.** Do not port scikit-learn's default
   parameters verbatim: the example must compare at least two settings of the knob that
   matters (regularization strength, number of clusters, tree depth, learning rate) so
   the cause-and-effect is visible. The spectrum is recorded in `knobs`.
5. **Implement the module.** Create the folder module with the file split listed in
   Decision 1. The module's header doc comment carries the gallery title, the dataset
   and its origin, the knob spectrum, and the run command — the same information
   scikit-learn's gallery docstring carries, so the generated page is a faithful port.
6. **Write the companion test.** The test asserts the numbers the gallery page quotes.
   The page is generated from what the test asserts, so the prose can never drift from
   the behavior.
7. **Register in the catalog**, flip the status, and add the cargo example target.

**Code style.** The example modules obey the repository conventions (builder pattern,
no abbreviations, `SK`/`sk_` prefixes, zero-copy views, no manual index loops — use
`map`/`azip!`/`par_azip!`), and additionally:

- The example module never calls the plotting API directly; it goes through the
  crate's plotting abstraction (Decision 5).
- The example module never hardcodes a seed; it declares the seed as a named constant
  next to the knob spectrum, so the page can quote it.
- Numeric output is printed through the crate's shared reporting helpers, so the
  gallery page and the terminal output agree.
- No example downloads anything (Decision 4).

**Worked example.** `examples/linear_model/plot_ols_ridge.py` → id
`ordinary_least_squares_and_ridge_regression`:

| scikit-learn | sciencekit |
|---|---|
| `datasets.load_diabetes()` | `datasets::load_diabetes()` (crate-local, embedded) |
| `linear_model.LinearRegression()` | `SKLinearRegression::builder()` |
| `linear_model.Ridge(alpha=…)` | `SKRidge::builder().alpha(…)` |
| `metrics.mean_squared_error` | `sciencekit_metrics` regression scorer |
| `plt.plot(…)` | `plotting::line_series(…)` through the abstraction |

The knob spectrum is `alpha ∈ {0.0, 1.0, 10.0, 100.0}` — the classic ridge path — with
the page explaining how the penalty trades bias for variance.

*Alternatives considered.* (a) Port from the scikit-learn *documentation* rather than
its source — rejected: the rendered page hides the data flow. (b) Copy scikit-learn's
example files verbatim and translate — rejected: scikit-learn's style (globals,
`plt` calls, `numpy` arrays) violates this repository's conventions.

### Decision 4 — The dataset surface is owned by the examples crate

`crates/sciencekit_examples/src/datasets/` provides the `sklearn.datasets` surface the
gallery needs:

- **Generators** mirroring `sklearn.datasets._samples_generator`: `make_classification`,
  `make_blobs`, `make_circles`, `make_moons`, `make_regression`, `make_sparse_spd_matrix`,
  `make_swiss_roll`, `make_multilabel_classification`, `make_low_rank_matrix`,
  `make_sparse_uncorrelated`. Each takes an explicit seed, honors the scikit-learn
  parameter semantics (`n_samples`, `n_features`, `noise`, `factor`, `shuffle`, …) and
  can emit a sparse design where scikit-learn does.
- **Embedded datasets**: iris, digits, wine, diabetes, breast cancer, linnerud — the
  small ones that fit in the repository. Their real shapes, feature names and target
  names are preserved.
- **No network.** Large download-based datasets (`fetch_olivetti_faces`,
  `fetch_20newsgroups`, `fetch_lfw_people`) are out of scope; the examples that need
  them are catalogued as `blocked` or `omitted` with the reason recorded.

*Rationale.* The gallery's most-used datasets are synthetic, and the synthetic
generators are also the didactic substrate of the book (the bias–variance and
clustering-assumption chapters need controllable ground truth). A dataset crate was
considered and rejected for now: it would need to become a first-class sciencekit crate
with its own public API, its own spec and its own release cadence, and the examples only
need a narrow, internal surface. Promoting it later is a mechanical move.

*Alternatives considered.* (a) Vendor scikit-learn's `.csv.gz` files — rejected for
size and provenance questions. (b) Depend on `polars`/`datafusion` interop crates —
rejected: those are Phase 7 and not needed for synthetic data.

### Decision 5 — Plotting goes through an internal abstraction with a disabled `scienceview` backend

The crate defines a small internal plotting surface — line series, scatter, bar,
histogram, image grid, contour — as an internal trait with a data-oriented
implementation. Examples call it; nothing else in the crate touches a plotting library.

The `scienceview`-backed implementation lives in `src/plotting/scienceview_backend.rs`
behind a Cargo feature `plots`, **not** enabled by default; the `plots` feature implies
the `examples` feature, since the plotting abstraction exists only for the corpus. The crate therefore builds,
tests and runs today with no plotting dependency at all; when `scienceview` publishes a
usable API, the feature is enabled and the backend is filled in — a change confined to
one file plus one manifest line.

*Rationale.* `scienceview` is not published and has no data API, so any direct
dependency is premature. But the *shape* of the plotting surface must be decided now,
because every example's `plotting.rs` is written against it; getting it wrong means
touching 281 files later. The abstraction also serves the multi-language goal: the same
semantic operations (scatter, line, bar) exist in every target language's plotting
library, so the abstraction is the natural unit of parity between languages.

*Alternatives considered.* (a) Depend on `scienceview` from day one behind a feature
and write the backend against a guessed API — rejected: it would be code written against
a nonexistent contract, and the guessed API would likely be wrong. (b) Ship no plotting
at all and print tables — rejected: the gallery is visual; the plots are the deliverable.
(c) Vendor a minimal SVG writer — rejected: duplicates what `scienceview` is becoming.

### Decision 6 — Naming without abbreviations, except canonical mathematical symbols

Example ids spell words out: `ordinary_least_squares_and_ridge_regression`, not
`plot_ols_ridge`; `k_means_clustering_on_handwritten_digits`, not `plot_kmeans_digits`.
The single exception is the canonical mathematical symbol of the algorithm — `k` in
k-means, `t` in t-SNE, `p` in p-value — which is standard notation, not an abbreviation.
The porting guide lists every allowed token explicitly, so the rule is checkable.

Binaries are named exactly `<example_id>`.

*Rationale.* The repository's convention forbids abbreviations in names; scikit-learn's
example ids are historical and abbreviate heavily. Porting them verbatim would import
that debt. The mapping between the scikit-learn name and the sciencekit id is explicit
in the catalog, so nothing is lost.

### Decision 7 — Two documentation surfaces, cross-linked

```
docs/                                   (main book — API, usage, algorithms)
├── src/SUMMARY.md
│   ├── Project …                       (existing)
│   ├── Examples                        (NEW: the gallery, by algorithm family)
│   │   ├── General examples
│   │   ├── Generalized linear models
│   │   ├── Pipelines and composite estimators
│   │   ├── Preprocessing
│   │   └── … one page per ported example
│   └── Machine Learning with Sciencekit   (NEW: a link out to the separate book)
│         └── → docs-book-ml/

docs-book-ml/                           (NEW: the separate machine-learning book)
├── book.toml                           (own configuration and theme)
├── src/
│   ├── SUMMARY.md
│   └── ch01-…chNN-….md                 (one chapter per concept, in teaching order)
└── …                                   (its own Pages deployment)
```

Both books live in the **same repository**, on the **`docs/documentations` branch**, in
**sibling directories**: `docs/` for the main book and `docs-book-ml/` for the
machine-learning book. The separate book is exposed as a link from the main book's
summary, so a reader arriving at the API reference is one click away from the didactic
material.

The machine-learning book is **strictly separate** from the `Development Explained`
book: it teaches a *method* (why regularization behaves as it does), while
`Development Explained` narrates a *code change* (why this Rust is shaped this way). The
examples work is **exempt** from the `Development Explained` obligation — the
machine-learning book carries the didactic duty for the example corpus, and the two must
not duplicate each other's prose. The exemption is recorded in `AGENTS.md` as part of
this change.

Each gallery page states: what the example demonstrates, the dataset and its origin, the
estimators and the knob spectrum, the code (as language tabs), the expected output, the
run command, and a link to the chapter that teaches the concept. Each chapter teaches
the concept from first principles, walks the knob spectrum and its cause-and-effect, and
walks through the real example code and the real companion tests — the same didactic
contract the `Development Explained` book already imposes on the repository.

The tab widget is implemented in `docs/skin/custom.js` (and the ML book's own skin):
consecutive code blocks carrying the same group identifier become one tab group, with
the first language selected and keyboard navigation. It degrades to plain code blocks
without JavaScript.

*Alternatives considered.* (a) One book with two parts — rejected: the ML book has a
different audience, a different length and its own publishing cadence; a separate book
keeps the main book an API reference. (b) The gallery as the only documentation —
rejected: it shows *what* an algorithm does, not *why* a configuration produces that
behavior. (c) The machine-learning book in its own repository — rejected: the example
code it explains changes in the same repository, and the cross-references between a
chapter and the example it walks through must not drift across repositories.

### Decision 8 — Multi-language tabs are a separate, later change

The gallery pages are designed for tabs, and the catalog carries a `languages` field, but
the tab *content* for Python, Go, Julia, JVM and Lua is not written in this change. The
reason is sequencing, not effort: those bindings do not exist, so any snippet written now
would encode a guessed API and would have to be rewritten wholesale when the binding
lands — and the binding API is itself decided by separate changes.

The follow-up is recorded in `tasks.md` as a deferred item, to be created **after**:

1. the bindings API implementation (so the shape is real), and
2. this change's code (so there is Rust code to be equivalent to).

When it happens, the porting recipe extends: each language gets the same example id, the
same dataset, the same knob spectrum, and the idiomatic plotting library of that
language — `matplotlib` for Python, `gonum/plot` for Go, `Plots`/`Makie` for Julia, and
the closest supported option for JVM and Lua.

### Decision 9 — The corpus grows family by family, as algorithms land

The first wave covers what exists today: linear models, preprocessing, imputation and
metrics — roughly 60 gallery entries, of which the genuinely portable subset is the
linear-model family, the preprocessing family, the imputation family, the metrics-driven
model-selection entries and the pipeline entries.

Each subsequent wave adds one family, gated on the sciencekit crate it needs. The wave
table is a section of the catalog's companion document, and `blocked_by` in the catalog
is the machine-readable form of the same information.

### Decision 10 — Every example is deterministic and seeded explicitly

Every example declares a named seed constant next to its knob spectrum. Examples that
use stochastic estimators pass the seed explicitly. The determinism requirement is
asserted by the companion test, which runs the example pipeline twice and compares.

## Risks / Trade-offs

- **The catalog rots.** 281 records with manual fields will drift from reality.
  → Mitigation: the crate's tests validate the catalog against the filesystem (every
  `ported` record has a module and an example target; every module has a record) and the
  gallery index is generated from the catalog.

- **The dataset surface is a second implementation of `sklearn.datasets`.** Divergence
  from scikit-learn's semantics is possible and would silently change results.
  → Mitigation: the dataset module's tests assert the documented shapes and the
  determinism; the porting guide requires recording the scikit-learn semantics being
  mirrored.

- **The plotting abstraction may not match `scienceview`'s eventual API.**
  → Mitigation: the abstraction is deliberately small and data-oriented; if `scienceview`
  diverges, only `scienceview_backend.rs` changes — the 281 example modules call the
  abstraction, not the library.

- **The examples crate grows large.** Hundreds of example targets in one crate slow
  compilation when the feature is on.
  → Mitigation: the feature is off by default, so the corpus never enters the ordinary
  build or the default test run; the first wave's compile cost is measured during
  implementation and, if it becomes a problem, families can be split into
  `sciencekit_examples` plus per-family crates without changing the catalog.

- **The separate ML book is a second publishing pipeline.** A second mdBook needs its
  own Pages deployment.
  → Mitigation: the book is built and deployed by a workflow variant of the existing
  documentation workflow; the task list carries it explicitly.

- **Porting 281 entries is a multi-quarter effort.** Treating the corpus as one change
  guarantees it never lands.
  → Mitigation: this change delivers infrastructure plus the first wave; every later
  wave is one family, one change, one PR.

### Decision 11 — Every embedded dataset ships its provenance

The classic datasets bundled in the crate are third-party data (Fisher's iris, the UCI
digits and wine datasets, Efron's diabetes study). Each one ships a provenance record
next to it — `PROVENANCE.md` inside the dataset's own folder — naming its original
source, its license, its citation and the transformations applied when it was embedded.
A test asserts that every embedded dataset has one.

*Rationale.* The examples are published documentation; a reader who copies an example
must be able to know where the data came from and whether they may redistribute it. The
record also documents the transformations (column order, scaling, target encoding) that
make the embedded copy match the scikit-learn one, which is otherwise invisible.

*Alternatives considered.* (a) Generate everything synthetically — rejected: the classic
datasets carry the semantics the gallery is built on (the diabetes dataset is *the*
ridge example). (b) Provenance as a single file for the whole module — rejected: the
datasets have different sources and licenses; a per-dataset record is the only honest
form.

### Decision 12 — The gallery page is generated from the example's own sources

A gallery page is **never hand-written with code pasted into it**. The page generator
reads the example's source files and inlines them; the prose comes from the catalog
record and from the companion test's assertions. The Rust tab is therefore the actual
code that runs, and a language whose file does not exist yet simply has no tab.

```
crates/sciencekit_examples/src/examples/linear_model/
    ordinary_least_squares_and_ridge_regression/
        ├── model_configuration.rs   ──┐
        ├── metrics_report.rs        ──┼── inlined into the page's code blocks
        └── …_tests.rs               ──┘
                        │
                        ▼
        docs/src/examples/linear_model/ordinary_least_squares_and_ridge_regression.md
        ├── prose  ← catalog record (intent, dataset, knob spectrum, status)
        ├── code   ← generated from the files above
        └── output ← generated from the companion test's assertions
```

When the multi-language change lands, each language contributes its own file
(`bindings/python/examples/<id>.py`, `bindings/go/examples/<id>.go`, …) and the
generator picks them up as additional tabs — no page rewrite is needed, because the page
was never the source of the code.

Each page carries a **credit line** naming the scikit-learn example it adapts and the
catalog's source path, so the traceability is visible to the reader and not only to the
maintainers.

*Rationale.* Copy-pasted code in documentation is the classic drift generator: the
documentation compiles, the tests pass, and the page shows an API that no longer exists.
Generating the page from the sources removes the failure mode by construction. The
credit line is the honest counterpart of that: the *logic* is ported, the *provenance* is
declared.

*Alternatives considered.* (a) Hand-written pages — rejected: the drift above, plus 281
pages of duplicated prose to maintain. (b) mdBook's `{{#include}}` — considered, and it
covers the Rust case, but it cannot join several files into one tab group and cannot
generate the prose from the catalog; a generator outside mdBook serves both books
identically.

### Decision 13 — The catalog stays internal

The catalog is a development instrument: it drives the generator, the parity checks and
the wave plan. It is **not** published as a reader-facing page — the gallery shows only
the ported examples, and readers never see a progress bar of the port.

*Rationale.* A published parity report invites the wrong reading ("only 14% is
implemented") for a project whose algorithms are landing continuously, and it makes a
transient project state part of the public documentation. The same information is
available to maintainers through the generator's own output.

*Alternatives considered.* (a) Publishing the parity report — rejected for the reason
above. (b) Publishing only a per-family status line inside each family page — deferred:
it is cheap to add later and becomes meaningful only when the first families are
complete.

### Decision 14 — Pages are generated on the code branch and carried to the docs branch

The gallery pages and the corpus they show live on different branches: the corpus on
`main` and its feature branches, the books on `docs/documentations`. The bridge is the
generator.

```
code branch (main / feat/…)                 docs branch (docs/documentations)
┌──────────────────────────────┐            ┌───────────────────────────────┐
│ crates/sciencekit_examples/  │            │ docs/src/examples/…md         │
│   src/examples/<id>/…        │            │   ← generated markdown,      │
│   catalog/catalog.json       │            │     committed by the docs PR  │
│            │                 │            │                               │
│            │ cargo run -p    │            │ docs-book-ml/…                 │
│            │ sciencekit_     │            │   ← generated the same way    │
│            │ examples --bin  │            │                               │
│            │ gallery-generator            └───────────────────────────────┘
└────────────┴────────────────┘
```

The generator runs on the **code** branch and writes the pages; the docs PR carries the
generated markdown. Because the generator is a pure function of the corpus, the docs
branch can verify freshness: a CI job on `docs/documentations` checks out the corpus at
the pinned revision, regenerates the pages, and fails if the result differs from what is
committed. Staleness is therefore detected rather than assumed away.

*Rationale.* Generating at docs-build time would always be fresh, but it makes the book
build depend on the network and on a pinned revision of another branch — a documentation
build should be reproducible from the repository alone. Committing the generated pages
keeps the book build hermetic and turns "the page shows code that no longer exists"
from an invisible defect into a red CI job.

*Alternatives considered.* (a) Generate during the docs build — rejected: network
dependency and a pinned cross-branch revision in the book build. (b) Hand-copy the code
into the pages — rejected: the drift this change exists to eliminate. (c) Move the
corpus to the docs branch — rejected: library code does not live on a documentation
branch.

### Decision 15 — The example target manifest block is derived, not hand-written

Cargo has no glob for targets, so every example needs an `[[example]]` entry — 281 of
them. Hand-maintaining them is a drift source, so the block is **derived**: a generator
rewrites the `[[example]]` section of the manifest between marker comments from the
`examples/` directory, and a test asserts that the committed block matches the directory.

```
crates/sciencekit_examples/Cargo.toml
   …
   # --- BEGIN GENERATED EXAMPLE TARGETS — do not edit by hand ---
   [[example]]
   name = "ordinary_least_squares_and_ridge_regression"
   path = "examples/ordinary_least_squares_and_ridge_regression/main.rs"
   required-features = ["examples"]
   # --- END GENERATED EXAMPLE TARGETS ---
```

*Rationale.* The manifest is the one place where the corpus's shape is duplicated. Any
duplication of that kind eventually disagrees with the truth. Deriving it turns a class
of silent defects into a failing test.

*Alternatives considered.* (a) Hand-written entries — rejected: 281 lines to maintain by
hand. (b) No `[[example]]` entries at all, relying on Cargo's auto-discovery — rejected:
without `required-features` the targets would be compiled by every `cargo test`.

### Decision 16 — The generator is a tool in the examples crate

The page generator is a **tool binary** in the unpublished crate
(`cargo run -p sciencekit_examples --bin gallery-generator`), not an example. It needs
the catalog and the example headers as a library, so it belongs next to them.

*Rationale.* A tool binary is not an example binary: it is not part of the corpus, it is
not part of the gallery, and it is not published. Keeping it in the same crate gives it
access to the corpus's types without exporting them.

*Alternatives considered.* (a) A separate `sciencekit_tools` crate — deferred: one tool
does not justify a crate, and it can move later. (b) A script in the docs branch — it
would have to re-implement the catalog reading and the header parsing in another
language. (c) An mdBook preprocessor — rejected: it cannot write files outside the book
directory and would not produce the machine-learning book's pages.

## Migration Plan

Nothing to migrate: the change adds a new crate, new documentation surfaces and a new
catalog. The rollback is a deletion of the crate and its book.

Ordering inside the change:

1. Crate scaffold, catalog schema, dataset surface, plotting abstraction (no example
   yet) — everything compiles and passes its tests.
2. The porting guide, the catalog's first records, and the first family of examples
   (linear models) end to end: catalog record → module → test → target → gallery page →
   book chapter.
3. The remaining families of the first wave, each following the same path.

## Open Questions

- **The title of the separate book.** "Machine Learning with Sciencekit" is the working
  title; the final title is a documentation decision.
- **The exact set of omitted gallery entries.** 13 release-highlight entries, the
  developer-facing entries (frozen estimators, callbacks, metadata routing, display
  objects, `is_fitted`), and the text-corpus entries are candidates for `omitted`. The
  final list is a catalog decision taken per family and does not affect the schema.
